import tensorflow as tf
import os
import math
import numpy as np
import tfcone.util.numerical as nm

_path = os.path.dirname(os.path.abspath(__file__))
_write_module = tf.load_op_library( _path + '/../../user-ops/backproject.so' )
backproject = _write_module.backproject

'''
    generate 1D-RamLak filter according to Kak & Slaney, chapter 3 equation 61
'''
def init_ramlak_1D( width, pixel_width_mm ):
    assert( width % 2 == 1 )

    hw = int( ( width-1 ) / 2 )
    f = [
            -1 / math.pow( i * math.pi * pixel_width_mm, 2 ) if i%2 == 1 else 0
            for i in range( -hw, hw+1 )
        ]
    f[hw] = 1/4 * math.pow( pixel_width_mm, 2 )

    return f


'''
    Generate 1D parker row-weights

    beta
        projection angle in [0, pi + 2*delta]
    delta
        overscan angle

'''
def init_parker_1D( beta, source_det_dist_mm, U, pixel_width_mm, delta ):
    assert( beta + nm.eps >= 0 and beta - nm.eps <= math.pi + 2*delta )

    if( beta <= 0.0 ):
        beta = nm.eps

    w = np.ones( ( U ), dtype = np.float )

    for u in range( 0, U ):
        alpha = math.atan( ( u+0.5 - U/2 ) * pixel_width_mm / source_det_dist_mm )

        if beta <= 2*delta - 2*alpha:
            # begin of scan
            w[u] = math.pow( math.sin( math.pi/4 * ( beta / (delta-alpha) ) ), 2 )

        elif beta >= math.pi - 2*alpha:
            # end of scan
            w[u] = math.pow( math.sin( math.pi/4 * ( ( math.pi + 2*delta - beta ) / ( delta + alpha ) ) ), 2 )

    return w


'''
    Generate 3D volume of parker weights

    U
        detector width

    returns
        numpy array of shape [#projections, 1, U]
'''
def init_parker_3D( primary_angles_rad, source_det_dist_mm, U, pixel_width_mm ):
    pa = primary_angles_rad

    # normalize angles to [0, 2*pi]
    pa -= pa[0]
    pa = np.where( pa < 0, pa + 2*math.pi, pa )

    # find rotation such that max(angles) is minimal
    tmp = np.reshape( pa, ( pa.size, 1 ) ) - pa
    tmp = np.where( tmp < 0, tmp + 2*math.pi, tmp )
    pa = tmp[:, np.argmin( np.max( tmp, 0 ) )]

    # according to conrad implementation
    delta = math.atan( float(U * pixel_width_mm) / 2 / source_det_dist_mm )

    # go over projections
    w = [
            np.reshape(
                init_parker_1D( pa[i], source_det_dist_mm, U, pixel_width_mm, delta ),
                ( 1, 1, U )
            )
            for i in range( 0, pa.size )
        ]

    return np.concatenate( w )


'''
    Generate 3D volume of cosine weights

    U
        detector width
    V
        detector height

    returns
        numpy array of shape [1, V, U]

'''
def init_cosine_3D( source_det_dist_mm, U, V, pixel_width_mm, pixel_height_mm ):
    cu = U/2 * pixel_width_mm
    cv = V/2 * pixel_height_mm
    sd2 = source_det_dist_mm**2

    w = np.zeros( ( 1, V, U ), dtype = np.float )

    for v in range( 0, V ):
        dv = ( (v+0.5) * pixel_height_mm - cv )**2
        for u in range( 0, U ):
            du = ( (u+0.5) * pixel_width_mm - cu )**2
            w[0,v,u] = source_det_dist_mm / math.sqrt( sd2 + dv + dv )

    return w


