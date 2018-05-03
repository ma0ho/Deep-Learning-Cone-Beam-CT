import tensorflow as tf
from tensorflow.python.framework import ops
import os
import math
import numpy as np
import util.numerical as nm
import util.types as t
from inout import dennerlein
import sys

_path = os.path.dirname(os.path.abspath(__file__))
_bp_module = tf.load_op_library( _path + '/../../lib/libbackproject.so' )
backproject = _bp_module.backproject
project = _bp_module.project


'''
    Compute the gradient of the backprojection op
    by invoking the forward projector.
'''
@ops.RegisterGradient( "Backproject" )
def _backproject_grad( op, grad ):
    proj = project(
            volume      = grad,
            geom        = op.get_attr( "geom" ),
            vol_shape   = op.get_attr( "vol_shape" ),
            vol_origin  = op.get_attr( "vol_origin" ),
            voxel_dimen = op.get_attr( "voxel_dimen" ),
            proj_shape  = op.get_attr( "proj_shape" )
        )
    return [ proj ]


'''
    Compute the gradient of the forward projection op
    by invoking the backprojector.
'''
@ops.RegisterGradient( "Project" )
def _project_grad( op, grad ):
    vol = backproject(
            proj        = grad,
            geom        = op.get_attr( "geom" ),
            vol_shape   = op.get_attr( "vol_shape" ),
            vol_origin  = op.get_attr( "vol_origin" ),
            voxel_dimen = op.get_attr( "voxel_dimen" ),
            proj_shape  = op.get_attr( "proj_shape" )
        )
    return [ vol ]


'''
    generate 1D-RamLak filter according to Kak & Slaney, chapter 3 equation 61

    TODO:   Does not work for example for pixel_width_mm = 0.5. Then we have
            a negative filter response.. Whats wrong here?

    Note: Conrad implements a slightly different variant, that's why results
    differ in the absolute voxel intensities
'''
def init_ramlak_1D( config ):
    assert( config.ramlak_width % 2 == 1 )

    hw = int( ( config.ramlak_width-1 ) / 2 )
    f = [
            -1 / math.pow( i * math.pi * config.pixel_shape.W, 2 ) if i%2 == 1 else 0
            for i in range( -hw, hw+1 )
        ]
    f[hw] = 1/4 * math.pow( config.pixel_shape.W, 2 )

    return f


'''
    Generate 1D parker row-weights

    beta
        projection angle in [0, pi + 2*delta]
    delta
        maximum fan angle
        => redundancy occurs in [pi, pi+2*delta]

'''
def init_parker_1D( config, beta, delta ):
    assert( beta + nm.eps >= 0 )

    w = np.ones( ( config.proj_shape.W ), dtype = np.float32 )

    for u in range( 0, config.proj_shape.W ):
        # current fan angle
        alpha = math.atan( ( u+0.5 - config.proj_shape.W/2 ) *
                config.pixel_shape.W / config.source_det_distance )

        if beta >= 0 and beta < 2 * (delta+alpha):
            # begin of scan
            w[u] = math.pow( math.sin( math.pi/4 * ( beta / (delta+alpha) ) ), 2 )
        elif beta >= math.pi + 2*alpha and beta < math.pi + 2*delta:
            # end of scan
            w[u] = math.pow( math.sin( math.pi/4 * ( ( math.pi + 2*delta - beta
                ) / ( delta - alpha ) ) ), 2 )
        elif beta >= math.pi + 2*delta:
            # out of range
            w[u] = 0.0

    return w

def init_riess_1D( config, beta, delta ):
    projW = config.proj_shape.W
    pixW = config.pixel_shape.W
    sdDist = config.source_det_distance

    w = np.zeros( ( projW ), dtype = np.float32 )

    # overscan angle
    # TODO: incorrect of for less than short scan
    over = 2*delta

    def w1(b, a):
        x = math.pi + over - b
        y = over - 2*a
        z = math.pi/2 * (x/y)
        return math.pow(math.sin(z), 2)

    def w2(b, a):
        x = b
        y = over + 2*a
        z = math.pi/2 * (x/y)
        return math.pow(math.sin(z), 2)

    for u in range( 0, projW ):

        # fan angle corresponding to u
        alpha = math.atan( ( u+0.5 - projW/2 ) *
                pixW / sdDist )

        if math.pi + 2*delta <= beta and beta <= math.pi + over:
            w[u] = w1(beta, alpha)
        elif math.pi + 2*over - 2*delta <= beta and beta <= math.pi + over:
            w[u] = 2 - w1(beta, alpha)
        elif 0 <= beta and beta <= 2*delta + over:
            w[u] = w2(beta, alpha)
        elif 0 <= beta and beta <= -over - 2*delta:
            w[u] = 2 - w2(beta, alpha)
        else:
            w[u] = 1

    return w

def init_schaefer_1D( config, beta, delta ):
    projW = config.proj_shape.W
    pixW = config.pixel_shape.W
    sdDist = config.source_det_distance

    w = np.zeros( ( projW ), dtype = np.float32 )

    for u in range( 0, projW ):

        # fan angle corresponding to u
        alpha = math.atan( ( u+0.5 - projW/2 ) *
                pixW / sdDist )

        if beta >= math.pi + 2*alpha and beta < math.pi + delta:
            # end of scan
            w[u] = math.pow( math.sin( math.pi/2 * ( ( math.pi + 2*delta - beta
                ) / ( delta - 2*alpha ) ) ), 2 )
        elif beta >= math.pi + 2*delta - 2*alpha and beta <= math.pi + delta:
            w[u] = 2- ( math.pow( math.sin( math.pi/2 * ( ( math.pi + 2*delta - beta
                ) / ( delta - 2*alpha ) ) ), 2 ) )
        elif beta >= 0 and beta <= 2*alpha + delta:
            # begin of scan
            w[u] = math.pow( math.sin( math.pi/2 * ( beta / (delta + 2*alpha) ) ), 2 )
        elif beta >= 0 and beta <= -delta - 2*alpha:
            w[u] = 2- ( math.pow( math.sin( math.pi/2 * ( beta / (delta +
                2*alpha) ) ), 2 ) )
        else:
            # out of range
            w[u] = 1

    return w

'''
    Generate 3D volume of parker weights

    U
        detector width

    returns
        numpy array of shape [#projections, 1, U]
'''
def init_redundancy_3D( config, primary_angles_rad, weights_type = 'parker' ):
    pa = primary_angles_rad

    # normalize angles to [0, 2*pi]
    pa -= pa[0]
    pa = np.where( pa < 0, pa + 2*math.pi, pa )

    # find rotation such that max(angles) is minimal
    tmp = np.reshape( pa, ( pa.size, 1 ) ) - pa
    tmp = np.where( tmp < 0, tmp + 2*math.pi, tmp )
    pa = tmp[:, np.argmin( np.max( tmp, 0 ) )]

    # delta = maximum fan_angle
    delta = math.atan( ( float(config.proj_shape.W * config.pixel_shape.W) / 2 )
            / config.source_det_distance )

    if weights_type == 'parker':
        f = lambda pi: init_parker_1D( config, pi, delta )
    elif weights_type == 'riess':
        f = lambda pi: init_riess_1D( config, pi, delta )
    elif weights_type == 'schaefer':
        f = lambda pi: init_schaefer_1D( config, pi, delta )

    # go over projections
    w = [
            np.reshape(
                f( pa[i] ),
                ( 1, 1, config.proj_shape.W )
            )
            for i in range( 0, pa.size )
        ]

    w = np.concatenate( w )

    return w

'''
    Generate 3D volume of cosine weights

    U
        detector width
    V
        detector height

    returns
        numpy array of shape [1, V, U]

'''
def init_cosine_3D( config ):
    cu = config.proj_shape.W/2 * config.pixel_shape.W
    cv = config.proj_shape.H/2 * config.pixel_shape.H
    sd2 = config.source_det_distance**2

    w = np.zeros( ( 1, config.proj_shape.H, config.proj_shape.W ), dtype =
            np.float32 )

    for v in range( 0, config.proj_shape.H ):
        dv = ( (v+0.5) * config.pixel_shape.H - cv )**2
        for u in range( 0, config.proj_shape.W ):
            du = ( (u+0.5) * config.pixel_shape.W - cu )**2
            w[0,v,u] = config.source_det_distance / math.sqrt( sd2 + dv + dv )

    return w


class ReconstructionConfiguration:

    '''
        proj_shape
            instance of ProjShape
        vol_shape
            shape of volume (instance of 3DShape)
        vol_origin
            volume origin in world coords (instance of 3DCoord)
        voxel_shape
            size of a voxel in mm (instance of 3DShape)
        pixel_shape
            size of a detector pixel in mm (instance of 2DShape)
        source_det_distance
            in mm
        ramlak_width
            in pixel
    '''
    def __init__(
            self,
            proj_shape,
            vol_shape,
            vol_origin,
            voxel_shape,
            pixel_shape,
            source_det_distance,
            ramlak_width
    ):
        assert( type( proj_shape ) is t.ShapeProj )
        assert( type( vol_shape ) is t.Shape3D )
        assert( type( vol_origin ) is t.Coord3D )
        assert( type( voxel_shape ) is t.Shape3D )
        assert( type( pixel_shape ) is t.Shape2D )

        self.proj_shape             = proj_shape
        self.vol_shape              = vol_shape
        self.vol_origin             = vol_origin
        self.voxel_shape            = voxel_shape
        self.pixel_shape            = pixel_shape
        self.source_det_distance    = source_det_distance
        self.ramlak_width           = ramlak_width


class Reconstructor:

    def __init__( self, config, angles, trainable = False, name = None,
            weights_type = 'parker' ):
        self.config = config
        self.trainable = trainable
        self.name = name

        with tf.name_scope( self.name, "Reconstruct" ) as scope:
            with tf.variable_scope( self.name, "Reconstruct" ):

                # init cosine weights
                cosine_w_np = init_cosine_3D( config )
                self.cosine_w = tf.Variable(
                        initial_value = cosine_w_np,
                        dtype = tf.float32,
                        name = 'cosine-weights',
                        trainable = False
                )

                # init redundancy weights
                # NOTE: Current configuration assumes that relative angles
                #       remain valid even if apply is invoked with different
                #       projection matrices!
                self.parker_w_np = init_redundancy_3D( self.config, angles,
                        weights_type )
                self.parker_w = tf.Variable(
                        initial_value = self.parker_w_np,
                        dtype = tf.float32,
                        name = 'parker-weights',
                        trainable = self.trainable
                )

                # init ramlak
                ramlak_1D = init_ramlak_1D( config )
                self.kernel = tf.Variable(
                        initial_value = ramlak_1D,
                        dtype = np.float32,
                        name = 'ramlak-weights',
                        trainable = False
                )
                self.kernel = tf.reshape( self.kernel, [ 1, self.config.ramlak_width, 1, 1 ] )

                # initializations for backprojection op
                self.vol_origin_proto = tf.contrib.util.make_tensor_proto(
                        config.vol_origin.toNCHW(), tf.float32 )
                self.voxel_dimen_proto = tf.contrib.util.make_tensor_proto(
                        config.voxel_shape.toNCHW(), tf.float32 )


    '''
        Reset all trainable vars
    '''
    def reset():
        with tf.name_scope( self.name, "Reconstruct" ) as scope:
            with tf.variable_scope( self.name, "Reconstruct" ):
                self.parker_w = tf.Variable(
                        initial_value = self.parker_w_np,
                        dtype = tf.float32,
                        name = 'parker-weights',
                        trainable = self.trainable
                )


    def save_vars():
        return [ self.parker_w ]

    '''
        proj
            the sinogram
        geom
            stack of projection matrices

        returns
            volume tensor
    '''
    def apply( self, proj, geom, fullscan = False ):
        with tf.name_scope( self.name, "Reconstruct", [ proj, geom ] ) as scope:
            with tf.variable_scope( self.name, "Reconstruct", [ proj, geom ] ):

                # COSINE
                proj = tf.multiply( proj, self.cosine_w, name = 'cosine-weighting' )

                # PARKER
                if not fullscan:
                    proj = tf.multiply( proj, self.parker_w, name = 'parker-weighting' )

                # RAMLAK
                s = self.config.proj_shape
                proj = tf.reshape( proj, [ s.N, 1, s.H, s.W ] )
#                proj = tf.nn.conv2d(
#                        input = proj,
#                        filter = self.kernel,
#                        strides = [ 1, 1, 1, 1 ],
#                        padding = 'SAME',
#                        data_format = 'NCHW',
#                        name = 'ramlak-filter'
#                )

                # TODO: Hack! Remove (and uncomment above) if
                # https://github.com/tensorflow/tensorflow/issues/11327 is
                # resolved
                N = self.config.proj_shape.N
                H = self.config.proj_shape.H
                W = self.config.proj_shape.W

                proja = []

                for i in range(0,9):
                    p = tf.slice( proj, [int(i*(N/9)),0,0,0], [int(N/9),1,H,W] )
                    p = tf.nn.conv2d(
                            input = p,
                            filter = self.kernel,
                            strides = [ 1, 1, 1, 1 ],
                            padding = 'SAME',
                            data_format = 'NCHW',
                            name = 'ramlak-filter'
                    )
                    proja.append( p )

                proj = tf.concat( [p for p in proja], 0 )

                proj = tf.reshape( proj, s.toNCHW() )

                # BACKPROJECTION
                geom_proto = tf.contrib.util.make_tensor_proto( geom, tf.float32 )
                vol = backproject(
                        projections = proj,
                        geom        = geom_proto,
                        vol_shape   = self.config.vol_shape.toNCHW(),
                        vol_origin  = self.vol_origin_proto,
                        voxel_dimen = self.voxel_dimen_proto,
                        proj_shape  = self.config.proj_shape.toNCHW(),
                        name        = 'backproject'
                    )

                self.pin = proj

                if fullscan:
                    vol /= 2

                return tf.nn.relu( vol, scope )


