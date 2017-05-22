import tensorflow as tf
import re
import tfcone.util.numerical as nm
import numpy as np
import math

def read( filename, data_type = tf.float32 ):
    assert( data_type == tf.float32 or data_type == tf.float64 )

    file_handle = open( filename, 'r' )
    file_contents = file_handle.read()
    file_handle.close()

    proj = []
    angles = []
    regex = re.compile( r"@\s\d*\n([\d.]+)\s+([\d.]+)\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+" )
    for match in regex.finditer( file_contents ):
        d = [ float( x ) for x in match.groups() ]
        angles += d[0:2]
        proj += d[2:]

    assert( len(proj) % 12 == 0 )

    angles = np.array( angles, dtype = np.float )
    angles = np.reshape( angles, ( int( len(angles)/2 ), 2 ) )
    angles_sum = np.sum( angles, 0 )
    angles_i = np.argmax( angles_sum )

    # rotation axis needs to be parallel to world coordinate system
    assert( angles_sum[ (angles_i+1)%2 ] < nm.eps )

    proj = np.reshape( proj, ( int( len(proj)/12 ), 3, 4 ) )

    # convert to radians
    angles = angles[:,angles_i] * (math.pi/180)

    return proj, angles


