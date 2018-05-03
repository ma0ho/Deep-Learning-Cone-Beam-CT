import tensorflow as tf
import os

'''
    Reads the given filename and returns its contents as a tensor. The shape of
    the returned tensor depends on the dennerlein header.

    shape
        Optional argument that is set as fixed shape
'''
def read( queue, name = None, shape = None ):
    HEADER_LENGTH = 6
    DATA_SIZE = 4

    with tf.name_scope( name, "DennerleinReader", [ queue, name ] ) as scope:
        reader = tf.WholeFileReader()

        _, raw_value = reader.read( queue )
        byte_value = tf.decode_raw( raw_value, tf.uint8, little_endian = True )

        # extract header
        header = tf.bitcast( tf.reshape(
                tf.slice( byte_value, [0], [HEADER_LENGTH] ),
                [3, 2]
            ), tf.uint16 )

        # extract data
        header = tf.cast( header, tf.int32 )
        x, y, z = tf.split( header, [1, 1, 1] )
        data_length = x*y*z*DATA_SIZE

        data = tf.bitcast( tf.reshape(
                tf.slice( byte_value, [HEADER_LENGTH], data_length ),
                tf.concat( [ x*y*z, [4]], 0 )
            ), tf.float32 )

        data = tf.reshape( data, tf.concat( [ z, y, x ], 0 ), name = scope )

        if shape:
            data.set_shape( shape )

        return data


'''
    Reads the given filename and returns its contents as a tensor. The shape of
    the returned tensor depends on the dennerlein header.
'''
def read_noqueue( fn ):
    HEADER_LENGTH = 6
    DATA_SIZE = 4

    raw_value = tf.read_file( fn )
    byte_value = tf.decode_raw( raw_value, tf.uint8, little_endian = True )

    # extract header
    header = tf.bitcast( tf.reshape(
            tf.slice( byte_value, [0], [HEADER_LENGTH] ),
            [3, 2]
        ), tf.uint16 )

    # extract data
    header = tf.cast( header, tf.int32 )
    x, y, z = tf.split( header, [1, 1, 1] )
    data_length = x*y*z*DATA_SIZE

    data = tf.bitcast( tf.reshape(
            tf.slice( byte_value, [HEADER_LENGTH], data_length ),
            tf.concat( [ x*y*z, [4]], 0 )
        ), tf.float32 )

    data = tf.reshape( data, tf.concat( [ z, y, x ], 0 ) )

    return data

_path = os.path.dirname(os.path.abspath(__file__))
_write_module = tf.load_op_library( _path + '/../../lib/libwrite_dennerlein.so' )
write = _write_module.write_dennerlein


