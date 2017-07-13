import tensorflow as tf

'''
    Export a single slice as png
'''
def writeSlice( sl, name ):
    sl = sl * ( 254 / tf.reduce_max( sl ) )
    sl = tf.cast( sl, tf.uint8 )
    shape = tf.shape( sl )
    png = tf.image.encode_png( tf.reshape( sl, [ shape[0], shape[1], 1 ] ) )
    fname = tf.constant( name )
    return tf.write_file( fname, png )


