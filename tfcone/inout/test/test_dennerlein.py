import tensorflow as tf
from tfcone.io import dennerlein as dl

def test_read_write32():
    if tf.gfile.Exists( 'test.bin' ):
        tf.gfile.Remove( 'test.bin' )

    v = tf.random_normal( [3,4,6] )
    w = dl.write( 'test.bin', v )

    with tf.control_dependencies( [w] ):
        r = dl.read( 'test.bin' )
        c = tf.abs( tf.reduce_sum( v-r ) )
        a = tf.assert_less( c, [10e-4] )
        sess = tf.Session()
        sess.run( a )

def test_read_write64():
    if tf.gfile.Exists( 'test.bin' ):
        tf.gfile.Remove( 'test.bin' )

    v = tf.random_normal( [3,4,6], dtype=tf.float64 )
    w = dl.write( 'test.bin', v )

    with tf.control_dependencies( [w] ):
        r = dl.read( 'test.bin', dtype=tf.float64 )
        c = tf.abs( tf.reduce_sum( v-r ) )
        a = tf.assert_less( c, [10e-4] )
        sess = tf.Session()
        sess.run( a )

test_read_write32()
test_read_write64()
