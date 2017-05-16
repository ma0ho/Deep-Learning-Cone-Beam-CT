import tensorflow as tf
import numpy as np
import os
from tfcone.inout import dennerlein, projtable
from tfcone.algo import ct

DATA_P = os.path.abspath(
        os.path.dirname( os.path.abspath( __file__ ) ) + '/../phantoms/conrad-2/'
    ) + '/'

# gives CHW
proj = dennerlein.read( DATA_P + 'shepp-logan-proj-filtered.bin' )

geom = projtable.read( DATA_P + 'projMat.txt' )
volume_shape = [200, 200, 200]
volume_origin = tf.contrib.util.make_tensor_proto( [-99.5, -99.5, -99.5],
        tf.float32 )
volume = ct.backproject( projections=proj, geom=geom, vol_shape=volume_shape,
        vol_origin=volume_origin)

write_op = dennerlein.write( '/tmp/test.bin', volume )


with tf.Session() as sess:
    v, _, p, g = sess.run( [volume, write_op, proj, geom] )
#    print( np.sum(v) )
#    print(g)


