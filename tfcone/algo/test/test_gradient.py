import tensorflow as tf
from tfcone.algo import ct
import numpy as np

class GradTest( tf.test.TestCase ):

    def testGrad( self ):
        proj = tf.random_normal( [5,10,10] )
        m = np.array( [ [ 1.2, 0, 0, 0 ],
                        [ 0, 0.75, 0, 0 ],
                        [ 0, 0.2, 0, 0.8 ] ] )
        m = m.reshape( [1,3,4] )
        m = np.concatenate( (m,m,m,m,m) )

        geom_proto = tf.contrib.util.make_tensor_proto( m, tf.float32 )
        vol_origin_proto = tf.contrib.util.make_tensor_proto( [0,0,0], tf.float32 )
        voxel_dimen_proto = tf.contrib.util.make_tensor_proto( [1,1,1], tf.float32 )
        vol = ct.backproject(
                projections = proj,
                geom        = geom_proto,
                vol_shape   = [10,10,10],
                vol_origin  = vol_origin_proto,
                voxel_dimen = voxel_dimen_proto,
                proj_shape  = [5,10,10],
                name        = 'backproject'
            )
        vol = tf.reduce_sum( vol )

        with self.test_session():
            ag, ng = tf.test.compute_gradient( proj, [5,10,10],
                    vol, [1], delta = 0.001 )

            self.assertTrue( ( np.abs( ag-ng ) < .5 ).all() )

        #r = np.concatenate(
        #        (np.arange(0,500).reshape((500,1)),ag,ng,np.abs(ag-ng)), axis = 1 )
        #np.set_printoptions( threshold = 10000000, suppress = True )
        #print( r )


if __name__ == '__main__':
    tf.test.main()


