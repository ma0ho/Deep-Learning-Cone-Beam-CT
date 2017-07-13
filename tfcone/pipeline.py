import tensorflow as tf
import numpy as np
import os
import copy
from tfcone.inout import dennerlein, projtable
from tfcone.algo import ct
from tfcone.util import types as t
from tfcone.util import plot as plt
from tensorflow.python.client import timeline
from tfcone.inout import png

# CONFIG
#-------------------------------------------------------------------------------------------

# GEOMETRY
RAMLAK_WIDTH        = 401
RAMLAK_WIDTH        = 51
VOL_SHAPE           = t.Shape3D(
                        W = 512, H = 512, D = 512
                    )
VOL_ORIG            = t.Coord3D(
                        X = -170, Y = -170, Z = -255
                    )
VOXEL_DIMS          = t.Shape3D(
                        W = 0.664, H = 0.664, D = 1
                    )
# NOTE: See todo at ct.init_ramlak_1D
PIXEL_DIMS          = t.Shape2D(
                    W = 1, H = 1
)
SOURCE_DET_DISTANCE = 1200
PROJ_SHAPE          = t.ShapeProj(
                        N = 360, W = 720, H = 880
                    )

# DATA HANDLING
DATA_P = os.path.abspath(
        os.path.dirname( os.path.abspath( __file__ ) ) + '/../phantoms/L067/'
    ) + '/'

# TRAINING CONFIG
LIMITED_ANGLE_SIZE  = 180   # #projections for limited angle
LEARNING_RATE       = 0.00001

# GPU RELATED STAFF
GPU_FRACTION        = .75
SAVE_GPU_MEM        = True


# SOME SETUP
#-------------------------------------------------------------------------------------------
CONF = ct.ReconstructionConfiguration(
        PROJ_SHAPE,
        VOL_SHAPE,
        VOL_ORIG,
        VOXEL_DIMS,
        PIXEL_DIMS,
        SOURCE_DET_DISTANCE,
        RAMLAK_WIDTH
)
CONF_LA = copy.deepcopy( CONF )
CONF_LA.proj_shape.N = LIMITED_ANGLE_SIZE


# GLOBALS
#-------------------------------------------------------------------------------------------
asserts = []


# READ DATA
#-------------------------------------------------------------------------------------------
proj = dennerlein.read( DATA_P + 'proj.bin' )
geom, angles = projtable.read( DATA_P + 'projMat.txt' )
proj_shape = tf.shape( proj )
with tf.control_dependencies( [ proj ] ):
    asserts.append( tf.assert_equal( tf.shape( proj ), [ CONF.proj_shape.N,
        CONF.proj_shape.H, CONF.proj_shape.W ] ) )


# RECONSTRUCT REFERENCE
#-------------------------------------------------------------------------------------------
reconstructor = ct.Reconstructor( CONF, angles, name = 'RefReconstructor' )
volume = reconstructor.apply( proj, geom, fullscan = True )
# copy back to cpu
with tf.device("/cpu:0"):
    volume = tf.Variable( volume, trainable = False, name = 'ref_volume' )


# RECONSTRUCT LIMITED ANGLE
#-------------------------------------------------------------------------------------------
la_reconstructor = ct.Reconstructor(
        CONF_LA, angles[0:LIMITED_ANGLE_SIZE],
        trainable = True,
        name = 'LAReconstructor'
)

#def test_grad( i ):
#    proj_la = tf.slice( proj, [ i, 0, 0 ], [ LIMITED_ANGLE_SIZE, -1, -1 ] )
#    geom_la = geom[i:i+LIMITED_ANGLE_SIZE]
#    volume_la = la_reconstructor.apply( proj_la, geom_la )
#    with tf.device("/cpu:0"):
#        loss = tf.losses.mean_squared_error( volume, volume_la )
#    grad = tf.gradients(loss,la_reconstructor.parker_w)[0]
#    #return dennerlein.write( '/tmp/test.bin', grad )
#    return tf.Print( grad, [grad], summarize = 300 )


# OPTIMIZATION
#-------------------------------------------------------------------------------------------
opt = tf.train.AdamOptimizer( LEARNING_RATE )

def train_step( i, filename = None ):
    proj_la = tf.slice( proj, [ i, 0, 0 ], [ LIMITED_ANGLE_SIZE, -1, -1 ] )
    geom_la = geom[i:i+LIMITED_ANGLE_SIZE]
    volume_la = la_reconstructor.apply( proj_la, geom_la )

    # eventually export central slice as png
    sl = volume_la[ int( VOL_SHAPE.D/2 ) ]
    writeop = [ png.writeSlice( sl, filename ) ] if filename != None else None

    with tf.control_dependencies( writeop ):
        # compute loss on cpu (avoid too many volume instances on gpu)
        with tf.device("/cpu:0"):
            loss = tf.losses.mean_squared_error( volume, volume_la )

        minop = opt.minimize( loss, colocate_gradients_with_ops = True )

    return minop, loss, la_reconstructor.parker_w


# WRITE RESULT
#-------------------------------------------------------------------------------------------
#write_op = dennerlein.write( '/tmp/test.bin', volume )


# LAUNCH
#-------------------------------------------------------------------------------------------
gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = GPU_FRACTION )

with tf.Session( config = tf.ConfigProto( gpu_options = gpu_options ) ) as sess:
    # make sure that graph is complete..
    train_step(0)

    sess.run( tf.global_variables_initializer() )

    # print trainable vars
    #variables_names = [v.name for v in tf.trainable_variables()]
    #values = sess.run(variables_names)
    #for k, v in zip(variables_names, values):
    #    print( "Variable: ", k )


    # tracing according to http://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # test
    #sess.run( [ write_op, asserts ] )

    for i in range( 0, CONF.proj_shape.N - LIMITED_ANGLE_SIZE ):
        print( "Iteration %d" % i )

        if i == 0:
            minop, lop, parker = train_step( i, "first.png" )
        elif i == CONF.proj_shape.N - LIMITED_ANGLE_SIZE - 1:
            minop, lop, parker = train_step( i, "last.png" )
        else:
            minop, lop, parker = train_step( i )

        _, loss, parker_np, _ = sess.run( [ minop, lop, parker, asserts ], options = run_options, run_metadata = run_metadata )

        print( "Loss: %f" % loss )
        #plt.plot_parker( pw_np, "parker%d.png" % i )

    # write timeline object to file
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format( show_memory=True )
    with open('timeline.json', 'w') as f:
        f.write(ctf)


