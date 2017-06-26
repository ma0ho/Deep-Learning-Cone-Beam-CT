import tensorflow as tf
import numpy as np
import os
from tfcone.inout import dennerlein, projtable
from tfcone.algo import ct
from tensorflow.python.client import timeline

# CONFIG
#-------------------------------------------------------------------------------------------
DATA_P = os.path.abspath(
        os.path.dirname( os.path.abspath( __file__ ) ) + '/../phantoms/conrad-4/'
    ) + '/'

RAMLAK_WIDTH = 401
VOLUME_SHAPE_VX = [ 600, 800, 800 ]
VOLUME_ORIGIN_MM = [ -150, -200, -200 ]
VOXEL_WIDTH_MM = 0.5
VOXEL_HEIGHT_MM = 0.5
VOXEL_DEPTH_MM = 0.5
# NOTE: See todo at ct.init_ramlak_1D
# TODO: Check, if we need to incorporate those in the back/forward projection
#       or if they are already encoded in the projection matrices..
PIXEL_WIDTH_MM = 1
PIXEL_HEIGHT_MM = 1
SOURCE_DET_DISTANCE = 1200
N = 500
U = 800
V = 600


# GLOBALS
#-------------------------------------------------------------------------------------------
asserts = []


# READ DATA
#-------------------------------------------------------------------------------------------
proj = dennerlein.read( DATA_P + 'proj.bin' )
geom, angles = projtable.read( DATA_P + 'projMat.txt' )
geom_tensor = tf.constant( geom, dtype = tf.float32 )
proj_shape = tf.shape( proj )
with tf.control_dependencies( [ proj ] ):
    asserts.append( tf.assert_equal( tf.shape( proj ), [ N, V, U ] ) )


# COSINE
#-------------------------------------------------------------------------------------------
cosine_w_np = ct.init_cosine_3D( SOURCE_DET_DISTANCE, U, V, VOXEL_WIDTH_MM,
        VOXEL_HEIGHT_MM )
cosine_w = tf.constant( cosine_w_np, dtype = tf.float32 )
proj_cosine = tf.multiply( proj, cosine_w )


# PARKER
#-------------------------------------------------------------------------------------------
parker_w_np = ct.init_parker_3D( angles, SOURCE_DET_DISTANCE, U, PIXEL_WIDTH_MM )
parker_w = tf.constant( parker_w_np, dtype = tf.float32 )
proj_parker = tf.multiply( proj_cosine, parker_w )


# RAMLAK
#-------------------------------------------------------------------------------------------

# TODO: Seems like cudnn does not support 3D convolutions.. Find a way to do
# that with conv2d..

# need format batch, depth, height, width, channel for conv3d
proj_batch = tf.reshape( proj_parker, [ 1, N, V, U, 1 ] )

def kernel_init( shape, dtype, partition_info = None ):
    kernel = tf.Variable( ct.init_ramlak_1D( RAMLAK_WIDTH, PIXEL_WIDTH_MM ), dtype = dtype )
    return tf.reshape( kernel, shape )

ramlak_batch = tf.layers.conv3d(
        inputs = proj_batch,
        filters = 1,
        kernel_size = [ 1, 1, RAMLAK_WIDTH ],
        padding = 'same',
        use_bias = False,
        kernel_initializer = kernel_init,
        name = 'ramlak-filter'
    )
proj_ramlak = tf.reshape( ramlak_batch, [ N, V, U ] )


# BACKPROJECTION
#-------------------------------------------------------------------------------------------
vo = tf.contrib.util.make_tensor_proto( VOLUME_ORIGIN_MM,
        tf.float32 )
geom_proto = tf.contrib.util.make_tensor_proto( geom, tf.float32 )
voxel_dimen_proto = tf.contrib.util.make_tensor_proto( [ VOXEL_DEPTH_MM,
        VOXEL_HEIGHT_MM, VOXEL_WIDTH_MM ], tf.float32 )
proj_shape_proto = tf.contrib.util.make_tensor_proto( [ N, V, U ], tf.int32 )
volume = ct.backproject(
        projections = proj_ramlak,
        geom = geom_proto,
        vol_shape = VOLUME_SHAPE_VX,
        vol_origin=vo,
        voxel_dimen = voxel_dimen_proto,
        proj_shape = [ N, V, U ],
        name = 'backprojection'
    )

fproj = ct.project(
        volume = volume,
        geom = geom_proto,
        vol_shape = VOLUME_SHAPE_VX,
        vol_origin=vo,
        voxel_dimen = voxel_dimen_proto,
        proj_shape = [ N, V, U ],
        name = 'forwardprojection'
    )


# WRITE RESULT
#-------------------------------------------------------------------------------------------
write_op = dennerlein.write( '/tmp/test.bin', fproj )

gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.5 )

with tf.Session( config = tf.ConfigProto( gpu_options = gpu_options ) ) as sess:
    sess.run( tf.global_variables_initializer() )

    # tracing according to http://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    v = sess.run( [ write_op ] + asserts, options = run_options, run_metadata = run_metadata )

    # write timeline object to file
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)


