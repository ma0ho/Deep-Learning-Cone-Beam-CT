import tensorflow as tf
import numpy as np
import os
from tfcone.inout import dennerlein, projtable
from tfcone.algo import ct
from tfcone.util import types as t
from tensorflow.python.client import timeline

# CONFIG
#-------------------------------------------------------------------------------------------
DATA_P = os.path.abspath(
        os.path.dirname( os.path.abspath( __file__ ) ) + '/../phantoms/conrad-5/'
    ) + '/'

RAMLAK_WIDTH = 401
VOL_SHAPE = t.Shape3D(
        W = 800, H = 800, D = 600
)
VOL_ORIG = t.Coord3D(
        X = -200, Y = -200, Z = -150
)
VOXEL_DIMS = t.Shape3D(
        W = 0.5, H = 0.5, D = 0.5
)
# NOTE: See todo at ct.init_ramlak_1D
# TODO: Check, if we need to incorporate those in the back/forward projection
#       or if they are already encoded in the projection matrices..
PIXEL_DIMS = t.Shape2D(
        W = 1, H = 1
)
SOURCE_DET_DISTANCE = 1200
PROJ_SHAPE = t.ShapeProj(
        N = 720, W = 800, H = 600
)
CONF = ct.ReconstructionConfiguration(
        PROJ_SHAPE,
        VOL_SHAPE,
        VOL_ORIG,
        VOXEL_DIMS,
        PIXEL_DIMS,
        SOURCE_DET_DISTANCE,
        RAMLAK_WIDTH
)
GPU_TOTAL_MEM_MIB = 7500


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
    asserts.append( tf.assert_equal( tf.shape( proj ), [ CONF.proj_shape.N,
        CONF.proj_shape.H, CONF.proj_shape.W ] ) )


# RECONSTRUCT
#-------------------------------------------------------------------------------------------
reconstructor = ct.Reconstructor( CONF )
volume = reconstructor.apply( proj, geom, angles )


# WRITE RESULT
#-------------------------------------------------------------------------------------------
write_op = dennerlein.write( '/tmp/test.bin', volume )


# LAUNCH
#-------------------------------------------------------------------------------------------
# determine the memory fraction that is needed by tensorflow
# we need to have the projections twice and the volume once in memory..
proj_size_bytes     = CONF.proj_shape.size() * 4
vol_size_bytes      = CONF.vol_shape.size() * 4
gpu_size_bytes      = GPU_TOTAL_MEM_MIB * 1024 * 1024
buffer_size_bytes   = 900 * 1024 * 1024   # reserve 900MB extra memory for TF
gpu_fraction = ( 2*proj_size_bytes + vol_size_bytes + buffer_size_bytes ) / gpu_size_bytes

gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = gpu_fraction )

with tf.Session( config = tf.ConfigProto( gpu_options = gpu_options ) ) as sess:
    sess.run( tf.global_variables_initializer() )

    # tracing according to http://stackoverflow.com/questions/34293714/can-i-measure-the-execution-time-of-individual-operations-with-tensorflow
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    v = sess.run( [ write_op ] + asserts, options = run_options, run_metadata = run_metadata )

    # write timeline object to file
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format( show_memory=True )
    with open('timeline.json', 'w') as f:
        f.write(ctf)


