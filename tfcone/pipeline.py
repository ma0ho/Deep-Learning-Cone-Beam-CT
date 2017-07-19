import tensorflow as tf
import numpy as np
import os
import copy
import random
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
DATA_P              = os.path.abspath(
                        os.path.dirname( os.path.abspath( __file__ ) )
                        + '/../phantoms/lowdose/'
                    ) + '/'
PROJ_FILES          = [ DATA_P + f for f in os.listdir( DATA_P )
                        if f.endswith(".proj.bin") ]
VOL_FILES           = [ DATA_P + f for f in os.listdir( DATA_P )
                        if f.endswith(".vol.bin") ]
PROJ_FILES.sort()
VOL_FILES.sort()
LOG_DIR             = '/tmp/train/'

# TRAINING CONFIG
LIMITED_ANGLE_SIZE  = 180   # #projections for limited angle
LEARNING_RATE       = 0.01
EPOCHS              = 1  # unlimited
BATCH_SIZE          = 1     # TODO: find out why > 1 causes OOM
TRACK_LOSS          = 10    # number of models/losses to track

# GPU RELATED STAFF
GPU_FRACTION        = .75
SAVE_GPU_MEM        = True
GPU_OPTIONS         = tf.GPUOptions( per_process_gpu_memory_fraction = GPU_FRACTION )


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


# SETUP INPUT PIPELINE
#-------------------------------------------------------------------------------------------
def input_pipeline( train_proj_fns, train_vol_fns, test_proj_fns, test_vol_fns ):
    train_proj = tf.train.string_input_producer( train_proj_fns, num_epochs = EPOCHS,
            shuffle = False )
    train_label = tf.train.string_input_producer( train_vol_fns, num_epochs = EPOCHS,
            shuffle = False )
    test_proj  = tf.train.string_input_producer( test_proj_fns,
            shuffle = False )
    test_label  = tf.train.string_input_producer( test_vol_fns,
            shuffle = False )

    train_proj = dennerlein.read( train_proj, 'TrainProjReader',
            PROJ_SHAPE.toNCHW() )
    train_label = dennerlein.read( train_label, 'TrainLabelReader',
            VOL_SHAPE.toNCHW() )
    test_proj = dennerlein.read( test_proj, 'TestProjReader',
            PROJ_SHAPE.toNCHW() )
    test_label = dennerlein.read( test_label, 'TestLabelReader',
            VOL_SHAPE.toNCHW() )

    # pick LA slice
    train_proj = train_proj[0:LIMITED_ANGLE_SIZE]
    test_proj = test_proj[0:LIMITED_ANGLE_SIZE]

    train_proj_batch, train_label_batch = tf.train.shuffle_batch(
            [ train_proj, train_label ],
            batch_size = BATCH_SIZE,
            capacity = 10,
            min_after_dequeue = 5
    )

    return train_proj_batch, train_label_batch, test_proj, test_label


# MODEL
#-------------------------------------------------------------------------------------------
class Model:

    def train_on_batch( self, train_proj_batch, train_label_batch, reconstructor, geom ):

        opt = tf.train.AdamOptimizer( LEARNING_RATE )

        train_step = tf.no_op()

        for i in range( 0, BATCH_SIZE ):
            with tf.device("/cpu:0"):
                train_proj = train_proj_batch[i]
                train_label = train_label_batch[i]
            volume_la = reconstructor.apply( train_proj, geom )
            if SAVE_GPU_MEM:
                # compute loss on cpu (avoid too many volume instances on gpu)
                with tf.device("/cpu:0"):
                    loss = tf.losses.mean_squared_error( train_label, volume_la )
            else:
                loss = tf.losses.mean_squared_error( train_label, volume_la )

            with tf.control_dependencies( [ train_step ] ):
                train_step = opt.minimize( loss, colocate_gradients_with_ops = True )

        return train_step

    def __init__( self, train_proj, train_vol, test_proj, test_vol, sess ):
        train_proj_fns_init = tf.placeholder( tf.string, shape = ( len(train_proj) ) )
        train_vol_fns_init = tf.placeholder( tf.string, shape = ( len(train_vol) ) )
        test_proj_fns_init = tf.placeholder( tf.string, shape = ( len(test_proj) ) )
        test_vol_fns_init = tf.placeholder( tf.string, shape = ( len(test_vol) ) )

        train_proj_fns = tf.Variable( train_proj_fns_init, trainable = False,
                collections = [] )
        train_vol_fns = tf.Variable( train_vol_fns_init, trainable = False,
                collections = [] )
        test_proj_fns = tf.Variable( test_proj_fns_init, trainable = False,
                collections = [] )
        test_vol_fns = tf.Variable( test_vol_fns_init, trainable = False,
                collections = [] )

        sess.run( train_proj_fns.initializer, feed_dict = { train_proj_fns_init:
            train_proj } )
        sess.run( train_vol_fns.initializer, feed_dict = { train_vol_fns_init:
            train_vol } )
        sess.run( test_proj_fns.initializer, feed_dict = { test_proj_fns_init:
            test_proj } )
        sess.run( test_vol_fns.initializer, feed_dict = { test_vol_fns_init:
            test_vol } )

        geom, angles = projtable.read( DATA_P + 'projMat.txt' )

        re = ct.Reconstructor(
                CONF_LA, angles[0:LIMITED_ANGLE_SIZE],
                trainable = True,
                name = 'LAReconstructor'
        )
        geom_la = geom[0:LIMITED_ANGLE_SIZE]
        with tf.device("/cpu:0"):
            batch, batch_label, test, test_label = input_pipeline( train_proj_fns,
                    train_vol_fns, test_proj_fns, test_vol_fns )

        self.train_op = self.train_on_batch( batch, batch_label, re, geom_la )

        self.test_vol = re.apply( test, geom_la )
        with tf.device("/cpu:0"):
            self.test_loss = tf.losses.mean_squared_error( test_label, self.test_vol )

def split_train_validation_set( offset):
    N = len( PROJ_FILES )
    ntrain = N - 1

    # put  test file to end
    proj_files = copy.deepcopy( PROJ_FILES )
    tmp = proj_files[ offset ]
    del proj_files[ offset ]
    proj_files += [ tmp ]
    vol_files = copy.deepcopy( VOL_FILES )
    tmp = vol_files[ offset ]
    del vol_files[ offset ]
    vol_files += [ tmp ]

    return proj_files[:-1], vol_files[:-1], [ proj_files[-1] ], [ vol_files[-1] ]

def train_model( offset = 0, save_path = 'model', track_loss = TRACK_LOSS,
        stop_crit = lambda l: False ):

    step = 0
    losses = []

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        sets = split_train_validation_set( offset )

        m = Model( sets[0], sets[1], sets[2], sets[3], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        c = lambda l: stop_crit( l ) if len( l ) >= track_loss else False

        saver = tf.train.Saver( max_to_keep = track_loss )

        try:
            while not coord.should_stop() and not c( losses ):
                print( 'Training step %d' % step )
                # TODO: Hack!
                for i in range( 0, 3 ):
                    sess.run( m.train_op )
                lv = sess.run( m.test_loss )
                print( 'Loss: %f' % lv )

                losses.append( lv )

                if len( losses ) > track_loss:
                    del losses[0]

                saver.save( sess, save_path, global_step = step )

                step += 1

        except tf.errors.OutOfRangeError:
            print( 'Done.' )
        finally:
            coord.request_stop()

        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

    return losses, step


# LABELS
#-------------------------------------------------------------------------------------------
def create_label( fn_proj, fn_vol, rec, geom ):
    proj = dennerlein.read_noqueue( fn_proj )
    volume = rec.apply( proj, geom, fullscan = True )
    return dennerlein.write( fn_vol, volume )

def update_labels():
    geom, angles = projtable.read( DATA_P + 'projMat.txt' )
    ref_reconstructor = ct.Reconstructor( CONF, angles, name = 'RefReconstructor' )

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        sess.run( tf.global_variables_initializer() )

        for fn_proj in PROJ_FILES:
            fn_vol = fn_proj.replace( 'proj', 'vol' )

            if not os.path.exists( fn_vol ):
                print( 'Creating label for %s' % fn_proj )
                sess.run( create_label( fn_proj, fn_vol, ref_reconstructor, geom ) )
                VOL_FILES.append( fn_vol )

        sess.close()

    tf.reset_default_graph()

    PROJ_FILES.sort()
    VOL_FILES.sort()


# TEST
#-------------------------------------------------------------------------------------------
def test_model( test_proj, test_label, save_path = 'model' ):

    step = 0
    losses = []

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        m = Model( [ test_proj ], [ test_label ], [ test_proj ], [ test_label ], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        c = lambda l: stop_crit( l ) if len( l ) >= track_loss else False

        saver = tf.train.Saver()
        saver.restore( sess, save_path )

        # TODO

        coord.request_stop()
        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

    return losses, step


# GO GO GO.. :)
#-------------------------------------------------------------------------------------------
print( 'Check if all projections have corresponding labels..' )
update_labels()

print( 'Dropping %s for test purposes..' % PROJ_FILES[-1] )
test_proj = PROJ_FILES[-1]
test_label = VOL_FILES[-1]
del PROJ_FILES[-1]
del VOL_FILES[-1]

losses = []
steps = []

for i in range( 0, len( PROJ_FILES ) ):
    print( 'Start training model %d' % (i+1) )

    log_file = LOG_DIR + ( 'model_%d' %i )
    l, s = train_model( 0, save_path = log_file, stop_crit = lambda l:
        np.mean( l ) > l[0] )

    losses.append( np.min( l ) )
    steps.append( s - ( TRACK_LOSS - np.argmin( l ) ) )

print( losses )
print( steps )

#test_model( test_proj, test_label )




# print trainable vars
#variables_names = [v.name for v in tf.trainable_variables()]
#values = sess.run(variables_names)
#for k, v in zip(variables_names, values):
#    print( "Variable: ", k )


# make sure that labels exist

