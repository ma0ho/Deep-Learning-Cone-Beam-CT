import tensorflow as tf
import numpy as np
import os
import copy
import random
from inout import dennerlein, projtable, png
from algo import ct
from util import types as t
from tensorflow.python.client import timeline
import re
import math
import argparse

# CONFIG
#-------------------------------------------------------------------------------------------

# GEOMETRY
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
RAMLAK_WIDTH        = 2*PROJ_SHAPE.W + 1

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
LEARNING_RATE       = 0.00000002
EPOCHS              = None  # unlimited
TRACK_LOSS          = 30    # number of models/losses to track for early stopping
TEST_EVERY          = 2     # number of training steps before test is run
WEIGHTS_TYPE        = 'parker'

# GPU RELATED STAFF
GPU_FRACTION        = .75
SAVE_GPU_MEM        = True
GPU_OPTIONS         = tf.GPUOptions( per_process_gpu_memory_fraction = GPU_FRACTION )


# SOME SETUP
#-------------------------------------------------------------------------------------------

# full angle configuration
CONF = ct.ReconstructionConfiguration(
        PROJ_SHAPE,
        VOL_SHAPE,
        VOL_ORIG,
        VOXEL_DIMS,
        PIXEL_DIMS,
        SOURCE_DET_DISTANCE,
        RAMLAK_WIDTH
)

# limited angle configuration
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
            batch_size = 1,
            capacity = 10,
            min_after_dequeue = 5
    )

    return train_proj_batch[0], train_label_batch[0], test_proj, test_label


# MODEL
#-------------------------------------------------------------------------------------------
class Model:

    def train_on_projections( self, train_proj, train_label, reconstructor, geom ):

        train_step = tf.no_op()

        volume_la = reconstructor.apply( train_proj, geom )

        if SAVE_GPU_MEM:
            # compute loss on cpu (avoid too many volume instances on gpu)
            with tf.device("/cpu:0"):
                loss = tf.losses.mean_squared_error( train_label, volume_la )
        else:
            loss = tf.losses.mean_squared_error( train_label, volume_la )

        with tf.control_dependencies( [ train_step ] ):
            gstep = tf.train.get_global_step()
            train_step = tf.train.GradientDescentOptimizer( LEARNING_RATE ).minimize(
                    loss,
                    colocate_gradients_with_ops = True,
                    global_step = gstep
                )

        return train_step

    def setTest( self, test_proj, test_vol, sess ):
        sess.run( self.test_proj_fns.initializer, feed_dict = {
                    self.test_proj_fns_init: test_proj
                }
            )
        sess.run( self.test_vol_fns.initializer, feed_dict = {
                    self.test_vol_fns_init: test_vol
                }
            )

    def __init__( self, train_proj, train_vol, test_proj, test_vol, sess ):
        self.train_proj_fns_init = tf.placeholder( tf.string, shape = ( len(train_proj) ) )
        self.train_vol_fns_init = tf.placeholder( tf.string, shape = ( len(train_vol) ) )
        self.test_proj_fns_init = tf.placeholder( tf.string, shape = ( len(test_proj) ) )
        self.test_vol_fns_init = tf.placeholder( tf.string, shape = ( len(test_vol) ) )

        self.train_proj_fns = tf.Variable( self.train_proj_fns_init, trainable = False,
                collections = [] )
        self.train_vol_fns = tf.Variable( self.train_vol_fns_init, trainable = False,
                collections = [] )
        self.test_proj_fns = tf.Variable( self.test_proj_fns_init, trainable = False,
                collections = [] )
        self.test_vol_fns = tf.Variable( self.test_vol_fns_init, trainable = False,
                collections = [] )
        sess.run( self.train_proj_fns.initializer, feed_dict = {
                    self.train_proj_fns_init: train_proj
                }
            )
        sess.run( self.train_vol_fns.initializer, feed_dict = {
                    self.train_vol_fns_init: train_vol
                }
            )
        sess.run( self.test_proj_fns.initializer, feed_dict = {
                    self.test_proj_fns_init: test_proj
                }
            )
        sess.run( self.test_vol_fns.initializer, feed_dict = {
                    self.test_vol_fns_init: test_vol
                }
            )

        geom, angles = projtable.read( DATA_P + 'projMat.txt' )

        re = ct.Reconstructor(
                CONF_LA, angles[0:LIMITED_ANGLE_SIZE],
                trainable = True,
                name = 'LAReconstructor',
                weights_type = WEIGHTS_TYPE
        )
        geom_la = geom[0:LIMITED_ANGLE_SIZE]

        with tf.device("/cpu:0"):
            train, train_label, test, test_label = input_pipeline(
                    self.train_proj_fns,
                    self.train_vol_fns,
                    self.test_proj_fns,
                    self.test_vol_fns
                )

        self.test_label = test_label

        if not tf.train.get_global_step():
            tf.train.create_global_step()

        self.train_op = self.train_on_projections( train, train_label, re, geom_la )
        self.test_vol = re.apply( test, geom_la )

        with tf.device("/cpu:0"):
            self.test_loss = tf.losses.mean_squared_error( test_label, self.test_vol )


def split_train_validation_set( offset ):
    N = len( PROJ_FILES ) - 1
    ntrain = N - 1

    validation_idx = (offset-1) % N

    # put  test file to end
    proj_files = copy.deepcopy( PROJ_FILES )
    del proj_files[offset]              # do not use test proj
    validation_proj = proj_files[validation_idx]
    del proj_files[validation_idx]

    vol_files = copy.deepcopy( VOL_FILES )
    del vol_files[ offset ]             # do not use test volume
    validation_vol = vol_files[validation_idx]
    del vol_files[validation_idx]

    return proj_files, vol_files, [validation_proj], [validation_vol]


def train_model( offset, save_path, resume ):
    losses = []
    stop_crit = lambda l: np.median( l[:int(len(l)/2)] ) < np.median( l[int(len(l)/2):] )

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        sets = split_train_validation_set( offset )
        m = Model( *sets, sess, weights_type = 'parker' )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        c = lambda l: stop_crit( l ) if len( l ) >= TRACK_LOSS else False

        saver = tf.train.Saver( max_to_keep = TRACK_LOSS )

        if resume:
            cp = tf.train.latest_checkpoint( save_path )
            if cp:
                print( 'Restoring session' )
                saver.restore( sess, cp )

        try:
            while not coord.should_stop() and not c( losses ):
                for i in range( TEST_EVERY ):
                    sess.run( m.train_op )

                lv, step = sess.run( [ m.test_loss, tf.train.get_global_step() ] )
                print( 'Step %d; Loss: %f' % ( step, lv ) )

                losses.append( lv )
                if len( losses ) > TRACK_LOSS:
                    del losses[0]

                saver.save( sess, save_path + 'model', global_step = step )

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
def write_test_volumes( test_proj, test_label ):

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        m = Model( [test_proj], [test_label], [test_proj], [test_label], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        # compute volumes before training and export dennerlein
        proj_filename = os.path.splitext( os.path.splitext( os.path.basename( test_proj ) )[0] )[0]
        out_fa = LOG_DIR + ( 'test_%s_%s_fa.bin' % ( proj_filename, WEIGHTS_TYPE ) )
        out_la = LOG_DIR + ( 'test_%s_%s_la.bin' % ( proj_filename, WEIGHTS_TYPE ) )

        vol_before = m.test_vol
        write_dennerlein = dennerlein.write( out_la, vol_before )
        write_dennerlein_label = dennerlein.write( out_fa, m.test_label )
        sess.run( [ write_dennerlein, write_dennerlein_label ]  )

        coord.request_stop()
        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

def test_model( validation_proj, validation_label, test_proj, test_label, save_path ):
    step = 0
    losses = []

    # find checkpoint files
    checkpoints = []
    with open( save_path + 'checkpoint' ) as f:
        f = f.readlines()
        pattern = re.compile( 'all_model_checkpoint_paths:\s\"(.+)\"' )
        for line in f:
            for match in re.finditer( pattern, line ):
                checkpoints.append( match.groups()[0] )

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        m = Model( [test_proj], [test_label], [test_proj], [test_label], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        # compute volume before training and export central slice
        print( 'Computing volume without trained parameters' )
        vol_before = m.test_vol
        write_png = png.writeSlice( vol_before[CONF.proj_shape.N // 2], LOG_DIR + 'slice_before_training.png' )
        write_png_label = png.writeSlice( m.test_label[CONF.proj_shape.N // 2], LOG_DIR + 'slice_label.png' )
        vol_before_np, _, _, vol_label_np = sess.run( [vol_before, write_png, write_png_label, m.test_label] )

        # find best checkpoint
        print( 'Searching best checkpoint' )
        saver = tf.train.Saver()
        m.setTest( validation_proj, validation_label, sess )

        best_cp_i = 0
        best_cp_loss = math.inf

        for i, cp in enumerate( checkpoints ):
            saver.restore( sess, cp )
            loss = sess.run( m.test_loss )

            if loss < best_cp_loss:
                best_cp_i = i
                best_cp_loss = loss
            print( '.', end = '', flush = True )
        print( '' )

        # load best model and set test volume
        print( 'Computing volume with trained parameters' )
        m.setTest( [test_proj], [test_label], sess )
        saver.restore( sess, checkpoints[best_cp_i] )

        # compute volume after training and export central slice + dennerlein
        vol_after = m.test_vol
        write_png = png.writeSlice( vol_after[CONF.proj_shape.N // 2], LOG_DIR + 'slice_after_training.png' )
        write_dennerlein = dennerlein.write( LOG_DIR + 'after_training.bin', vol_after )
        vol_after_np, _, _ = sess.run( [vol_after, write_png, write_dennerlein]  )

        coord.request_stop()
        coord.join( threads )
        sess.close()

    tf.reset_default_graph()

    return losses, step


# GO GO GO.. :)
#-------------------------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group( required = True )
    group.add_argument( "--train", action="store_true" )
    group.add_argument( "--test", action="store_true" )
    rgroup = parser.add_mutually_exclusive_group()
    segroup = rgroup.add_argument_group()
    segroup.add_argument( "--start", type=int, default = 0 )
    segroup.add_argument( "--end", type=int, default = -1 )
    rgroup.add_argument( "--only", type=int )
    parser.add_argument( "--resume", action="store_true" )
    args = parser.parse_args()

    print( 'Writing result to %s' % LOG_DIR )

    print( 'Check if all projections have corresponding labels..' )
    update_labels()

    if args.only is not None:
        start = args.only
        end = args.only + 1
    else:
        start = args.start
        end = args.end if args.end > 0 else len( PROJ_FILES )

    # TRAIN
    if args.train:
        for i in range( start, end ):
            print( 'Start training model %d' % (i) )
            print( 'Leaving %s for test purposes..' % PROJ_FILES[i] )

            save_path = LOG_DIR + ( 'model_%d/' %i )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            l, s = train_model( i,
                    save_path = save_path,
                    resume = args.resume
                )

    # TEST
    if args.test:
        for i in range( start, end ):
            print( 'Testing model %d' % i )
            test_proj = PROJ_FILES[i]
            test_label = VOL_FILES[i]

            print( 'Writing test volumes for %s' % test_proj )
            write_test_volumes( test_proj, test_label )

            _, _, validation_proj, validation_label = split_train_validation_set( i )

            save_path = LOG_DIR + ( 'test_model_%d/' %i )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            test_model( validation_proj, validation_label, test_proj, test_label, '/tmp/train/model_%d/' % i, save_path )

