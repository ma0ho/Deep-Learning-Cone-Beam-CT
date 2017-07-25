import tensorflow as tf
import numpy as np
import os
import copy
import random
from tfcone.inout import dennerlein, projtable, png
from tfcone.algo import ct
from tfcone.util import types as t
from tfcone.util import plot as plt
from tensorflow.python.client import timeline
from skimage.measure import compare_ssim, compare_psnr
import re
import math
import argparse

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
LEARNING_RATE       = 0.00000002
EPOCHS              = None  # unlimited
BATCH_SIZE          = 1     # TODO: find out why > 1 causes OOM
TRACK_LOSS          = 30    # number of models/losses to track

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
                gstep = tf.train.get_global_step()
                lr = tf.train.exponential_decay( LEARNING_RATE, gstep, 200, 1 )
                train_step = tf.train.GradientDescentOptimizer( lr ).minimize(
                        loss, colocate_gradients_with_ops = True, global_step =
                        gstep )

        return train_step

    def setTest( self, test_proj, test_vol, sess ):
        sess.run( self.test_proj_fns.initializer, feed_dict = { self.test_proj_fns_init:
            test_proj } )
        sess.run( self.test_vol_fns.initializer, feed_dict = { self.test_vol_fns_init:
            test_vol } )

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

        sess.run( self.train_proj_fns.initializer, feed_dict = { self.train_proj_fns_init:
            train_proj } )
        sess.run( self.train_vol_fns.initializer, feed_dict = { self.train_vol_fns_init:
            train_vol } )
        sess.run( self.test_proj_fns.initializer, feed_dict = { self.test_proj_fns_init:
            test_proj } )
        sess.run( self.test_vol_fns.initializer, feed_dict = { self.test_vol_fns_init:
            test_vol } )

        geom, angles = projtable.read( DATA_P + 'projMat.txt' )

        re = ct.Reconstructor(
                CONF_LA, angles[0:LIMITED_ANGLE_SIZE],
                trainable = True,
                name = 'LAReconstructor'
        )
        geom_la = geom[0:LIMITED_ANGLE_SIZE]
        with tf.device("/cpu:0"):
            batch, batch_label, test, test_label = input_pipeline( self.train_proj_fns,
                    self.train_vol_fns, self.test_proj_fns, self.test_vol_fns )

        self.test_label = test_label
        self.parker_w = re.parker_w

        if not tf.train.get_global_step():
            tf.train.create_global_step()

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

def train_model( offset = 0, save_path = '/tmp/', track_loss = TRACK_LOSS,
        stop_crit = lambda l: False, resume = False ):

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

        if resume:
            cp = tf.train.latest_checkpoint( save_path )
            if cp:
                print( 'Restoring sessing' )
                saver.restore( sess, cp )

        try:
            while not coord.should_stop() and not c( losses ):
                # TODO: Hack!
                for i in range( 0, 2 ):
                    sess.run( m.train_op )
                lv, step = sess.run( [ m.test_loss, tf.train.get_global_step() ] )
                print( 'Step %d' % step )
                print( 'Loss: %f' % lv )

                losses.append( lv )

                if len( losses ) > track_loss:
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
def write_test_volumes( test_proj, test_label, out_path = LOG_DIR ):

    with tf.Session( config = tf.ConfigProto( gpu_options = GPU_OPTIONS ) ) as sess:
        m = Model( [ test_proj ], [ test_label ], [ test_proj ], [ test_label ], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        # compute volumes before training and export dennerlein
        print( 'Exporting volumes without trained parameters' )
        vol_before = m.test_vol
        write_dennerlein = dennerlein.write( out_path + 'test_la.bin',
                vol_before )
        write_dennerlein_label = dennerlein.write( out_path + 'test_fa.bin',
                m.test_label )
        sess.run( [ write_dennerlein, write_dennerlein_label ]  )

        coord.request_stop()
        coord.join( threads )
        sess.close()

    tf.reset_default_graph()


def test_model( validation_proj, validation_label, test_proj, test_label,
        save_path = 'model', out_path = LOG_DIR ):

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
        m = Model( [ test_proj ], [ test_label ], [ test_proj ], [ test_label ], sess )

        sess.run( tf.global_variables_initializer() )
        sess.run( tf.local_variables_initializer() )

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = sess, coord = coord )

        # compute volume before training and export central slice
        print( 'Computing volume without trained parameters' )
        vol_before = m.test_vol
        write_png = png.writeSlice( vol_before[ int( CONF.proj_shape.N / 2 ) ],
                out_path + 'slice_before.png' )
        write_png_label = png.writeSlice( m.test_label[ int( CONF.proj_shape.N / 2 ) ],
                out_path + 'slice_label.png' )
        vol_before_np, _, _, vol_label_np, parker_w_before_np = sess.run( [ vol_before, write_png,
            write_png_label, m.test_label, m.parker_w ]  )

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
        m.setTest( [ test_proj ], [ test_label ], sess )
        saver.restore( sess, checkpoints[best_cp_i] )

        # compute volume after training and export central slice + dennerlein
        vol_after = m.test_vol
        write_png = png.writeSlice( vol_after[ int( CONF.proj_shape.N / 2 ) ],
                out_path + 'slice_after.png' )
        write_dennerlein = dennerlein.write( out_path + 'after.bin', vol_after )
        vol_after_np, _, _, parker_w_after_np = sess.run( [ vol_after,
            write_png, write_dennerlein, m.parker_w ]  )

        # plot + export parker weights
        plt.plot_parker( parker_w_before_np, out_path + 'parker_before.png' )
        plt.plot_parker( parker_w_after_np, out_path + 'parker_after.png' )
        np.save( out_path + 'parker_before.npy', parker_w_before_np )
        np.save( out_path + 'parker_after.npy', parker_w_after_np )

        # compute ssim and psnr
        print( 'Computing ssim and psnr' )
        drange = vol_label_np.max()

        ssim_before = compare_ssim( vol_before_np, vol_label_np, data_range = drange,
                gaussian_weights = True, sigma = 1.5, use_sample_covariance = False )
        ssim_after = compare_ssim( vol_after_np, vol_label_np, data_range = drange,
                gaussian_weights = True, sigma = 1.5, use_sample_covariance = False )

        psnr_before = compare_psnr( vol_label_np, vol_before_np, data_range =
                drange )
        psnr_after = compare_psnr( vol_label_np, vol_after_np, data_range =
                drange )

        # write to file
        with open( out_path + 'measures.csv', 'w' ) as f:
            f.write( 'ssim_before,ssim_after,psnr_before,psnr_after\n' )
            f.write( '%f,%f,%f,%f' % ( ssim_before, ssim_after, psnr_before,
                psnr_after ) )

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

    args = parser.parse_args()

    print( 'Check if all projections have corresponding labels..' )
    update_labels()

    print( 'Dropping %s for test purposes..' % PROJ_FILES[-1] )
    test_proj = PROJ_FILES[-1]
    test_label = VOL_FILES[-1]
    del PROJ_FILES[-1]
    del VOL_FILES[-1]

    if args.only:
        start = args.only
        end = args.only + 1
    else:
        start = args.start
        end = args.end if args.end > 0 else len( PROJ_FILES )

    # TRAIN
    if args.train:
        losses = []
        steps = []

        for i in range( start, end ):
            print( 'Start training model %d' % (i+1) )

            save_path = LOG_DIR + ( 'model_%d/' %i )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            l, s = train_model( 0, save_path = save_path, stop_crit = lambda l:
                np.median( l[:int(len(l)/2)] ) < np.median( l[int(len(l)/2):] ),
                resume = True
            )

            losses.append( np.min( l ) )
            steps.append( np.argmin( l ) )


    # TEST
    if args.test:
        write_test_volumes( test_proj, test_label )

        for i in range( start, end ):
            print( 'Testing model %d' % i )

            _, _, validation_proj, validation_label = split_train_validation_set( i )

            save_path = LOG_DIR + ( 'test_model_%d/' %i )
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            test_model( validation_proj, validation_label, test_proj, test_label,
                    '/tmp/train/model_%d/' % i, save_path )




# print trainable vars
#variables_names = [v.name for v in tf.trainable_variables()]
#values = sess.run(variables_names)
#for k, v in zip(variables_names, values):
#    print( "Variable: ", k )


# make sure that labels exist

