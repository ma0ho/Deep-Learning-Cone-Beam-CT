import tensorflow as tf
import os

_path = os.path.dirname(os.path.abspath(__file__))
_write_module = tf.load_op_library( _path + '/../../user-ops/backproject.so' )
backproject = _write_module.backproject
