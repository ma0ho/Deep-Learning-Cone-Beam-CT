import tensorflow as tf
import os
import math

_path = os.path.dirname(os.path.abspath(__file__))
_write_module = tf.load_op_library( _path + '/../../user-ops/backproject.so' )
backproject = _write_module.backproject

'''
    generate 1D-RamLak filter according to Kak & Slaney, chapter 3 equation 61
'''
def init_ramlak_1D( width, pixel_width_mm ):
    assert( width % 2 == 1 )

    hw = int( ( width-1 ) / 2 )
    f = [
            -1 / math.pow( i * math.pi * pixel_width_mm, 2 ) if i%2 == 1 else 0
            for i in range( -hw, hw+1 )
        ]
    f[hw] = 1/4 * math.pow( pixel_width_mm, 2 )
    return f

