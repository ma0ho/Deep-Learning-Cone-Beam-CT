import tensorflow as tf
from tfcone.io import projtable as pt
from PIL import Image
import numpy as np

def test_read_projtable():
    t = pt.read( 'projtable.txt' )
    x = t[1]
    print(x)

    sess = tf.Session()
    v = sess.run(x)
    print(v)
