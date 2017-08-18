import numpy as np
import sys
from skimage import io

in_f = sys.argv[1]
slice_n = int(sys.argv[2])
out_f = sys.argv[3]

raw = np.fromfile( in_f, np.uint8 )

header = raw[0:6].view( np.uint16 )
N, H, W = header[0], header[1], header[2]
data = raw[6:].view( np.float32 ).reshape( (N,H,W) )

sl = data[slice_n]

sl -= sl.min()
sl /= sl.max()

io.imsave( out_f, sl )



