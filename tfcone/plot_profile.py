# from https://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import sys
from skimage import io

def extract( im, x0, y0, x1, y1, num ):
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    return scipy.ndimage.map_coordinates(im, np.vstack((x,y)))

lafile = sys.argv[1]
outfile = sys.argv[2]

# read
im_la = io.imread( lafile ).astype(np.float32)
H, W = im_la.shape

# normalize
m = im_la.max()
im_la /= m

# set line in pixel coords
x0, y0 = W/2, 0
x1, y1 = W/2, H

p_la = extract( im_la, x0, y0, x1, y1, H )

#-- Plot...
fig = plt.figure()
ax = fig.add_subplot( '111' )
#axes[0].imshow(im1)
#axes[0].plot([x0, x1], [y0, y1], 'ro-')
#axes[0].axis('image')
#axes[1].imshow(im2)
#axes[1].plot([x0, x1], [y0, y1], 'ro-')
#axes[1].axis('image')

ax.plot( p_la, label = 'limited angle $\mathbf{f}_l$', color = 'red' )
#ax.get_xaxis().set_ticks([])
ax.legend( loc = 8 )
ax.set_xlabel( 'position [px]' )
ax.set_ylabel( 'normalized intensity' )
ax.set_xlim( ( 0, H ) )
ax.set_ylim( ( 0.0, 1 ) )
#axes[2].plot(np.abs(z2-z1))

#plt.show()
plt.savefig( outfile )
plt.close()


