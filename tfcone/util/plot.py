import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
    w
        parker weights as angles x 1 x W array
'''
def plot_parker( w, p ):
    x = np.arange( 0, w.shape[2] )
    y = np.arange( 0, w.shape[0] )

    fig = plt.figure()
    ax = Axes3D( fig )

    x, y = np.meshgrid( x, y )
    z = np.array( [ w[y,0,x] for x, y in zip( np.ravel(x), np.ravel(y) ) ] )
    z = z.reshape( x.shape )

    ax.azim = 70
    ax.elev += 30
    ax.plot_surface( x, y, z )

    plt.savefig( p )
    plt.close()

