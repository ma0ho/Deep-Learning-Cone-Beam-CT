class ShapeProj:
    def __init__( self, N, W, H ):
        self.N = N
        self.W = W
        self.H = H

    def toNCHW( self ):
        return [ self.N, self.H, self.W ]

    def size( self ):
        return self.N * self.W * self.H


class Coord3D:
    def __init__( self, X, Y, Z ):
        self.X = X
        self.Y = Y
        self.Z = Z

    def toNCHW( self ):
        return [ self.Z, self.Y, self.X ]


class Shape2D:
    def __init__( self, W, H ):
        self.W = W
        self.H = H

    def toNCHW( self ):
        return [ self.H, self.W ]

    def size( self ):
        return self.W * self.H



class Shape3D:
    def __init__( self, W, H, D ):
        self.W = W
        self.H = H
        self.D = D

    def toNCHW( self ):
        return [ self.D, self.H, self.W ]

    def size( self ):
        return self.D * self.W * self.H


