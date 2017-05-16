#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <math.h>


using namespace tensorflow;
using shape_inference::ShapeHandle; 
using std::cerr;
using std::endl;


REGISTER_OP("Backproject")
   .Input("projections: float")
   .Input("geom: float")
   .Attr("vol_shape: shape")
   .Attr("vol_origin: tensor")
   .Output("vol: float")
   .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
   {
      TensorShapeProto sp;
      ShapeHandle sh;
      auto status = c->GetAttr( "vol_shape", &sp );
      status.Update( c->MakeShapeFromShapeProto( sp, &sh ) );
      c->set_output( 0, sh );
      return status;
   } )
;

class BackprojectOp : public OpKernel
{

private: 
   TensorShape vol_shape_;
   int X_, Y_, Z_;
   float ox_, oy_, oz_;

public:

   explicit BackprojectOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK( context, context->GetAttr( "vol_shape", &vol_shape_ ) );
      Z_ = vol_shape_.dim_size( 0 );
      X_ = vol_shape_.dim_size( 1 );
      Y_ = vol_shape_.dim_size( 2 );

      Tensor t;
      OP_REQUIRES_OK( context, context->GetAttr( "vol_origin", &t ) );
      auto et = t.tensor<float, 1>();
      oz_ = et( 0 );
      ox_ = et( 1 );
      oy_ = et( 2 );
   }

   void Compute(OpKernelContext* context) override
   {
      // grab input
      const auto proj = context->input( 0 ).tensor<float, 3>();
      const auto geom = context->input( 1 ).tensor<float, 3>();
      const int N = proj.dimension( 0 );   // number of projections
      const int U = proj.dimension( 2 );   // size of projection planes
      const int V = proj.dimension( 1 );

      // create output
      Tensor* volume_tensor = nullptr;
      OP_REQUIRES_OK( context, context->allocate_output( 0,
               vol_shape_, &volume_tensor ) );
      auto volume = volume_tensor->tensor<float, 3>();
      volume.setZero();

      // normalize projection matrices such that backprojection
      // weight (third row * voxel) is 1 at isocenter
      //
      // assumption: volume is centered at isocenter
      const float iso[] = { X_ / 2.0f + ox_, Y_ / 2.0f + oy_, Z_ / 2.0f + oz_  };
      Eigen::Tensor<float, 3> geom_norm( N, 3, 4 );

      // foreach projection
      for( int n = 0; n < N; ++n )
      {
         // persp = third row * voxel
         float persp = geom(n,2,3);
         persp += geom(n,2,0) * iso[0] + geom(n,2,1) * iso[1] + geom(n,2,2) * iso[2];
         persp = 1 / persp;

         // loop over matrix
         for( int i = 0; i < 3; ++i )
         {
            for( int j = 0; j < 4; ++j )
            {
               geom_norm(n,i,j) = geom(n,i,j) * persp;
            }
         }
      }

      // iterate over slices
      for( int z = 0; z < Z_; ++z )
      {
         // iterate within slice z
         for( int x = 0; x < X_; ++x )
         {
            for( int y = 0; y < Y_; ++y )
            {
               // iterate over projections
               for( int n = 0; n < N; ++n )
               {
                  float cx = x + ox_, cy = y + oy_, cz = z + oz_;

                  float d = 1 / ( geom_norm(n,2,0) * cx + geom_norm(n,2,1) * cy + geom_norm(n,2,2) * cz + geom_norm(n,2,3) );
                  float u = ( geom_norm(n,0,0) * cx + geom_norm(n,0,1) * cy + geom_norm(n,0,2) * cz + geom_norm(n,0,3) ) * d;
                  float v = ( geom_norm(n,1,0) * cx + geom_norm(n,1,1) * cy + geom_norm(n,1,2) * cz + geom_norm(n,1,3) ) * d;

                  // four neighbours on projection
                  int u1 = ((int)u), v1 = ((int)v);
                  int u2 = u1+1, v2 = v1+1;
                 
                  if( u1 >= 0 && v1 >= 0 && u2 < U && v2 < V )
                  {
                     float wu = u - ((float)u1);
                     float wv = v - ((float)v1);
                     float i1 = proj(n,v1,u1) * ( 1.0f - wu ) + proj(n,v1,u2) * wu;
                     float i2 = proj(n,v2,u1) * ( 1.0f - wu ) + proj(n,v2,u2) * wu;
                     volume(z,y,x) += ( i1 * ( 1.0f - wv ) + i2 * wv ) * d * d;
                  }
               }
            }
         }
      }

   }

};

REGISTER_KERNEL_BUILDER( Name( "Backproject" ).Device( DEVICE_CPU ), BackprojectOp );



