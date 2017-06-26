#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <math.h>
#include "backproject.hcu"
#include <assert.h>


using namespace tensorflow;
using shape_inference::ShapeHandle; 
using std::cerr;
using std::endl;


REGISTER_OP("Backproject")
   .Input("projections: float")
   .Attr("geom: tensor")
   .Attr("vol_shape: shape")
   .Attr("proj_shape: shape")
   .Attr("vol_origin: tensor")
   .Attr("voxel_dimen: tensor")
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


REGISTER_OP("Project")
   .Input("volume: float")
   .Attr("geom: tensor")
   .Attr("vol_shape: shape")
   .Attr("proj_shape: shape")
   .Attr("vol_origin: tensor")
   .Attr("voxel_dimen: tensor")
   .Output("projections: float")
   .SetShapeFn( []( ::tensorflow::shape_inference::InferenceContext* c )
   {
      TensorShapeProto sp;
      ShapeHandle sh;
      auto status = c->GetAttr( "proj_shape", &sp );
      status.Update( c->MakeShapeFromShapeProto( sp, &sh ) );
      c->set_output( 0, sh );
      return status;
   } )
;

class BackprojectOp : public OpKernel
{

protected: 

   // volume size in voxels
   TensorShape vol_shape_;
   int X_, Y_, Z_;

   // number of projections
   TensorShape proj_shape_;
   int U_, V_, N_;

   // volume origin in mm
   float ox_, oy_, oz_;

   // voxel dimensions in mm
   float sx_, sy_, sz_;

   // projection matrices
   Eigen::Tensor<float, 3, Eigen::RowMajor> geom_;
   

public:

   explicit BackprojectOp(OpKernelConstruction* context) : OpKernel(context) {
      OP_REQUIRES_OK( context, context->GetAttr( "vol_shape", &vol_shape_ ) );
      Z_ = vol_shape_.dim_size( 0 );
      Y_ = vol_shape_.dim_size( 1 );
      X_ = vol_shape_.dim_size( 2 );

      OP_REQUIRES_OK( context, context->GetAttr( "proj_shape", &proj_shape_ ) );
      N_ = proj_shape_.dim_size( 0 );
      V_ = proj_shape_.dim_size( 1 );
      U_ = proj_shape_.dim_size( 2 );

      Tensor t;
      OP_REQUIRES_OK( context, context->GetAttr( "vol_origin", &t ) );
      auto et = t.tensor<float, 1>();
      oz_ = et( 0 );
      oy_ = et( 1 );
      ox_ = et( 2 );

      Tensor s;
      OP_REQUIRES_OK( context, context->GetAttr( "voxel_dimen", &s ) );
      auto es = s.tensor<float, 1>();
      sz_ = es( 0 );
      sy_ = es( 1 );
      sx_ = es( 2 );

      Tensor g;
      OP_REQUIRES_OK( context, context->GetAttr( "geom", &g ) );
      auto gt = g.tensor<float, 3>();
      N_ = gt.dimension( 0 );
      geom_ = Eigen::Tensor<float, 3, Eigen::RowMajor>( N_, 3, 4 );

      // normalize projection matrices such that backprojection
      // weight (third row * voxel) is 1 at isocenter
      //
      // assumption: volume is centered at isocenter
      const float iso[] = {
         X_/2.0f * sx_ + ox_,
         Y_/2.0f * sy_ + oy_,
         Z_/2.0f * sz_ + oz_
      };

      // foreach projection
      for( int n = 0; n < N_; ++n )
      {
         // persp = third row * voxel
         float persp = gt(n,2,3);
         persp += gt(n,2,0) * iso[0] + gt(n,2,1) * iso[1] + gt(n,2,2) * iso[2];
         persp = 1 / persp;

         // loop over matrix
         for( int i = 0; i < 3; ++i )
         {
            for( int j = 0; j < 4; ++j )
            {
               geom_(n,i,j) = gt(n,i,j) * persp;
            }
         }
      }
   }

   void Compute(OpKernelContext* context) override
   {
      // grab input
      const auto proj = context->input( 0 ).tensor<float, 3>();
      // TODO: Check that #projections == #matrices
      const int U = proj.dimension( 2 );   // size of projection planes
      const int V = proj.dimension( 1 );

      // create output
      Tensor* volume_tensor = nullptr;
      OP_REQUIRES_OK( context, context->allocate_output( 0,
               vol_shape_, &volume_tensor ) );
      auto volume = volume_tensor->tensor<float, 3>();
      volume.setZero();

      // iterate over slices
      for( int z = 0; z < Z_; ++z )
      {
         // iterate within slice z
         for( int y = 0; y < Y_; ++y )
         {
            for( int x = 0; x < X_; ++x )
            {
               // iterate over projections
               for( int n = 0; n < N_; ++n )
               {
                  float cx = x*sx_ + ox_,
                        cy = y*sy_ + oy_,
                        cz = z*sz_ + oz_;

                  float d = 1 / ( geom_(n,2,0) * cx + geom_(n,2,1) * cy + geom_(n,2,2) * cz + geom_(n,2,3) );
                  float u = ( geom_(n,0,0) * cx + geom_(n,0,1) * cy + geom_(n,0,2) * cz + geom_(n,0,3) ) * d;
                  float v = ( geom_(n,1,0) * cx + geom_(n,1,1) * cy + geom_(n,1,2) * cz + geom_(n,1,3) ) * d;

                  // four neighbours on projection
                  int u1 = ((int)u),
                      v1 = ((int)v);
                  int u2 = u1+1,
                      v2 = v1+1;
                 
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


class BackprojectCudaOp : public BackprojectOp
{

public:
   
   explicit BackprojectCudaOp(OpKernelConstruction* context) : BackprojectOp(context) {
      assert( N_ <= MAX_PROJ_STACK_SIZE ); 

      cuda_init_backproject( geom_.data(),
                             U_, V_, N_,
                             X_, Y_, Z_,
                             ox_, oy_, oz_,
                             sx_, sy_, sz_ );
   }

   void Compute(OpKernelContext* context) override
   {
      // grab input
      const auto proj = context->input( 0 ).tensor<float, 3>();
      // TODO: Check that #projections == #matrices

      // create output
      Tensor* volume_tensor = nullptr;
      OP_REQUIRES_OK( context, context->allocate_output( 0,
               vol_shape_, &volume_tensor ) );
      auto volume = volume_tensor->tensor<float, 3>();

      cuda_backproject( proj.data(), volume.data() );
   }

};


class ProjectCudaOp : public BackprojectCudaOp
{

public:
   
   explicit ProjectCudaOp(OpKernelConstruction* context) : BackprojectCudaOp(context) {}

   void Compute(OpKernelContext* context) override
   {
      // grab input
      const auto vol = context->input( 0 ).tensor<float, 3>();
      // TODO: Check that #projections == #matrices

      // create output
      Tensor* proj_tensor = nullptr;
      OP_REQUIRES_OK( context, context->allocate_output( 0,
               proj_shape_, &proj_tensor ) );
      auto proj = proj_tensor->tensor<float, 3>();

      cuda_project( vol.data(), proj.data() );
   }

};

REGISTER_KERNEL_BUILDER( Name( "Backproject" ).Device( DEVICE_CPU ), BackprojectOp );
REGISTER_KERNEL_BUILDER( Name( "Backproject" ).Device( DEVICE_GPU ), BackprojectCudaOp );

REGISTER_KERNEL_BUILDER( Name( "Project" ).Device( DEVICE_GPU ), ProjectCudaOp );


