#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cuda.h>
#include "backproject.hcu"
#include <iostream>

using namespace std;

// from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ float geom_[MAX_PROJ_STACK_SIZE * 12];
__constant__ int3 proj_shape_;
__constant__ int3 vol_shape_;
__constant__ float3 vol_orig_;
__constant__ float3 voxel_size_;
static int3 proj_shape_host_;
static int3 vol_shape_host_;
texture<float, cudaTextureType2D, cudaReadModeElementType> projTex_;

inline __device__
float3 map( float3&& vp, int n )
{
   const float* matrix = &(geom_[n*12]);
   return make_float3(
         matrix[0] * vp.x + matrix[1] * vp.y + matrix[2] * vp.z + matrix[3],
         matrix[4] * vp.x + matrix[5] * vp.y + matrix[6] * vp.z + matrix[7],
         matrix[8] * vp.x + matrix[9] * vp.y + matrix[10] * vp.z + matrix[11]
   );
}

__global__
void kernel( size_t projOffset,
             float* vol,
             const float* proj )
{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;
   const int k = blockIdx.z*blockDim.z + threadIdx.z;
   
   if( i >= vol_shape_.x  || j >= vol_shape_.y || k >= vol_shape_.z )
      return;

   const float x = i*voxel_size_.x + vol_orig_.x;
   const float y = j*voxel_size_.y + vol_orig_.y;
   const float z = k*voxel_size_.z + vol_orig_.z;

   float val = 0.0f;
   
   for( int n = 0; n < proj_shape_.z; ++n )
   {
      auto ip = map( make_float3( x, y, z ), n );

      ip.z = 1.0f / ip.z;
      ip.x *= ip.z;
      ip.y *= ip.z;

      // check if we are within detector (x coord is checked by tex2D)
      if( ip.y >= 0.5 && ip.y <= ((float)proj_shape_.y) - 0.5 )
      {
         // projections in 2D texture are stacked in y direction..
         //float offsY = proj_shape_.x * proj_shape_.y * n;
         //float offsX = projOffset / sizeof(float);
         //val += tex2D( projTex_, ip.x + offsX, ip.y + offsY ) * ip.z * ip.z;
         if( ip.x >= 0.5 && ip.x <= ((float)proj_shape_.x) - 0.5 ) {
            int x = (int)ip.x;
            int y = (int)ip.y;
            val += proj[n*(proj_shape_.x+4)*proj_shape_.y + y*(proj_shape_.x+4) + x ] * ip.z*ip.z;
         }
      }
   }

   // linear volume address
   const unsigned int l = vol_shape_.x * ( k*vol_shape_.y + j ) + i;
   vol[l] = val;
}

__host__
void cuda_init_backproject( float* geom,
                            int U, int V, int N,
                            int X, int Y, int Z,
                            float ox, float oy, float oz,
                            float sx, float sy, float sz )
{
   proj_shape_host_ = make_int3( U, V, N );
   vol_shape_host_ = make_int3( X, Y, Z );
   auto vol_orig = make_float3( ox, oy, oz );
   auto voxel_size = make_float3( sx, sy, sz );

   gpuErrchk( cudaMemcpyToSymbol( geom_, geom, 12 * sizeof(float) * N ) );
   gpuErrchk( cudaMemcpyToSymbol( proj_shape_, &proj_shape_host_, sizeof(int3) ) );
   gpuErrchk( cudaMemcpyToSymbol( vol_shape_, &vol_shape_host_, sizeof(int3) ) );
   gpuErrchk( cudaMemcpyToSymbol( vol_orig_, &vol_orig, sizeof(float3) ) );
   gpuErrchk( cudaMemcpyToSymbol( voxel_size_, &voxel_size, sizeof(float3) ) );
}

__host__
void cuda_backproject( const float* proj, float* vol )
{
   projTex_.addressMode[0] = cudaAddressModeBorder;
   projTex_.addressMode[1] = cudaAddressModeBorder;
   projTex_.addressMode[2] = cudaAddressModeBorder;
   projTex_.filterMode = cudaFilterModeLinear;
   projTex_.normalized = false;
   auto projDesc = cudaCreateChannelDesc<float>();
   size_t projOffset = 0;
   gpuErrchk( cudaBindTexture2D( &projOffset, projTex_, vol, projDesc,
            proj_shape_host_.x, proj_shape_host_.y,
            ( proj_shape_host_.x + (8-proj_shape_host_.x%8) ) * sizeof(float) )
         );

   // launch kernel
   const unsigned int gridsize_x = (vol_shape_host_.x-1) / BLOCKSIZE_X + 1;
   const unsigned int gridsize_y = (vol_shape_host_.y-1) / BLOCKSIZE_Y + 1;
   const unsigned int gridsize_z = (vol_shape_host_.z-1) / BLOCKSIZE_Z + 1;
   const dim3 grid = dim3( gridsize_x, gridsize_y, gridsize_z );
   const dim3 block = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z );
   kernel<<< grid, block >>>( projOffset, vol, proj );

   // cleanup
   gpuErrchk( cudaUnbindTexture( projTex_ ) );
}

#endif


