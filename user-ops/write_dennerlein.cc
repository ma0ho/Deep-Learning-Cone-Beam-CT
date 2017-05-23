#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <string>
#include <fstream>


using namespace tensorflow;
using std::string;
using std::ofstream;

REGISTER_OP("WriteDennerlein")
   .Attr("T: {float, double}")
   .Input("filename: string")
   .Input("data: T")
;

template<typename T>
class WriteDennerleinOp : public OpKernel
{

protected:
   template<typename TT>
   inline void write( std::ofstream& s, const TT* v, size_t N = 1 )
   {
      s.write( reinterpret_cast<const char*>( v ), N * sizeof( TT ) ); 
   }

public:

   explicit WriteDennerleinOp(OpKernelConstruction* context) : OpKernel(context) {
   }

   void Compute( OpKernelContext* context ) override
   {
      // grab input
      const string filename = context->input( 0 ).scalar<string>()();
      const auto data = context->input( 1 );
      const auto data_tensor = data.flat<T>();
      const auto shape = data.shape();
      const unsigned short Z = shape.dim_size( 0 ), Y = shape.dim_size( 1 ), X = shape.dim_size( 2 );

      std::ofstream s( filename, std::ios_base::out | std::ios_base::binary );

      // write data
      write( s, &X );
      write( s, &Y );
      write( s, &Z );
      write( s, data_tensor.data(), X*Y*Z );
     
      s.flush();
      s.close();
   }

};

REGISTER_KERNEL_BUILDER(
      Name( "WriteDennerlein" ).Device( DEVICE_CPU ).TypeConstraint<float>("T"),
      WriteDennerleinOp<float>
);

REGISTER_KERNEL_BUILDER(
      Name( "WriteDennerlein" ).Device( DEVICE_CPU ).TypeConstraint<double>("T"),
      WriteDennerleinOp<double>
);


