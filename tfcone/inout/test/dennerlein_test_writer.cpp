#include <fstream>
#include <iostream>

struct __attribute__((__packed__)) Dennerlein32 {

   // header
   unsigned short x = 3;
   unsigned short y = 4;
   unsigned short z = 6;

   // data
   float data[3*4*6];

};

struct Dennerlein64 {

   // header
   unsigned short x = 3;
   unsigned short y = 4;
   unsigned short z = 6;

   // data
   double data[3*4*6];

};

int main() {
   {
      Dennerlein32 d32;
      for( int i = 0; i < d32.x*d32.y*d32.z; ++i )
         d32.data[i] = i/2.0;

      std::ofstream ofstr("dennerlein32.bin", std::ios_base::out | std::ios_base::binary);
      ofstr.write( reinterpret_cast<char*>(&d32), sizeof(Dennerlein32) );
      ofstr.close();
   }
   
   {
      Dennerlein64 d64;
      for( int i = 0; i < d64.x*d64.y*d64.z; ++i )
         d64.data[i] = i/2.0;

      std::ofstream ofstr("dennerlein64.bin");
      ofstr.write( reinterpret_cast<char*>(&d64), sizeof(Dennerlein64) );
      ofstr.close();
   }
}
