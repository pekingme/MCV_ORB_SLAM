#ifndef CONVERTER_H
#define CONVERTER_H

# include "opencv2/opencv.hpp"

using namespace cv;

namespace MCVORBSLAM
{
    class Converter
    {
    public:
      // Convert 4x4 homogeneous matrix to 6x1 rotation_rodrigues + translation representation
        static Matx61d HomogeneousToVec ( const Matx44d &H );

        // Convert 6x1 rotation rotation_rodrigues + translation representation to 4x4 homogeneous matrix
        static Matx44d VecToHomogeneous ( const Matx61d &vec );

        // Return inverted homogeneous matrix
        static Matx44d InvertHomogeneous ( const Matx44d &H );
    };
}
#endif // CONVERTER_H
