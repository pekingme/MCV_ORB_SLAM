#include "converter.h"

namespace MCVORBSLAM
{
    
    Matx61d Converter::HomogeneousToVec ( const Matx44d &H )
    {
        Matx33d R = H.get_minor<3, 3> ( 0, 0 );
        Matx31d t = H.get_minor<3, 1> ( 0, 3 );
        Matx31d rodrigues;
        Rodrigues ( R, rodrigues );

        Matx61d vec (
            rodrigues ( 0, 0 ), rodrigues ( 1, 0 ), rodrigues ( 2, 0 ),
            t ( 0, 0 ), t ( 1, 0 ), t ( 2, 0 ) );

        return vec;
    }

    Matx44d Converter::VecToHomogeneous ( const Matx61d &vec )
    {
        Matx31d rodrigues = vec.get_minor<3, 1> ( 0, 0 );
        Matx31d t = vec.get_minor<3, 1> ( 3, 0 );
        Matx33d R;
        Rodrigues ( rodrigues, R );

        Matx44d H (
            R ( 0, 0 ),  R ( 0, 1 ), R ( 0, 2 ), t ( 0, 0 ),
            R ( 1, 0 ),  R ( 1, 1 ), R ( 1, 2 ), t ( 1, 0 ),
            R ( 2, 0 ),  R ( 2, 1 ), R ( 2, 2 ), t ( 2, 0 ),
            0.0, 0.0, 0.0, 1.0 );

        return H;
    }

    Matx44d Converter::InvertHomogeneous ( const Matx44d &H )
    {
        Matx33d R = H.get_minor<3, 3> ( 0, 0 );
        Matx31d t = H.get_minor<3, 1> ( 0, 3 );
        R = R.t();
        t = -R * t;

        Matx44d H_inv (
            R ( 0, 0 ), R ( 0, 1 ), R ( 0, 2 ), t ( 0, 0 ),
            R ( 1, 0 ), R ( 1, 1 ), R ( 1, 2 ), t ( 1, 0 ),
            R ( 2, 0 ), R ( 2, 1 ), R ( 2, 2 ), t ( 2, 0 ),
            0.0, 0.0, 0.0, 1.0 );

        return H_inv;
    }
}
