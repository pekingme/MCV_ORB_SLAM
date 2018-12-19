#include "include/utils.h"

#include <iostream>

using namespace std;

namespace MCVORBSLAM
{

    bool Utils::FileExists ( const string &file_path )
    {
        if ( FILE *file = fopen ( file_path.c_str(),  "r" ) )
        {
            fclose ( file );
            return true;
        }
        else
        {
            cerr << "File not found: " << file_path << endl;
            return false;
        }
    }

    double Utils::DurationInMilliseconds ( const chrono::_V2::system_clock::time_point &start,
                                           const chrono::_V2::system_clock::time_point &end )
    {
        return static_cast<double> ( chrono::duration_cast<chrono::milliseconds> ( end - start ).count() );
    }

    double Utils::Horner ( const double *coeffs, const int degree, const double x )
    {
        double result = 0.0;

        for ( int i = degree - 1; i >= 0; i-- )
        {
            result = result * x + coeffs[i];
        }

        return result;
    }
}
