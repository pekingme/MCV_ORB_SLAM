#ifndef UTILS_H
#define UTILS_H

#include <string>
# include <chrono>
# include <vector>

using namespace std;

namespace MCVORBSLAM
{
    class Utils
    {
    public:
        // Return if a file exists.
        static bool FileExists ( const string &file_path );

        // Return number of milliseconds between two time points.
        static double DurationInMilliseconds ( const chrono::high_resolution_clock::time_point &start,
                                               const chrono::high_resolution_clock::time_point &end );

	// Evaluate polynomial equation with given input using Horner method.
        static double Horner ( const double* coeffs, const int degree, const double x );
    };
}

#endif // UTILS_H
