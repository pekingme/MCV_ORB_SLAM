#ifndef UTILS_H
#define UTILS_H

#include <string>

using namespace std;

namespace MCVORBSLAM
{
    class Utils
    {
    public:
        static bool FileExists ( const string &file_path );
    };
}

#endif // UTILS_H
