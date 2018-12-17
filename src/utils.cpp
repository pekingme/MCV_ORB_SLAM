#include "include/utils.h"

#include <iostream>

using namespace std;

namespace MCVORBSLAM
{
bool Utils::FileExists ( const string &file_path )
{
    if ( FILE*file = fopen ( file_path.c_str(),  "r" ) )
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
}
