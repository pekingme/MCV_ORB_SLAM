#ifndef FRAMEPUBLISHER_H
#define FRAMEPUBLISHER_H

# include "map.h"

# include <string>

using namespace std;

namespace MCVORBSLAM
{
    class FramePublisher
    {
    public:
        FramePublisher ( Map *map, const string &yaml_path );
    };
}

#endif // FRAMEPUBLISHER_H
