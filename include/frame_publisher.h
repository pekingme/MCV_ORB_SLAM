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
      // Constructor
        FramePublisher ( Map *map, const string &yaml_path );
        
        // Update from tracking process.
        void Update();
    };
}

#endif // FRAMEPUBLISHER_H
