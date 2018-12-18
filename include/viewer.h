#ifndef VIEWER_H
#define VIEWER_H

# include "frame_publisher.h"
# include "map_publisher.h"
# include "tracking.h"
# include "system.h"

# include <string>

using namespace std;

namespace MCVORBSLAM
{
    class System;
    
    class Viewer
    {
    public:
        // Constructor
        Viewer ( System *system, FramePublisher *frame_publisher, MapPublisher *map_publisher, Tracking *tracking, const string &yaml_path );

        // Function called in loop.
        void Run();
    };
}

#endif // VIEWER_H
