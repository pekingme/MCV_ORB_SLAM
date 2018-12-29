#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

# include "multi_frame.h"

namespace MCVORBSLAM
{
    class LoopClosing
    {
    public:
        // Function called in loop.
        void Run();
        
        // Set frame's info for feature matching.
        void SetFrameStatus(const MultiFrame& frame);
    };
}

#endif // LOOPCLOSING_H
