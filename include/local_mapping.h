#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

# include "multi_frame.h"

namespace MCVORBSLAM
{
    class LocalMapping
    {
    public:
        // Function called in loop.
        void Run();
        
        // Set frame's info for feature matching.
        void SetFrameStatus(const MultiFrame& frame);
    };
}

#endif // LOCALMAPPING_H
