#ifndef ORBMATCHER_H
#define ORBMATCHER_H

# include "multi_frame.h"

namespace MCVORBSLAM
{
    class ORBMatcher
    {
    public:
        // Constructor
        ORBMatcher ( const double nnratio = 0.6, const int descriptor_size = 32, const bool check_orientation = true, const bool use_mask = false );

        // Search matches for initialization.
        int SearchForInitialization ( const MultiFrame &frame1, const MultiFrame &frame2, vector<Vec2d> *keypoint_positions, vector<int> *matches_12, const int window_size =10 );

    private:

        // The ratio of the distance of the best match to the second best match.
        double nn_ratio_;

        // The size of descriptor.
        int descriptor_size_;

        // Whether consider orientation of features to be matched.
        bool check_orientation_;

        // Use mask while matching descriptors.
        bool use_mask_;

        // Lower bound of threshold of distance to be considered as matched.
        int threshold_low_;

        // Higher bound of threshold of distance to be considered as matched.
        int threshold_high_;
        
        // Orientation histogram length
        static const int HISTO_LENGTH;
    };
}

#endif // ORBMATCHER_H
