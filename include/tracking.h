#ifndef TRACKING_H
#define TRACKING_H

# include "system.h"
# include "orb_vocabulary.h"
# include "map.h"
# include "multi_frame.h"
# include "frame_publisher.h"
# include "feature_extractor.h"

# include <mutex>

# include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace MCVORBSLAM
{
    class System;
    class Map;
    class FramePublisher;

    class Tracking
    {
    public:

        // Constructor
        Tracking ( Map *const map, FramePublisher *const frame_publisher, const vector<FeatureExtractor *> &extractors );

        // Tracking states enum.
        enum TrackingState
        {
            SYSTEM_NOT_READY = -1,
            NO_IMAGES_YET = 0,
            NOT_INITIALIZED = 1,
            INITIALIZING = 2,
            WORKING = 3,
            LOST = 4
        };

        // Process multi frame captured by camera system. Input frames must be grayscale.
        void ProcessMultiFrame ( const vector<Mat> &multi_frame, const double timestamp );

    private:
        // Perform stateful tracking for multi frame.
        void Track();

        // Initialization phase: construct a initi map.
        void Initialize();

        // Track current frame with last processed frame. Used when no motion model used,
        // or not enough keyframe in map, or a recent frame just be processed using
        // relocalization, or tracking with motion model fails.
        bool TrackWithLastFrame();

        // Track current frame with motion predicted based on last processed frame.
        bool TrackWithMotionModel();

        // Relocalize current frame from all known keyframes if tracking is lost, or manual
        // requested; otherwise, relocalize against local window around last keyframe.
        bool Relocalize();

        // With initial camera pose, search map points in local map and optimize camera pose
        bool TrackLocalMap();

        // Check if current frame should be used as a keyframe.
        bool NeedNewKeyFrame();

        // Create a new keyframe from current frame.
        void CreateNewKeyFrame();



        // Reset all components and state.
        void Reset();

        // Thread safe: if relocalization is requested.
        bool RelocalizationRequested();




        // Map
        Map *map_;

        // Frame publisher
        FramePublisher *frame_publisher_;

        // Feature extractors
        vector<FeatureExtractor *> extractors_;





        // Current tracking state
        TrackingState current_state_;

        // Last processed tracking state
        TrackingState last_processed_state_;

        // Current processing multi frame
        MultiFrame *current_frame_;

        // Last processed multi frame
        MultiFrame *last_frame_;

        // Current predicted velocity.
        Matx44d velocity_;

        // Last relocalized multi frame id
        int last_relocalized_frame_id_;

        // Motion model is in use
        bool use_motion_model;

        // Relocalization is manually requested, thread safe needed
        bool manual_relocalization_requested;

        // Mutex
        mutex mutex_manual_relocalization_requested;
    };
}

#endif // TRACKING_H
