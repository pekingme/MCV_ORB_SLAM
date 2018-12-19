#ifndef SYSTEM_H
#define SYSTEM_H

# include "camera_system.h"
# include "orb_vocabulary.h"
# include "feature_extractor.h"
# include "map.h"
# include "keyframe_database.h"
# include "tracking.h"
# include "local_mapping.h"
# include "loop_closing.h"
# include "viewer.h"
# include "map_publisher.h"
# include "frame_publisher.h"

# include <vector>
# include <thread>

using namespace std;

namespace MCVORBSLAM
{
    class Viewer;
    class Tracking;

    class System
    {
    public:
        // Initialize the processing components: tracking, local mapping, loop closing, and viewer.
        // Should be the last step before start.
        void Initialize ( const string &yaml_path, const bool init_tracking = true, const bool init_local_mapping = true,
                          const bool init_loop_closing = true, const bool init_viewer = true );

        // Load feature extractor based on parameters in YAML file.
        void LoadExtractor ( const string &yaml_path );

        // Load ORB vocabulary from file.
        void LoadVocabulary ( const string &vocabulary_path );

        // Process the given frames captured by a camera system. Images must be synchronized.
        // This is stateful, and will manage tracking state. Input images: BGR (CV_8UC3)
        // or grayscale (CV_8U). BGR will be converted to grayscale if needed.
        void TrackMultiFrame ( vector<Mat> *const multi_frame, const double &timestamp );

        void SetCameraSystem ( const CameraSystem &camera_system )
        {
            camera_system_ = camera_system;

            cout << endl << "Camera system calibration is loaded." << endl;
        }

    private:
        // Camera system calibration for current video set related functions.
        CameraSystem camera_system_;

        // Extractors coupled with cameras in camera system
        vector<FeatureExtractor *> feature_extractors_;

        // ORB vocabulary.
        ORBVocabulary *vocabulary_;

        // Map
        Map *map_;


        // Camera models
        vector<CameraModel> camera_models_;
        // Keyframe database
        KeyframeDatabase *keyframe_database_;

        // Tracking process
        Tracking *tracking_;

        // Local mapping process
        LocalMapping *local_mapping_;
        thread *local_mapping_thread_;

        // Loop closing process
        LoopClosing *loop_closing_;
        thread *loop_closing_thread_;

        // Viewer, use Pangolin
        Viewer *viewer_;
        MapPublisher *map_publisher_;
        FramePublisher *frame_publisher_;
        thread *viewer_thread_;

    };
}

#endif // SYSTEM_H
