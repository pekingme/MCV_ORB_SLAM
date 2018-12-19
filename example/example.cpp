#include "utils.h"
#include "system.h"
#include "camera_system.h"
#include "multi_video.h"

#include <vector>
#include <string>
#include <chrono>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace
{
    const char *about = "This example shows how to use MCV_ORB_SLAM.\n"
                        "There are three modes 1) map generating 2) video mapping 3) picture localization.\n"
                        "\tMap generating uses the original SLAM flow to construct the map first time, \n"
                        "\tVideo remapping is to use the generated map to localize all frames with input video, \n"
                        "\tPicture localization is to use the generated map to localize input pictures.\n";
    const char *keys =
        "{help h ?||Print help message}"
        "{@configuration|<none>|Configuration file, see example for details.}"
        "{-m mode|1|Execution mode 1, 2, or 3}";
}

int main ( int argc, char **argv )
{
    // Check arguments.
    CommandLineParser parser ( argc, argv, keys );
    parser.about ( about );

    if ( parser.has ( "help" ) || argc != 3 )
    {
        parser.printMessage();
        return 0;
    }

    string configuration_file_path = parser.get<string> ( "@configuration" );

    if ( !parser.check() )
    {
        parser.printErrors();
        return 0;
    }

    // Check if configuration file exists.
    if ( !MCVORBSLAM::Utils::FileExists ( configuration_file_path ) )
    {
        cerr << "Configuration file doesn't exists." << endl;
        return 0;
    }

    // Load configuration file.
    FileStorage config_file ( configuration_file_path, FileStorage::READ );

    // Create SLAM and execute in different mode.
    MCVORBSLAM::System SLAM;
    int mode = parser.get<int> ( "mode" );

    switch ( mode )
    {
        // map generating: original SLAM
    case 1:
    {
        // Load camera system to System
        string camera_system_yaml_path = config_file["CameraCalibration"];
        MCVORBSLAM::CameraSystem camera_system = MCVORBSLAM::CameraSystem::LoadFromYaml ( camera_system_yaml_path );
        SLAM.SetCameraSystem ( camera_system );

        // Load extractors and vocabulary to System
        string setting_yaml_path = config_file["SlamSetting"];
        SLAM.LoadExtractor ( setting_yaml_path );
        string vocabulary_path = config_file["Vocabulary"];
        SLAM.LoadVocabulary ( vocabulary_path );

        // Initialize tracking, local mapping, loop closing and viewer
        SLAM.Initialize ( setting_yaml_path );

        // Load synchronized video
        string video_list_path = config_file["SynchedVideos"];
        MCVORBSLAM::MultiVideo multi_video ( video_list_path );

        // Set target frame rate
        multi_video.SetTargetFPS ( 20 );

        // Wait for user to manually start.
        cvWaitKey ( 0 );

        // Main loop to perform SLAM with sampled frames.
        vector<Mat> frames;
        double time_stamp;

        while ( multi_video.HasNext() )
        {
            // Read next frames
            if ( multi_video.Next ( &frames, &time_stamp, true ) )
            {
                SLAM.TrackMultiFrame ( &frames, time_stamp );
            }

            // A chance to stop early.
            char enter = cvWaitKey ( 1 );

            if ( enter == 'q' )
            {
                break;
            }
        }

        // TODO Save map bundle to result

        break;
    }

    // Video remapping: pure localization with video from multi camera system
    case 2:
    {
        // Load camera system to System
        string camera_syste_yaml_path = config_file["CameraCalibration"];
        // Load extractors to System
        string setting_yaml_path = config_file["SlamSetting"];
        // Load map and keyframe database
        string map_binary_path = config_file["ReuseMap"];
        // Initialize tracking and viewer (local mapping and loop closing is not used)
        // Load synchronized video
        // Main loop to perform relocalization with sampled frames
        // Save full trajectory to result
        break;
    }

    // Picture remapping: pure localization with pictures from single camera
    case 3:
    {
        // Initiate client: each client has a fixed camera calibration
        // Load map and keyframe database
        string map_binary_path = config_file["ReuseMap"];
        // Initialize tracking and viewer (local mapping and loop closing is not used)
        // Load client's pictures: load picture files for localization requests
        // Perform localization
        // Save estimated position to result
        break;
    }

    default:
    {
        cerr <<  "Unknown mode" <<  endl;
        exit ( 0 );
    }
    }

}

