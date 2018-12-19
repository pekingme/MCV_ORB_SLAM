#ifndef CAMERASYSTEM_H
#define CAMERASYSTEM_H

# include "camera_model.h"

# include <string>
# include <vector>

# include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{
    class CameraSystem
    {
    public:
        // Default constructor, should never use directly.
        CameraSystem() {}

        // Utility function load a camera system calibration from YAML file.
        // Camera system pose is initialized as identity.
        static CameraSystem LoadFromYaml ( const string &yaml_path );



        CameraModel GetCameraModel ( const int camera_index ) const
        {
            return camera_models_[camera_index];
        }

        int GetCameraCount() const
        {
            return camera_count_;
        }

    private:
        // Inner constructor.
        CameraSystem ( const vector<CameraModel> &camera_models, const vector<Matx44d> &camera_relative_poses );

        // Number of cameras in this camera system.
        int camera_count_;

        // Camera models
        vector<CameraModel> camera_models_;

        // Relative poses of each single camera: Pt_system = Pose * Pt_camera
        vector<Matx44d> camera_relative_poses_;

        // Relative poses of each single camera in vector form: [rotation_rodrigues, translation]
        vector<Matx61d> camera_relative_poses_vec_;

    };
}
#endif // CAMERASYSTEM_H
