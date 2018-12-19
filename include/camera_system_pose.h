#ifndef CAMERAPOSE_H
#define CAMERAPOSE_H

# include "camera_system.h"

# include <vector>

# include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{
    class CameraSystemPose
    {
    public:
        // Default constructor, should never use this.
        CameraSystemPose() {}
        
        // Actual constructor.
        CameraSystemPose(const CameraSystem& camera_system);

        // Camera system pose: Pt_world = {pose} * Pt_system
        Matx44d pose_;

        // Camera system transform: Pt_system = {transform} * Pt_world;
        Matx44d pose_inv_;

        // Camera system pose in vector form: [rotation_rodrigues, translation]
        Matx61d pose_vec_;

        // Absolute poses of all cameras: Pt_world = Pose * Pt_camera
        vector<Matx44d> camera_poses_;

        // Absolute transform of all cameras: Pt_camera = Transform * Pt_world
        vector<Matx44d> camera_poses_inv_;
    };
}

#endif // CAMERAPOSE_H
