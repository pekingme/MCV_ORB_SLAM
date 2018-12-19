#include "camera_system_pose.h"

namespace MCVORBSLAM
{
    CameraSystemPose::CameraSystemPose ( const CameraSystem &camera_system )
    {
        pose_ = Matx44d::eye();
        pose_inv_ = Matx44d::eye();
        pose_vec_ = Matx61d::zeros();
        camera_poses_ = vector<Matx44d> ( camera_system.GetCameraCount(), Matx44d::eye() );
        camera_poses_inv_ = vector<Matx44d> ( camera_system.GetCameraCount(), Matx44d::eye() );
    }
}
