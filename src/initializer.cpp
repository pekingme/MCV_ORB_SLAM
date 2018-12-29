#include "initializer.h"

namespace MCVORBSLAM
{
    Initializer::Initializer ( const MultiFrame &current_frame )
    {
        reference_frame_ = current_frame;
        keypoints_1_ = current_frame.GetKeyPoints();
        keyrays_1_ = current_frame.GetKeyRays();
        camera_system_ = current_frame.GetCameraSystem();
    }

}
