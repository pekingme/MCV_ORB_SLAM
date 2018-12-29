#ifndef INITIALIZER_H
#define INITIALIZER_H

# include "multi_frame.h"


namespace MCVORBSLAM
{
    class Initializer
    {
    public:
      Initializer(const MultiFrame& current_frame);
      
    private:
      
      // Reference frame.
      MultiFrame reference_frame_;
      
      // Keypoints in reference frame.
      vector<KeyPoint> keypoints_1_;
      
      // Keyrays in reference frame.
      vector<Vec3d> keyrays_1_;
      
      // Camera system
      CameraSystem* camera_system_;
    };
}

#endif // INITIALIZER_H
