#ifndef MULTIFRAME_H
#define MULTIFRAME_H

# include "map_point.h"
# include "camera_system.h"
# include "camera_system_pose.h"
# include "feature_extractor.h"

# include <vector>

# include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{
    class MultiFrame
    {
    public:

        // Default constructor, should never use directly.
        MultiFrame() {}
        
        // Actual constructor for use.
        MultiFrame ( vector<Mat> frames, double timestamp, vector<FeatureExtractor *> extractors );

        // Return camera system pose when this frame was captured.
        Matx44d GetPose()
        {
            return camera_pose_.pose_;
        }

        // An index for this multi frame object.
        int id_;

        // Map points observed in this frame.
        vector<MapPoint *> map_points_;

        // Indices of outlier map points.
        vector<int> outliers_;

    private:

        // Camera system calibration
        CameraSystem camera_system_;

        // Camera system pose
        CameraSystemPose camera_pose_;
    };
}

#endif // MULTIFRAME_H
