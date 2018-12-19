#ifndef MULTIFRAME_H
#define MULTIFRAME_H

# include "map_point.h"
# include "camera_system.h"
# include "camera_system_pose.h"
# include "feature_extractor.h"

# include <unordered_map>
# include <vector>

# include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{
# define FRAME_GRID_ROWS 48
# define FRAME_GRID_COLS 64

    class MultiFrame
    {
    public:

        // Default constructor, should never use directly.
        MultiFrame() {}

        // Actual constructor for use.
        MultiFrame ( const vector<Mat> &frames, const double timestamp, const CameraSystem &camera_system,
                     const vector<FeatureExtractor *> &extractors, const bool initializing );

        // Return camera system pose when this frame was captured.
        Matx44d GetPose()
        {
            return camera_pose_.pose_;
        }

        // Next frame id, using static variable to manage auto increment.
        // Need to be thread safe if used in multiple threads.
        static size_t next_id_;
        
        // An index for this multi frame object.
        size_t id_;

        // Map points observed in this frame.
        vector<MapPoint *> map_points_;

        // Indices of outlier map points.
        vector<int> outliers_;

    private:
	// Find grid cell position based on keypoint position. Return whether find one.
        bool FindGridCell ( const int camera_index, const KeyPoint &keypoint, int *grid_x, int *grid_y );


        // Original frames
        vector<Mat> frames_;

        // Timestamp
        double timestamp_;

        // Camera system calibration
        CameraSystem camera_system_;

        // Feature extractors
        vector<FeatureExtractor *> extractors_;

        // Camera system pose
        CameraSystemPose camera_pose_;

        // Descriptors of keypoints
        vector<Mat> descriptors_;

        // Learned masks of descriptors
        vector<Mat> descriptor_masks_;

        // Resolution of frames
        vector<Size> resolutions_;

        // Sizes of grids in each frame
        vector<Size2d> grid_sizes_;

        // Map of keypoint ids: [camera index][row index][column index] -> contained keypoint ids
        vector<vector<vector<vector<size_t>>>> keypoint_grid_map_;

        // Number of levels in pyramid for each camera.
        vector<int> scale_levels_;

        // Scale factor between levels for each camera.
        vector<double> scale_factors_;

        // Absolute scale factor for each level in each camera.
        vector<vector<double>> scale_abs_factors_;

        // Inversed square of absolute scale factor for each level in each camera.
        // Absolute scale is always greater than 1. This is the inverse square of that.
        vector<vector<double>> scale_abs_factors_inv_sqr_;

        // All keypoints in all frames
        vector<KeyPoint> keypoints_;

        // All keyrays in all frames
        vector<Vec3d> keyrays_;

        // Map from keypoint index to camear index
        unordered_map<size_t, int> keypoint_camera_map_;

        // Map from keypoint index to its image-wise index
        unordered_map<size_t, size_t> keypoint_imagewise_map_;
        
        // Flags a keypoint is outlier.
        vector<bool> keypoint_outliers_;

    };
}

#endif // MULTIFRAME_H
