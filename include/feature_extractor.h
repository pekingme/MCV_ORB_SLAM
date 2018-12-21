#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

# include "camera_model.h"

# include <vector>

# include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace MCVORBSLAM
{
    // Feature extractor detects keypoints from frame in pyramid levels and computes the
    // descriptors. Keypoints detection options include using FAST and AGAST algorithm;
    // BRIEF (as ORB), distorted BRIEF, and masked distorted BRIEF descriptors can be extracted.
    class FeatureExtractor
    {
    public:
        FeatureExtractor ( const int feature_count, const int level_count, const float level_scale, const bool learn_mask,
                           const bool use_agast, const int fast_agast_threshold, const int fast_agast_type,
                           const bool use_distorted_brief, const int descriptor_size, const int score_type );

        // Operator to extract features from frame.
        void operator() ( InputArray frame, vector<KeyPoint> *keypoints, const CameraModel &camera_model,
                          OutputArray descriptors, OutputArray descriptor_masks, const bool initializing );

        double GetScaleFactor() const
        {
            return level_scale_;
        }

        double GetLevelCount() const
        {
            return level_count_;
        }

    private:

        // Computes the pyramid of the image.
        void ComputePyramid ( const Mat &image, const Mat &mask, vector<Mat> *image_pyramid, vector<Mat> *mask_pyramid );

        // Compute keypoints from each level of the image pyramid.
        void ComputeKeypointOctTree ( const vector<Mat> &image_pyramid, const vector<Mat> &mask_pyramid, vector<vector<KeyPoint>> *keypoints );

        // Compute selected descriptors.
        void ComputeDescriptors ( const Mat &level_image, const vector< KeyPoint > &level_keypoints, const vector< Vec2d > &undistorted_keypoints,
                                  const CameraModel &camera_model, Mat *level_descriptors, Mat *level_descriptor_masks );

        // Distribute keypoints to its level.
        void DistributeOctTree ( const vector< KeyPoint > &keypoints, vector< KeyPoint > *level_keypoints,
                                 const int border_x_min, const int border_x_max,
                                 const int border_y_min, const int border_y_max,
                                 const int level );

        // Compute orientation
        void ComputeOrientation ( const Mat &image, vector< KeyPoint > *level_keypoints );

        // Target number of features.
        int feature_count_;

        // Number of levels of pyramid.
        int level_count_;

        // Scale factor between levels in pyramid.
        double level_scale_;

        // Learn descriptor masks
        bool learn_mask_;

        // Use Agast for feature detection instead of FAST algorithm.
        bool use_agast_;

        // Feature detection threshold.
        int fast_agast_threshold_;

        // Feature detection algorithm type.
        int fast_agast_type_;

        // Use distorted BRIEF as descriptor instead of ORB
        bool use_distorted_brief_;

        // Bit size of descriptor
        int descriptor_size_;

        // Scoring algorithm for feature sorting.
        int score_type_;

        // Scale factors of each level in pyramid.
        vector<double> scale_factors_;

        // Number of features per level.
        vector<int> level_feature_count_;

        // Descriptor pattern.
        vector<double> pattern_;

        // The boundary of u to each v for a quater circular patch.
        vector<int> patch_bound_u_;
    };
}

#endif // FEATUREEXTRACTOR_H
