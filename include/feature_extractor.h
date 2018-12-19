#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

# include "camera_model.h"

# include <vector>

# include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace MCVORBSLAM
{
    class FeatureExtractor
    {
    public:
        FeatureExtractor ( const int feature_count, const int level_count, const float level_scale, const bool learn_mask,
                           const bool use_agast, const int fast_agast_threshold, const int fast_agast_type,
                           const bool use_mdBRIEF, const int descriptor_size, const int score_type );

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
      
      // Use mdBRIEF as descriptor instead of ORB
      bool use_mdBRIEF_;
      
      // Bit size of descriptor
      int descriptor_size_;
      
      // Scoring algorithm for feature sorting.
      int score_type_;
    };
}

#endif // FEATUREEXTRACTOR_H
