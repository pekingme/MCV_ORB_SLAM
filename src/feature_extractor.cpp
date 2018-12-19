#include "feature_extractor.h"

namespace MCVORBSLAM
{
    FeatureExtractor::FeatureExtractor ( const int feature_count, const int level_count, const float level_scale, const bool learn_mask,
                                         const bool use_agast, const int fast_agast_threshold, const int fast_agast_type,
                                         const bool use_mdBRIEF, const int descriptor_size, const int score_type )
        : feature_count_ ( feature_count ), level_count_ ( level_count ), level_scale_ ( level_scale ), learn_mask_ ( learn_mask ),
          use_agast_ ( use_agast ), fast_agast_threshold_ ( fast_agast_threshold ), fast_agast_type_ ( fast_agast_type ),
          use_mdBRIEF_ ( use_mdBRIEF ), descriptor_size_ ( descriptor_size ), score_type_ ( score_type ) {}

    void FeatureExtractor::operator() ( InputArray frame, vector<KeyPoint> *keypoints, const CameraModel &camera_model,
                                        OutputArray descriptors, OutputArray descriptor_masks, const bool initializing )
    {
        // TODO extract features with lower criterias if initializing, otherwise use regular criterias.
    }

}
