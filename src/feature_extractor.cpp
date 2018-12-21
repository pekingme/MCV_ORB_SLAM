#include "feature_extractor.h"

namespace MCVORBSLAM
{
    const int EDGE_THRESHOLD = 19;
    const int PATCH_SIZE = 32;

    FeatureExtractor::FeatureExtractor ( const int feature_count, const int level_count, const float level_scale, const bool learn_mask,
                                         const bool use_agast, const int fast_agast_threshold, const int fast_agast_type,
                                         const bool use_distorted_brief, const int descriptor_size, const int score_type )
        : feature_count_ ( feature_count ), level_count_ ( level_count ), level_scale_ ( level_scale ), learn_mask_ ( learn_mask ),
          use_agast_ ( use_agast ), fast_agast_threshold_ ( fast_agast_threshold ), fast_agast_type_ ( fast_agast_type ),
          use_distorted_brief_ ( use_distorted_brief ), descriptor_size_ ( descriptor_size ), score_type_ ( score_type )
    {
        // TODO scale factors
        // TODO assign pattern
        // TODO assign patch bound u

    }

    void FeatureExtractor::operator() ( InputArray frame, vector<KeyPoint> *keypoints, const CameraModel &camera_model,
                                        OutputArray descriptors, OutputArray descriptor_masks, const bool initializing )
    {
        if ( frame.empty() )
        {
            return;
        }

        assert ( frame.type() == CV_8UC1 );

        // Compute the scale pyramid of image.
        vector<Mat> image_pyramid, mask_pyramid;
        Mat frame_mat = frame.getMat();
        Mat mask_mat = camera_model.GetMirrorMask ( 0 );
        ComputePyramid ( frame_mat, mask_mat, &image_pyramid, &mask_pyramid );

        // Compute keypoints in all levels of pyramid.
        vector<vector<KeyPoint>> keypoints_temp;
        ComputeKeypointOctTree ( image_pyramid, mask_pyramid, &keypoints_temp );
        int keypoints_count = 0;

        for ( int i = 0; i < level_count_; i++ )
        {
            keypoints_count += keypoints_temp[i].size();
        }

        // Prepare for descriptor computation.
        Mat descriptors_temp, descriptor_masks_temp;

        if ( keypoints_count == 0 )
        {
            descriptors.release();
            descriptor_masks.release();
        }
        else
        {
            descriptors.create ( keypoints_count, descriptor_size_, CV_8U );
            descriptor_masks.create ( keypoints_count, descriptor_size_, CV_8U );
            descriptors_temp = descriptors.getMat();
            descriptor_masks_temp = descriptor_masks.getMat();
        }

        keypoints->clear();
        keypoints->reserve ( keypoints_count );

        // Compute descriptors
        int row_to_write = 0;
        const double focal_length = -camera_model.GetPoly() [0];

        for ( int level = 0; level < level_count_; level++ )
        {
            vector<KeyPoint> &level_keypoints = keypoints_temp[level];
            int level_keypoints_count = level_keypoints.size();

            if ( level_keypoints_count == 0 )
            {
                continue;
            }

            // Get scaled image and blur it.
            Mat &level_image = image_pyramid[level];
            boxFilter ( level_image, level_image, level_image.depth(), Size ( 5, 5 ), Point ( -1, -1 ), true, BORDER_REFLECT_101 );

            // Undistort keypoint coordinates if distorted BRIEF is used.
            vector<Vec2d> undistorted_keypoints ( level_keypoints_count );
            double scale = scale_factors_[level];

            if ( use_distorted_brief_ )
            {
                for ( int i = 0; i < level_keypoints_count; i++ )
                {
                    camera_model.UndistortPoints ( static_cast<double> ( level_keypoints[i].pt.x * scale ),
                                                   static_cast<double> ( level_keypoints[i].pt.y * scale ),
                                                   focal_length,
                                                   & ( undistorted_keypoints[i] ( 0 ) ),
                                                   & ( undistorted_keypoints[i] ( 1 ) ) );
                }
            }

            Mat level_descriptors = descriptors_temp.rowRange ( row_to_write, row_to_write + level_keypoints_count );
            Mat level_descriptor_masks = descriptor_masks_temp.rowRange ( row_to_write, row_to_write + level_keypoints_count );
            ComputeDescriptors ( level_image, level_keypoints, undistorted_keypoints, camera_model, &level_descriptors, &level_descriptor_masks );

            row_to_write +=  level_keypoints_count;

            // Scale keypoint coordinates
            if ( level > 0 )
            {
                for ( vector<KeyPoint>::iterator keypoint_it = level_keypoints.begin(); keypoint_it != level_keypoints.end(); keypoint_it++ )
                {
                    keypoint_it->pt *= scale;
                }
            }

            // Add keypoints to output.
            keypoints->insert ( keypoints->end(), level_keypoints.begin(), level_keypoints.end() );
        }
    }

    void FeatureExtractor::ComputePyramid ( const Mat &image, const Mat &mask, vector< Mat > *image_pyramid, vector<Mat> *mask_pyramid )
    {
        for ( int level = 0; level < level_count_; level++ )
        {
            // Determine scaled size.
            double scale = 1.0 / scale_factors_[level];
            Size scaled_size ( cvRound ( ( double ) image.cols * scale ), cvRound ( ( double ) image.rows * scale ) );
            Size scaled_size_with_border ( scaled_size.width + EDGE_THRESHOLD * 2, scaled_size.height + EDGE_THRESHOLD * 2 );
            // Initialize scaled image and scaled mask.
            Mat image_temp ( scaled_size_with_border, image.type() ), mask_temp;
            ( *image_pyramid ) [level] = image_temp ( Rect ( EDGE_THRESHOLD, EDGE_THRESHOLD, scaled_size.width, scaled_size.height ) );

            if ( !mask.empty() )
            {
                mask_temp = Mat ( scaled_size_with_border, mask.type() );
                ( *mask_pyramid ) [level] = mask_temp ( Rect ( EDGE_THRESHOLD, EDGE_THRESHOLD, scaled_size.width, scaled_size.height ) );
            }

            // Resize the scaled image and mask.
            if ( level > 0 )
            {
                resize ( ( *image_pyramid ) [level - 1], ( *image_pyramid ) [level], scaled_size, 0, 0, INTER_LINEAR );
                copyMakeBorder ( ( *image_pyramid ) [level], image_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                 BORDER_REFLECT_101 + BORDER_ISOLATED );

                if ( !mask.empty() )
                {
                    resize ( ( *mask_pyramid ) [level - 1], ( *mask_pyramid ) [level], scaled_size, 0, 0, INTER_NEAREST );
                    copyMakeBorder ( ( *mask_pyramid ) [level], mask_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                     BORDER_CONSTANT + BORDER_ISOLATED );
                }
            }
            else
            {
                copyMakeBorder ( image, image_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                 BORDER_REFLECT_101 );

                if ( !mask.empty() )
                {
                    copyMakeBorder ( mask, mask_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                     BORDER_CONSTANT + BORDER_ISOLATED );
                }
            }
        }
    }


    void FeatureExtractor::ComputeKeypointOctTree ( const vector< Mat > &image_pyramid, const vector<Mat> &mask_pyramid, vector< vector< KeyPoint > > *keypoints )
    {
        keypoints->resize ( level_count_ );

        const double preset_cell_size = 30.0;
        Ptr<AgastFeatureDetector> agast = AgastFeatureDetector::create ( fast_agast_threshold_, true, fast_agast_type_ );
        Ptr<FastFeatureDetector> fast = FastFeatureDetector::create ( fast_agast_threshold_, true, fast_agast_type_ );

        for ( int level = 0; level < level_count_; level++ )
        {
            const int border_x_min = EDGE_THRESHOLD - 3;
            const int border_y_min = EDGE_THRESHOLD - 3;
            const int border_x_max = image_pyramid[level].cols - EDGE_THRESHOLD + 3;
            const int border_y_max = image_pyramid[level].rows - EDGE_THRESHOLD + 3;

            const double width = border_x_max - border_x_min;
            const double height = border_y_max - border_y_min;

            const int cell_cols = width / preset_cell_size;
            const int cell_rows = height / preset_cell_size;
            const int cell_width = ceil ( width / cell_cols );
            const int cell_height = ceil ( height / cell_rows );

            vector<KeyPoint> keypoints_temp;
            keypoints_temp.reserve ( feature_count_ * 10 );

            for ( int i = 0; i < cell_rows; i++ )
            {
                const double cell_y_min = border_y_min + i * cell_height;
                const double cell_y_max = min ( static_cast<double> ( border_y_max ), cell_y_min + cell_height + 6 );

                if ( cell_y_min >= border_y_max - 6 )
                {
                    continue;
                }

                for ( int j = 0; j < cell_cols; j++ )
                {
                    const double cell_x_min = border_x_min + j * cell_width;
                    const double cell_x_max = min ( static_cast<double> ( border_x_max ),  cell_x_min + cell_width + 6 );

                    if ( cell_x_min >= border_x_min - 6 )
                    {
                        continue;
                    }

                    vector<KeyPoint> cell_keypoints;

                    if ( use_agast_ )
                    {
                        agast->detect ( image_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ), cell_keypoints,
                                        mask_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ) );
                    }
                    else
                    {
                        fast->detect ( image_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ), cell_keypoints,
                                       mask_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ) );
                    }

                    if ( !cell_keypoints.empty() )
                    {
                        for ( vector<KeyPoint>::iterator keypoint_it = cell_keypoints.begin(); keypoint_it != cell_keypoints.end(); keypoint_it++ )
                        {
                            ( *keypoint_it ).pt.x += j * cell_width;
                            ( *keypoint_it ).pt.y += i * cell_height;
                            keypoints_temp.push_back ( *keypoint_it );
                        }
                    }
                }
            }

            vector<KeyPoint> &level_keypoints = ( *keypoints ) [level];
            level_keypoints.reserve ( feature_count_ );
            DistributeOctTree ( keypoints_temp, &level_keypoints, border_x_min, border_x_max, border_y_min, border_y_max, level );

            const int scaled_patch_size = PATCH_SIZE * scale_factors_[level];

            // Correct keypoints to right position.
            for ( size_t i = 0; i < level_keypoints.size(); i++ )
            {
                level_keypoints[i].pt.x += border_x_min;
                level_keypoints[i].pt.y += border_y_min;
                level_keypoints[i].octave = level;
                level_keypoints[i].size = scaled_patch_size;
            }
        }

        // Compute orientations
        for ( int level = 0; level < level_count_; level++ )
        {
            ComputeOrientation ( image_pyramid[level], & ( *keypoints ) [level] );
        }
    }

    void FeatureExtractor::ComputeDescriptors ( const Mat &level_image, const vector< KeyPoint > &level_keypoints,
            const vector< Vec2d > &undistorted_keypoints, const CameraModel &camera_model, Mat *level_descriptors, Mat *level_descriptor_masks )
    {

    }

    void FeatureExtractor::DistributeOctTree ( const vector< KeyPoint > &keypoints, vector< KeyPoint > *level_keypoints,
            const int border_x_min, const int border_x_max,
            const int border_y_min, const int border_y_max,
            const int level )
    {

    }

    void FeatureExtractor::ComputeOrientation ( const Mat &image, vector< KeyPoint > *level_keypoints )
    {

    }

}
