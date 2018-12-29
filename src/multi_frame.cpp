#include "multi_frame.h"
#include "utils.h"

#include <chrono>

namespace MCVORBSLAM
{
    size_t MultiFrame::next_id_ = 0;

    MultiFrame::MultiFrame ( const vector< Mat > &frames, const double timestamp, const CameraSystem &camera_system,
                             const vector< FeatureExtractor * > &extractors, const bool initializing )
        : frames_ ( frames ), timestamp_ ( timestamp ), camera_system_ ( camera_system ), extractors_ ( extractors )
    {
        chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

        // Initilize parameters for feature extraction.
        int camera_count = camera_system_.GetCameraCount();

        descriptors_.resize ( camera_count );
        descriptor_masks_.resize ( camera_count );
        resolutions_.resize ( camera_count );
        grid_sizes_.resize ( camera_count );
        keypoint_grid_map_.resize ( camera_count );
        scale_factors_.resize ( camera_count );
        scale_levels_.resize ( camera_count );
        scale_abs_factors_.resize ( camera_count );
        scale_abs_factors_inv_sqr_.resize ( camera_count );

        vector<vector<KeyPoint>> keypoints_temp ( camera_count );
        vector<vector<Vec3d>> keyrays_temp ( camera_count );

        #pragma omp parallel for num_threads(camera_count)

        for ( int c = 0; c < camera_count; c++ )
        {
            // Initialize parameters respect to each camera.
            CameraModel camera_model = camera_system_.GetCameraModel ( c );
            resolutions_[c] = Size ( camera_model.GetWidth(), camera_model.GetHeight() );
            grid_sizes_[c] = Size2d ( static_cast<double> ( resolutions_[c].width ) / static_cast<double> ( FRAME_GRID_COLS ),
                                      static_cast<double> ( resolutions_[c].height ) / static_cast<double> ( FRAME_GRID_ROWS ) );

            // Extract features
            ( *extractors[c] ) ( frames[c], &keypoints_temp[c], camera_model, descriptors_[c], descriptor_masks_[c], initializing );

            // Calculate ray as observations
            double x = 0.0, y = 0.0, z = 0.0;
            keyrays_temp[c].resize ( keypoints_temp[c].size() );

            for ( size_t i = 0; i < keypoints_temp.size(); i++ )
            {
                camera_model.ImageToCamera ( static_cast<double> ( keypoints_temp[c][i].pt.x ),
                                             static_cast<double> ( keypoints_temp[c][i].pt.y ),
                                             &x, &y, &z );
                keyrays_temp[c][i] = Vec3d ( x, y, z );
            }

            // Scale pyramid info.
            scale_factors_[c] = extractors_[c]->GetScaleFactor();
            scale_levels_[c] = extractors_[c]->GetLevelCount();

            scale_abs_factors_[c].resize ( scale_levels_[c] );
            scale_abs_factors_[c][0] = 1.0;
            scale_abs_factors_inv_sqr_[c].resize ( scale_levels_[c] );
            scale_abs_factors_inv_sqr_[c][0] = 1.0;

            for ( int i = 1; i < scale_levels_[c]; i++ )
            {
                scale_abs_factors_[c][i] = scale_abs_factors_[c][i - 1] * scale_factors_[c];
                scale_abs_factors_inv_sqr_[c][i] = 1 / ( scale_abs_factors_[c][i] * scale_abs_factors_[c][i] );
            }

            // Prepare inner structure of grid map.
            keypoint_grid_map_[c] = vector<vector<vector<size_t>>> ( FRAME_GRID_COLS );

            for ( size_t i = 0; i < FRAME_GRID_COLS; i++ )
            {
                keypoint_grid_map_[c][i] = vector<vector<size_t>> ( FRAME_GRID_ROWS );
            }
        }

        // Now save each keypoint in order.
        for ( int c = 0; c < camera_count; c++ )
        {
            for ( size_t i = 0; i < keyrays_temp[c].size(); i++ )
            {
                size_t keypoint_index = keypoints_.size();
                keypoints_.push_back ( keypoints_temp[c][i] );
                keyrays_.push_back ( keyrays_temp[c][i] );
                keypoint_camera_map_[keypoint_index] = c;
                keypoint_imagewise_map_[keypoint_index] = i;
                KeyPoint &keypoint = keypoints_temp[c][i];
                // Fill grids if possible.
                int grid_x, grid_y;

                if ( FindGridCell ( c, keypoint, &grid_x, &grid_y ) )
                {
                    keypoint_grid_map_[c][grid_x][grid_y].push_back ( keypoint_index );
                }
            }
        }

        // Initlize outlier flags.
        keypoint_outliers_ = vector<bool> ( keypoints_.size(), false );
        id_ = next_id_++;
        
        // Copy some parameters from extractor to frame.
        use_mask_ = extractors_[0]->UseMask();
        descriptor_size_ = extractors_[0]->DescriptorSize();

        chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
        cout << "--- Feature Extraction (" << Utils::DurationInMilliseconds ( start, end ) << "ms) - FrameId: " << id_ << " ---" << endl;
    }

    bool MultiFrame::FindGridCell ( const int camera_index, const KeyPoint &keypoint, int *grid_x, int *grid_y )
    {
        *grid_x = cvRound ( ( keypoint.pt.x ) / grid_sizes_[camera_index].width );
        *grid_y = cvRound ( ( keypoint.pt.y ) / grid_sizes_[camera_index].height );

        // Becarse of undistortion, keypoint could be outsize of frame boundary.
        if ( *grid_x < 0 || *grid_x >= FRAME_GRID_COLS || *grid_y < 0 || *grid_y >= FRAME_GRID_ROWS )
        {
            return false;
        }

        return true;
    }


}
