#include "tracking.h"
#include "converter.h"

namespace MCVORBSLAM
{

    Tracking::Tracking ( System *const system, Map *const map, FramePublisher *const frame_publisher, const CameraSystem &camera_system, const vector<FeatureExtractor *> &extractors )
        : system_ ( system ), map_ ( map ), frame_publisher_ ( frame_publisher ), camera_system_ ( camera_system ), extractors_ ( extractors )
    {

    }


    void Tracking::ProcessMultiFrame ( const vector< Mat > &multi_frame, const double timestamp )
    {
        // Extract features based on current tracking state.
        if ( current_state_ == WORKING || current_state_ == LOST )
        {
            current_frame_ = new MultiFrame ( multi_frame, timestamp, camera_system_, extractors_, false );
        }
        else
        {
            current_frame_ = new MultiFrame ( multi_frame, timestamp, camera_system_, extractors_, true );
        }

        // Notify local mapping and loop closing the frame info.
        system_->NotifyFrameStatus ( current_frame_ );

        Track();

        last_frame_ = current_frame_;
    }

    void Tracking::Track()
    {
        last_processed_state_ = current_state_;

        // Depending on current tracking state to perform different tasks.
        if ( current_state_ == NO_IMAGES_YET )
        {
            current_state_ = NOT_INITIALIZED;
        }
        else if ( current_state_ == NOT_INITIALIZED || current_state_ == INITIALIZING )
        {
            // Construct an initial map to start.
            Initialize();
        }
        else
        {
            bool tracked = false;

            // Estimate an initial camera pose.
            if ( current_state_ == WORKING || !RelocalizationRequested() )
            {
                // Tracking is ok, and no relocalization is manually requested.
                if ( use_motion_model && map_->KeyFramesCount() >= 2
                        && current_frame_->id_ >= last_relocalized_frame_id_ + 2 )
                {
                    // Motion model is used and there are enough keyframes in map,
                    // and no recent frames are relocalized.
                    tracked = TrackWithMotionModel();
                }

                // Track from last frame, if track with motion model is unavailable or failed.
                tracked = tracked || TrackWithLastFrame();
            }
            else
            {
                // Relocalize, if tracking is lost or relocalization is manually requested.
                tracked = Relocalize();
            }

            // If initial camera pose is estimated, track local map
            tracked = tracked || TrackLocalMap();

            // Now current frame is localized, check if it contributes to the map.
            if ( tracked )
            {
                if ( NeedNewKeyFrame() )
                {
                    CreateNewKeyFrame();
                }

                // Remove all outlier map points from the frame.
                for ( unsigned int i = 0; i < current_frame_->outliers_.size(); i++ )
                {
                    current_frame_->map_points_[i] = NULL;
                }
            }

            if ( tracked )
            {
                current_state_ = WORKING;

                // TODO log current frame's position
            }
            else
            {
                current_state_ = LOST;
            }

            // Reset if tracking get lost soon after initialization, bad luck.
            if ( current_state_ == LOST )
            {
                if ( map_->KeyFramesCount() < 3 )
                {
                    Reset();
                    return;
                }
            }

            // Update current motion.
            if ( use_motion_model )
            {
                if ( tracked )
                {
                    velocity_ = current_frame_->GetPose() * Converter::InvertHomogeneous ( last_frame_->GetPose() );
                }
                else
                {
                    velocity_ = Matx44d::eye();
                }
            }
        }

        // Update publishers.
        frame_publisher_->Update();
    }

    void Tracking::Initialize()
    {
        vector<KeyPoint> current_keypoints = current_frame_->GetKeyPoints();

        if ( current_state_ == NOT_INITIALIZED )
        {
            // Only store current frame if it has enough features.
            if ( current_keypoints.size() > 100 )
            {
                initial_reference_frame_ = MultiFrame ( current_frame_ );
                keypoint_positions_.resize ( current_keypoints.size() );

                for ( int i = 0; i < current_keypoints.size(); i++ )
                {
                    keypoint_positions_[i] = Vec2d ( current_keypoints[i].pt.x, current_keypoints[i].pt.y );
                }

                initializer_ = new Initializer ( current_frame_ );

                current_state_ = INITIALIZING;
            }
        }
        else if ( current_state_ == INITIALIZING )
        {
            // Set back to not initialized if next frame doesn't have enough features.
            if ( current_keypoints.size() <= 100 )
            {
                fill ( keypoint_matches_.begin(), keypoint_matches_.end(), -1 );

                current_state_ = NOT_INITIALIZED;
            }
            else
            {
                // TODO Find correspondences.
                ORBMatcher matcher(0.9, current_frame_->DescripterSize(), false, current_frame_->UseMask());
                int matches = matcher.SearchForInitialization();

                // Discard if there are not enough correspondences.
                if ( matches < 100 )
                {
                    current_state_ = NOT_INITIALIZED;
                    return;
                }

                cv::Matx33d rotation; // Current camera rotation
                cv::Vec3d translation; // Current camera translation
                vector<bool> triangulated; // Triangulated Correspondences
                int leading_camera = 0;

                if ( initializer_-> Initialize() )
                {
                    for ( size_t i = 0, iend = keypoint_matches_.size(); i < iend; ++i )
                    {
                        if ( keypoint_matches_[i] >= 0 && !triangulated[i] )
                        {
                            keypoint_matches_[i] = -1;
                            --matches;
                        }
                    }

                    CreateInitialMap ( );
                }
            }
        }
    }

    bool Tracking::TrackWithLastFrame()
    {
        // TODO
        return false;
    }

    bool Tracking::TrackWithMotionModel()
    {
        // TODO
        return false;
    }

    bool Tracking::Relocalize()
    {
        // TODO
        return false;
    }

    bool Tracking::TrackLocalMap()
    {
        // TODO
        return false;
    }

    bool Tracking::NeedNewKeyFrame()
    {
        // TODO
        return false;
    }

    void Tracking::CreateNewKeyFrame()
    {
        // TODO
    }

    void Tracking::Reset()
    {
        // TODO
    }

    bool Tracking::RelocalizationRequested()
    {
        unique_lock<mutex> lock ( mutex_manual_relocalization_requested );
        return manual_relocalization_requested;
    }
}
