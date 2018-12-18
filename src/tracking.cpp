#include "tracking.h"
#include "converter.h"

namespace MCVORBSLAM
{

    Tracking::Tracking ( Map *const map, FramePublisher *const frame_publisher, const vector<FeatureExtractor *> &extractors )
        : map_ ( map ), frame_publisher_ ( frame_publisher ), extractors_ ( extractors )
    {

    }


    void Tracking::ProcessMultiFrame ( const vector< Mat > &multi_frame, const double timestamp )
    {
        // Extract features based on current tracking state.
        if ( current_state_ == WORKING || current_state_ == LOST )
        {
            current_frame_ = new MultiFrame ( multi_frame, timestamp, extractors_ );
        }
        else
        {
            current_frame_ = new MultiFrame ( multi_frame, timestamp, extractors_ );
        }

        // Notify local mapping and loop closing the frame info.

        Track();
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

            last_frame_ = current_frame_;
        }

        // Update publishers.
        frame_publisher_->Update();
    }

    void Tracking::Initialize()
    {
        // TODO
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
