#include "shift_video.h"

namespace MCVORBSLAM
{
    ShiftVideo::ShiftVideo ( const string &video_path, const string &camera_name, const double offset )
        : video_path_ ( video_path ), camera_name_ ( camera_name ), offset_ ( offset )
    {
        if ( !Utils::FileExists ( video_path ) )
        {
            cerr << "Video file doesn't exist" << endl;
            exit ( 0 );
        }
    }

    ShiftVideo::~ShiftVideo()
    {
        if ( capture_.isOpened() )
        {
            capture_.release();
        }
    }

    void ShiftVideo::ReadFrame ( const double global_time, Mat *frame )
    {
        // Load VideoCapture object if not available.
        if ( &capture_ == NULL || !capture_.isOpened() )
        {
            capture_ = VideoCapture ( video_path_ );
        }

        if ( !capture_.isOpened() )
        {
            cerr << "Cannot load video file: " << video_path_ << endl;
            exit ( 0 );
        }

        double local_time = global_time - offset_;

        // Set frame to empty if video has not started yet.
        if ( local_time < 0.0 )
        {
            frame->create ( 0, 0, frame->type() );
            return;
        }

        capture_.set ( CV_CAP_PROP_POS_MSEC, local_time * 1000.0 );
        capture_.read ( *frame );
    }

}
