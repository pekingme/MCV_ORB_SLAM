#include "multi_video.h"

#include <algorithm>

namespace MCVORBSLAM
{
    MultiVideo::MultiVideo ( const string &video_list_path ) : global_time_ ( 0.0 ), outdated_ ( true )
    {
        // Check if video list file exists
        if ( !Utils::FileExists ( video_list_path ) )
        {
            cerr << "Synchronized video list file doesn't exist" << endl;
            exit ( 0 );
        }

        // Clear all videos.
        ReleaseVideos();

        // Read contents from video list file.
        FileStorage video_list_file ( video_list_path, FileStorage::READ );

        max_shift_ = static_cast<double> ( video_list_file["MaxShift"] );
        FileNode video_node = video_list_file["Videos"];
        FileNodeIterator video_node_it = video_node.begin();
        FileNodeIterator video_node_it_end = video_node.end();

        for ( ; video_node_it != video_node_it_end; video_node_it++ )
        {
            string file_path = static_cast<string> ( ( *video_node_it ) ["File"] );
            string camera_name = static_cast<string> ( ( *video_node_it ) ["CameraName"] );
            double offset = static_cast<double> ( ( *video_node_it ) ["Offset"] );

            ShiftVideo shift_video ( file_path, camera_name, offset );

            videos_.push_back ( shift_video );
        }

        video_list_file.release();
    }

    bool MultiVideo::HasNext()
    {
        int empty_count;

        do
        {
            // Buffer new frames if they are out-dated.
            if ( outdated_ )
            {
                BufferFrames();
            }

            // Count number of frames are not empty.
            empty_count = 0;

            for ( unsigned int i = 0; i < frames_.size(); i++ )
            {
                if ( frames_[i].empty() )
                {
                    empty_count ++;
                }
            }

            // Increament global time to next available position if there are empty frames
            if ( empty_count != 0)
            {
                global_time_ += 1.0 / target_fps_;
                outdated_ = true;
            }
            // If all frames are empty, break loop.
        }
        while ( outdated_ && empty_count != static_cast<int> ( frames_.size() ) );

        return empty_count == 0;
    }

    bool MultiVideo::Next ( vector< Mat > *frames, double *time_stamp, const bool grayscale )
    {
        if ( HasNext() )
        {
            *time_stamp = global_time_;
            frames->resize ( frames_.size() );

            for ( unsigned int i = 0; i < frames_.size(); i++ )
            {
                if ( !grayscale || frames_[i].empty() || frames_[i].channels() == 1 )
                {
                    ( *frames ) [i] = frames_[i];
                }
                else
                {
                    Mat gray_frame;
                    cvtColor ( frames_[i], gray_frame, CV_BGR2GRAY );
                    ( *frames ) [i] = gray_frame;
                }
            }

            // Increment global time if
            global_time_ += 1.0 / target_fps_;
            outdated_ = true;

            return true;
        }
        else
        {
            cout << endl << "Video reached the end." << endl;
            return false;
        }
    }

    void MultiVideo::BufferFrames()
    {
        frames_.resize ( videos_.size() );

        for ( unsigned int i = 0; i < videos_.size(); i++ )
        {
            videos_[i].ReadFrame ( global_time_, &frames_[i] );
        }

        outdated_ = false;
    }

    void MultiVideo::ReleaseVideos()
    {
        for ( unsigned int i = 0; i < videos_.size(); i++ )
        {
            videos_[i].~ShiftVideo();
        }
    }


}

