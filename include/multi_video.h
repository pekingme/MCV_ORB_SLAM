#ifndef MULTIVIDEO_H
#define MULTIVIDEO_H

# include "shift_video.h"

# include <vector>

# include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{

    class MultiVideo
    {
    public:
        // Constructor
        MultiVideo ( const string &video_list_path );

        // Return if there are more frame set so that all camera capture something
        bool HasNext();

        // Return the next available set of frames.
        bool Next ( vector<Mat> *frames, double *time_stamp, const bool grayscale );

        // Jump video to a given global time.
        void JumpToTime ( double time )
        {
            global_time_ = time;
        }

        // Set frame rate to target value.
        void SetTargetFPS ( int fps )
        {
            target_fps_ = fps;
        }

        string GetCameraName ( const int index ) const
        {
            return videos_[index].GetCameraName();
        }

    private:

        // Buffer frames based on current global time.
        void BufferFrames();
        // Release all videos currently opened.
        void ReleaseVideos();

        // Max amount of time shifts considered, in seconds;
        double max_shift_;
        // Bundled vides
        vector<ShiftVideo> videos_;
        // Target fps
        int target_fps_;

        // Current global time
        double global_time_;
        // Frames at current global time
        vector<Mat> frames_;
        // Indicate current frames and global time match or not
        bool outdated_;

    };
}

#endif // MULTIVIDEO_H
