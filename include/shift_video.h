#ifndef SHIFTVIDEO_H
#define SHIFTVIDEO_H

# include "utils.h"

# include <string>

# include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

namespace MCVORBSLAM
{
    class ShiftVideo
    {
    public:
        ShiftVideo ( const string &video_path, const string &camera_name, const double offset );
        ~ShiftVideo();

        void ReadFrame ( const double global_time, Mat *frame );

        string GetCameraName() const
        {
            return camera_name_;
        }

    private:
        // OpenCV VideoCapture object
        VideoCapture capture_;
        // Video file location
        string video_path_;
        // Capturing camera name
        string camera_name_;
        // Relative time offset in seconds
        double offset_;
    };
}

#endif // SHIFTVIDEO_H
