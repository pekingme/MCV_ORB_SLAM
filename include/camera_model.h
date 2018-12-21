#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

# include "utils.h"

# include <vector>
# include <string>

# include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{
    class CameraModel
    {
    public:
        CameraModel ( const string &camera_name, const int width, const int height, const double u0, const double v0,
                      vector<double> &affine, vector<double> &poly, vector<double> &inv_poly );

        // Calculate coordinates respect to camera coordinate system based on point on image.
        // Input: (u, v) - coordinates on image with left-top corner as origin.
        // Output: (x, y, z) - 3D scene point, positive direction: right, down, forward.
        void ImageToCamera ( const double u, const double v, double *x, double *y, double *z ) const;

        // Reproject the input image point to the image based on its ray.
        void UndistortPoints(const double in_x, const double in_y, const double focal_length, double* out_x, double* out_y) const;
        
        int GetWidth()
        {
            return width_;
        }

        int GetHeight()
        {
            return height_;
        }

        vector<double> GetPoly() const
        {
            return poly_;
        }

        Mat GetMirrorMask ( int level ) const
        {
            return mirror_masks_[level];
        }

    private:

        // Camera name
        string camera_name_;
        // FOV width in pixel
        int width_;
        // FOV height in pixel
        int height_;
        // Camera center x
        double u0_;
        // Camera center y
        double v0_;
        // Affine matrix [c, d; e, 1]
        double c_;
        // Affine matrix [c, d; e, 1]
        double d_;
        // Affine matrix [c, d; e, 1]
        double e_;
        // A useful coefficient c - d * e
        double inv_affine_;
        // Poly parameters
        vector<double> poly_;
        // Poly parameters degree
        int poly_degree_;
        // Inverse poly parameters
        vector<double> inv_poly_;
        // Inverse poly parameters degree
        int inv_poly_degree_;


        // Mirror masks on pyramid levels.
        vector<Mat> mirror_masks_;

    };
}

#endif // CAMERAMODEL_H
