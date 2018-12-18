#ifndef CAMERAMODEL_H
#define CAMERAMODEL_H

# include <vector>
# include <string>

using namespace std;

namespace MCVORBSLAM
{
    class CameraModel
    {
    public:
        CameraModel ( const string &camera_name, const int width, const int height, const double u0, const double v0,
                      vector<double> &affine, vector<double> &poly, vector<double> &inv_poly );
    };
}

#endif // CAMERAMODEL_H
