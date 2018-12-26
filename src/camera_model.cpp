#include "camera_model.h"
#include "utils.h"

namespace MCVORBSLAM
{
    CameraModel::CameraModel ( const string &camera_name, const int width, const int height, const double u0, const double v0,
                               vector< double > &affine, vector< double > &poly, vector< double > &inv_poly )
        : camera_name_ ( camera_name ), width_ ( width ), height_ ( height ), u0_ ( u0 ), v0_ ( v0 ),
          c_ ( affine[0] ), d_ ( affine[1] ), e_ ( affine[2] ), inv_affine_ ( c_ - d_ * e_ ),
          poly_ ( poly ), poly_degree_ ( poly.size() ), inv_poly_ ( inv_poly ), inv_poly_degree_ ( inv_poly_.size() ) {}

    void CameraModel::ImageToCamera ( const double u, const double v, double *x, double *y, double *z ) const
    {
        const double u_t = u - u0_;
        const double v_t = v - v0_;
        // inverse affine matrix image to sensor plane conversion
        *x = ( u_t - d_ * v_t ) / inv_affine_;
        *y = ( -e_ * u_t + c_ * v_t ) / inv_affine_;
        const double rho = hypot ( *x, *y );
        // z is flipped because Scaramuzza's model, z points to the mirror,
        // which is opposite of camera forward.
        *z = -Utils::Horner ( poly_.data(), poly_degree_, rho );

        // normalize vectors spherically
        double norm = hypot ( rho, *z );
        *x /= norm;
        *y /= norm;
        *z /= norm;
    }

    void CameraModel::CameraToImage ( const double x, const double y, const double z, double *u, double *v ) const
    {
        double norm = hypot ( x, y );

        if ( norm == 0.0 )
        {
            norm = 1e-14;
        }

        // z is flipped, i.e. theta is flipped.
        const double theta = atan ( z / norm );
        const double rho = -Utils::Horner ( inv_poly_.data(), inv_poly_degree_, theta );

        const double uu = x / norm * rho;
        const double vv = y / norm * rho;

        // Affine matrix
        *u = uu * c_ + vv * d_ + u0_;
        *v = uu * e_ + vv + v0_;
    }

    void CameraModel::UndistortPoint ( const double in_x, const double in_y, const double focal_length, double *out_x, double *out_y ) const
    {
        double x, y, z;
        ImageToCamera ( in_x, in_y, &x, &y, &z );
        *out_x = x / z * focal_length;
        *out_y = y / z * focal_length;
    }

    void CameraModel::DistortPoint ( const double in_x, const double in_y, double *out_x, double *out_y ) const
    {
        CameraToImage ( in_x, in_y, -poly_[0], out_x, out_y );
    }

}
