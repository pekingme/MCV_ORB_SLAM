#include "camera_model.h"
#include "utils.h"

namespace MCVORBSLAM
{
    CameraModel::CameraModel ( const string &camera_name, const int width, const int height, const double u0, const double v0,
                               vector< double > &affine, vector< double > &poly, vector< double > &inv_poly )
        : camera_name_ ( camera_name ), width_ ( width ), height_ ( height ), u0_ ( u0 ), v0_ ( v0 ),
          c_ ( affine[0] ), d_ ( affine[1] ), e_ ( affine[2] ), inv_affine_ ( c_ - d_ * e_ ),
          poly_ ( poly ), poly_degree_ ( poly.size() ), inv_poly_ ( inv_poly ), inv_poly_degree_ ( inv_poly_.size() ) {}

    void CameraModel::ImageToCamera ( const double u, const double v, double *x, double *y, double *z )
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
}
