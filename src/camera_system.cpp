#include "camera_system.h"
#include "converter.h"
#include "utils.h"

#include <iostream>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

namespace MCVORBSLAM
{
    CameraSystem CameraSystem::LoadFromYaml ( const string &yaml_path )
    {
        // Check if yaml file exists
        if ( !Utils::FileExists ( yaml_path ) )
        {
            cerr << endl << "Camera system calibration file doesn't exist." << endl;
            exit ( 0 );
        }

        // Load intrinsic and extrinsic for each camera in calibration file.
        FileStorage file ( yaml_path,  FileStorage::READ );
        FileNode camera_node = file["Cameras"];
        FileNodeIterator camera_node_it = camera_node.begin();
        FileNodeIterator camera_node_it_end = camera_node.end();

	  
        vector<CameraModel> camera_models;
        vector<Matx44d> camera_relative_poses;

        for ( ; camera_node_it != camera_node_it_end; camera_node_it++ )
        {
            string camera_name = ( *camera_node_it ) ["Name"];
            int width = ( *camera_node_it ) ["Width"];
            int height = ( *camera_node_it ) ["Height"];
            double u0 = ( *camera_node_it ) ["U0"];
            double v0 = ( *camera_node_it ) ["V0"];
            vector<double> affine, poly, inv_poly, extrinsic;
            affine.push_back ( ( double ) ( *camera_node_it ) ["C"] );
            affine.push_back ( ( double ) ( *camera_node_it ) ["D"] );
            affine.push_back ( ( double ) ( *camera_node_it ) ["E"] );
            ( *camera_node_it ) ["Poly"] >> poly;
            ( *camera_node_it ) ["InversePoly"] >> inv_poly;
            ( *camera_node_it ) ["Extrinsic"] >> extrinsic;

            CameraModel camera_model ( camera_name, width, height, u0, v0, affine, poly, inv_poly );
            camera_models.push_back ( camera_model );

            Matx61d pose_vec ( extrinsic.data() );
            Matx44d pose = Converter::VecToHomogeneous ( pose_vec );
            camera_relative_poses.push_back ( pose );
        }
        
        file.release();

        // Create camera system object.
        CameraSystem camera_system ( camera_models, camera_relative_poses );
        
        return camera_system;
    }

    CameraSystem::CameraSystem ( const vector< CameraModel > &camera_models, const vector< Matx44d > &camera_relative_poses )
        : camera_count_ ( camera_models.size() ), camera_models_ ( camera_models ), camera_relative_poses_ ( camera_relative_poses )
    {
        camera_relative_poses_vec_.resize ( camera_count_ );

        for ( int c = 0; c < camera_count_; c++ )
        {
            camera_relative_poses_vec_[c] = Converter::HomogeneousToVec ( camera_relative_poses_[c] );
        }
    }

}
