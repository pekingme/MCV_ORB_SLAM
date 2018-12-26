#include "feature_extractor.h"

#include <list>

namespace MCVORBSLAM
{
    const double RAD_IN_DEGREE = CV_PI / 180.0;
    const int EDGE_THRESHOLD = 19;
    const int PATCH_SIZE = 32;
    const int HALF_PATCH_SIZE = 16;

    FeatureExtractor::FeatureExtractor ( const int feature_count, const int level_count, const float level_scale, const bool learn_mask,
                                         const bool use_agast, const int fast_agast_threshold, const int fast_agast_type,
                                         const bool use_distorted_brief, const int descriptor_size, const int score_type )
        : feature_count_ ( feature_count ), level_count_ ( level_count ), level_scale_ ( level_scale ), learn_mask_ ( learn_mask ),
          use_agast_ ( use_agast ), fast_agast_threshold_ ( fast_agast_threshold ), fast_agast_type_ ( fast_agast_type ),
          use_distorted_brief_ ( use_distorted_brief ), descriptor_size_ ( descriptor_size ), score_type_ ( score_type )
    {
        // Calculate scale factors for all levels.
        scale_factors_.resize ( level_count );
        scale_factors_[0] = 1.0;

        double level_scale_inv_sum = 1.0;

        for ( int i = 1; i < level_count; i++ )
        {
            scale_factors_[i] = scale_factors_[i - 1] * level_scale;
            level_scale_inv_sum += 1.0 / scale_factors_[i];
        }

        // Distribute keypoint number for each level
        level_feature_count_.resize ( level_count );

        int feature_total = 0;

        for ( int i = 0; i < level_count - 1; i++ )
        {
            level_feature_count_[i] = cvRound ( feature_count / level_scale_inv_sum * ( 1.0 / scale_factors_[i] ) );
            feature_total += level_feature_count_[i];
        }

        level_feature_count_[level_count - 1] = max ( feature_count - feature_total, 0 );


        // Assign pattern
        const int sample_count = 2 * 8 * descriptor_size;
        const Point *pattern_start = ( const Point * ) LEARNED_PATTERN_ORB_64;
        copy ( pattern_start, pattern_start + sample_count, back_inserter ( pattern_ ) );

        // Calculate boundary of circular patch for orientation calculation.
        patch_bound_u_.resize ( HALF_PATCH_SIZE + 1 );


        int vmax = cvFloor ( HALF_PATCH_SIZE * sqrt ( 2.f ) / 2 + 1 );
        int vmin = cvCeil ( HALF_PATCH_SIZE * sqrt ( 2.f ) / 2 );
        const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;

        for ( int v = 0; v <= vmax; ++v )
        {
            patch_bound_u_[v] = cvRound ( sqrt ( hp2 - v * v ) );
        }

        // Make sure we are symmetric
        for ( int v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v, ++v0 )
        {
            while ( patch_bound_u_[v0] == patch_bound_u_[v0 + 1] )
            {
                ++v0;
            }

            // Very tricky way to do this...
            patch_bound_u_[v] = v0;
            ++v0;
        }
    }

    void FeatureExtractor::operator() ( InputArray frame, vector<KeyPoint> *keypoints, const CameraModel &camera_model,
                                        OutputArray descriptors, OutputArray descriptor_masks, const bool initializing )
    {
        if ( frame.empty() )
        {
            return;
        }

        assert ( frame.type() == CV_8UC1 );

        // Compute the scale pyramid of image.
        vector<Mat> image_pyramid, mask_pyramid;
        Mat frame_mat = frame.getMat();
        Mat mask_mat = camera_model.GetMirrorMask ( 0 );
        ComputePyramid ( frame_mat, mask_mat, &image_pyramid, &mask_pyramid );

        // Compute keypoints in all levels of pyramid.
        vector<vector<KeyPoint>> keypoints_temp;
        ComputeKeypointOctTree ( image_pyramid, mask_pyramid, &keypoints_temp );
        int keypoints_count = 0;

        for ( int i = 0; i < level_count_; i++ )
        {
            keypoints_count += keypoints_temp[i].size();
        }

        // Prepare for descriptor computation.
        Mat descriptors_temp, descriptor_masks_temp;

        if ( keypoints_count == 0 )
        {
            descriptors.release();
            descriptor_masks.release();
        }
        else
        {
            descriptors.create ( keypoints_count, descriptor_size_, CV_8U );
            descriptor_masks.create ( keypoints_count, descriptor_size_, CV_8U );
            descriptors_temp = descriptors.getMat();
            descriptor_masks_temp = descriptor_masks.getMat();
        }

        keypoints->clear();
        keypoints->reserve ( keypoints_count );

        // Compute descriptors
        int row_to_write = 0;
        const double focal_length = -camera_model.GetPoly() [0];

        for ( int level = 0; level < level_count_; level++ )
        {
            vector<KeyPoint> &level_keypoints = keypoints_temp[level];
            int level_keypoints_count = level_keypoints.size();

            if ( level_keypoints_count == 0 )
            {
                continue;
            }

            // Get scaled image and blur it.
            Mat &level_image = image_pyramid[level];
            boxFilter ( level_image, level_image, level_image.depth(), Size ( 5, 5 ), Point ( -1, -1 ), true, BORDER_REFLECT_101 );

            // Undistort keypoint coordinates if distorted BRIEF is used.
            vector<Vec2d> undistorted_keypoints ( level_keypoints_count );
            double scale = scale_factors_[level];

            if ( use_distorted_brief_ )
            {
                for ( int i = 0; i < level_keypoints_count; i++ )
                {
                    camera_model.UndistortPoint ( static_cast<double> ( level_keypoints[i].pt.x * scale ),
                                                  static_cast<double> ( level_keypoints[i].pt.y * scale ),
                                                  focal_length,
                                                  & ( undistorted_keypoints[i] ( 0 ) ),
                                                  & ( undistorted_keypoints[i] ( 1 ) ) );
                }
            }

            Mat level_descriptors = descriptors_temp.rowRange ( row_to_write, row_to_write + level_keypoints_count );
            Mat level_descriptor_masks = descriptor_masks_temp.rowRange ( row_to_write, row_to_write + level_keypoints_count );
            ComputeDescriptors ( level_image, level_keypoints, undistorted_keypoints, camera_model, &level_descriptors, &level_descriptor_masks );

            row_to_write +=  level_keypoints_count;

            // Scale keypoint coordinates
            if ( level > 0 )
            {
                for ( vector<KeyPoint>::iterator keypoint_it = level_keypoints.begin(); keypoint_it != level_keypoints.end(); keypoint_it++ )
                {
                    keypoint_it->pt *= scale;
                }
            }

            // Add keypoints to output.
            keypoints->insert ( keypoints->end(), level_keypoints.begin(), level_keypoints.end() );
        }
    }

    void FeatureExtractor::ComputePyramid ( const Mat &image, const Mat &mask, vector< Mat > *image_pyramid, vector<Mat> *mask_pyramid )
    {
        for ( int level = 0; level < level_count_; level++ )
        {
            // Determine scaled size.
            double scale = 1.0 / scale_factors_[level];
            Size scaled_size ( cvRound ( ( double ) image.cols * scale ), cvRound ( ( double ) image.rows * scale ) );
            Size scaled_size_with_border ( scaled_size.width + EDGE_THRESHOLD * 2, scaled_size.height + EDGE_THRESHOLD * 2 );
            // Initialize scaled image and scaled mask.
            Mat image_temp ( scaled_size_with_border, image.type() ), mask_temp;
            ( *image_pyramid ) [level] = image_temp ( Rect ( EDGE_THRESHOLD, EDGE_THRESHOLD, scaled_size.width, scaled_size.height ) );

            if ( !mask.empty() )
            {
                mask_temp = Mat ( scaled_size_with_border, mask.type() );
                ( *mask_pyramid ) [level] = mask_temp ( Rect ( EDGE_THRESHOLD, EDGE_THRESHOLD, scaled_size.width, scaled_size.height ) );
            }

            // Resize the scaled image and mask.
            if ( level > 0 )
            {
                resize ( ( *image_pyramid ) [level - 1], ( *image_pyramid ) [level], scaled_size, 0, 0, INTER_LINEAR );
                copyMakeBorder ( ( *image_pyramid ) [level], image_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                 BORDER_REFLECT_101 + BORDER_ISOLATED );

                if ( !mask.empty() )
                {
                    resize ( ( *mask_pyramid ) [level - 1], ( *mask_pyramid ) [level], scaled_size, 0, 0, INTER_NEAREST );
                    copyMakeBorder ( ( *mask_pyramid ) [level], mask_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                     BORDER_CONSTANT + BORDER_ISOLATED );
                }
            }
            else
            {
                copyMakeBorder ( image, image_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                 BORDER_REFLECT_101 );

                if ( !mask.empty() )
                {
                    copyMakeBorder ( mask, mask_temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                                     BORDER_CONSTANT + BORDER_ISOLATED );
                }
            }
        }
    }


    void FeatureExtractor::ComputeKeypointOctTree ( const vector< Mat > &image_pyramid, const vector<Mat> &mask_pyramid, vector< vector< KeyPoint > > *keypoints )
    {
        keypoints->resize ( level_count_ );

        const double preset_cell_size = 30.0;
        Ptr<AgastFeatureDetector> agast = AgastFeatureDetector::create ( fast_agast_threshold_, true, fast_agast_type_ );
        Ptr<FastFeatureDetector> fast = FastFeatureDetector::create ( fast_agast_threshold_, true, fast_agast_type_ );

        for ( int level = 0; level < level_count_; level++ )
        {
            const int border_x_min = EDGE_THRESHOLD - 3;
            const int border_y_min = EDGE_THRESHOLD - 3;
            const int border_x_max = image_pyramid[level].cols - EDGE_THRESHOLD + 3;
            const int border_y_max = image_pyramid[level].rows - EDGE_THRESHOLD + 3;

            const double width = border_x_max - border_x_min;
            const double height = border_y_max - border_y_min;

            const int cell_cols = width / preset_cell_size;
            const int cell_rows = height / preset_cell_size;
            const int cell_width = ceil ( width / cell_cols );
            const int cell_height = ceil ( height / cell_rows );

            vector<KeyPoint> keypoints_temp;
            keypoints_temp.reserve ( feature_count_ * 10 );

            for ( int i = 0; i < cell_rows; i++ )
            {
                const double cell_y_min = border_y_min + i * cell_height;
                const double cell_y_max = min ( static_cast<double> ( border_y_max ), cell_y_min + cell_height + 6 );

                if ( cell_y_min >= border_y_max - 6 )
                {
                    continue;
                }

                for ( int j = 0; j < cell_cols; j++ )
                {
                    const double cell_x_min = border_x_min + j * cell_width;
                    const double cell_x_max = min ( static_cast<double> ( border_x_max ),  cell_x_min + cell_width + 6 );

                    if ( cell_x_min >= border_x_min - 6 )
                    {
                        continue;
                    }

                    vector<KeyPoint> cell_keypoints;

                    if ( use_agast_ )
                    {
                        agast->detect ( image_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ), cell_keypoints,
                                        mask_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ) );
                    }
                    else
                    {
                        fast->detect ( image_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ), cell_keypoints,
                                       mask_pyramid[level].rowRange ( cell_y_min, cell_y_max ).colRange ( cell_x_min, cell_x_max ) );
                    }

                    if ( !cell_keypoints.empty() )
                    {
                        for ( vector<KeyPoint>::iterator keypoint_it = cell_keypoints.begin(); keypoint_it != cell_keypoints.end(); keypoint_it++ )
                        {
                            ( *keypoint_it ).pt.x += j * cell_width;
                            ( *keypoint_it ).pt.y += i * cell_height;
                            keypoints_temp.push_back ( *keypoint_it );
                        }
                    }
                }
            }

            vector<KeyPoint> &level_keypoints = ( *keypoints ) [level];
            level_keypoints.reserve ( feature_count_ );
            DistributeOctree ( keypoints_temp, &level_keypoints, border_x_min, border_x_max, border_y_min, border_y_max, level );

            const int scaled_patch_size = PATCH_SIZE * scale_factors_[level];

            // Correct keypoints to right position.
            for ( size_t i = 0; i < level_keypoints.size(); i++ )
            {
                level_keypoints[i].pt.x += border_x_min;
                level_keypoints[i].pt.y += border_y_min;
                level_keypoints[i].octave = level;
                level_keypoints[i].size = scaled_patch_size;
            }
        }

        // Compute orientations
        for ( int level = 0; level < level_count_; level++ )
        {
            ComputeOrientation ( image_pyramid[level], & ( *keypoints ) [level] );
        }
    }

    void FeatureExtractor::ComputeDescriptors ( const Mat &level_image, const vector< KeyPoint > &level_keypoints,
            const vector< Vec2d > &undistorted_keypoints, const CameraModel &camera_model, Mat *level_descriptors, Mat *level_descriptor_masks )
    {
        level_descriptors->create ( ( int ) level_keypoints.size(), descriptor_size_, CV_8UC1 );
        level_descriptor_masks->create ( ( int ) level_keypoints.size(), descriptor_size_, CV_8UC1 );

        if ( learn_mask_ )
        {
            #pragma omp parallel for num_threads(2)

            for ( size_t i = 0; i < level_keypoints.size(); ++i )
            {
                ComputeMaskedDistortedDescriptors ( level_image, level_keypoints[i], undistorted_keypoints[i],
                                                    camera_model, level_descriptors->ptr<uchar> ( i ), level_descriptor_masks->ptr<uchar> ( i ) );
            }
        }
        else if ( use_distorted_brief_ )
        {
            #pragma omp parallel for num_threads(2)

            for ( size_t i = 0; i < level_keypoints.size(); ++i )
            {
                ComputeDistortedDescriptors ( level_image, level_keypoints[i], undistorted_keypoints[i],
                                              camera_model, level_descriptors->ptr<uchar> ( i ) );
            }
        }
        else
        {
            #pragma omp parallel for num_threads(2)

            for ( size_t i = 0; i < level_keypoints.size(); ++i )
            {
                ComputeORBDescriptors ( level_image, level_keypoints[i], camera_model, level_descriptors->ptr<uchar> ( i ) );
            }
        }
    }

    void FeatureExtractor::DistributeOctree ( const vector< KeyPoint > &keypoints, vector< KeyPoint > *level_keypoints,
            const int border_x_min, const int border_x_max,
            const int border_y_min, const int border_y_max,
            const int level )
    {
        // Compute the number of initial nodes, determined by the ratio of the area.
        const int init_node_count = cvRound ( static_cast<double> ( border_x_max - border_x_min ) / ( border_y_max - border_y_min ) );
        const int node_width = static_cast<double> ( border_x_max - border_x_min ) / init_node_count;

        list<OctreeNode> nodes;

        // Create initial nodes.
        vector<OctreeNode *> init_nodes;
        init_nodes.resize ( init_node_count );

        for ( int i = 0; i < init_node_count; i++ )
        {
            OctreeNode node;
            node.left_ = node_width * static_cast<double> ( i );
            node.right_ = node_width * static_cast<double> ( i + 1 );
            node.up_ = 0;
            node.bottom_ = border_y_max - border_y_min;
            node.keypoints_.reserve ( keypoints.size() );

            nodes.push_back ( node );
            init_nodes[i] = &node;
        }

        // Assign keypoints to initial nodes.
        for ( size_t i = 0; i < keypoints.size(); i++ )
        {
            const KeyPoint &keypoint = keypoints[i];
            init_nodes[keypoint.pt.x / node_width]->keypoints_.push_back ( keypoint );
        }

        // Remove empty nodes and set flag if only has one keypoint.
        list<OctreeNode>::iterator node_it = nodes.begin();

        while ( node_it != nodes.end() )
        {
            if ( node_it->keypoints_.size() == 1 )
            {
                node_it->no_more_child_ = true;
                node_it++;
            }
            else if ( node_it->keypoints_.empty() )
            {
                node_it = nodes.erase ( node_it );
            }
            else
            {
                node_it++;
            }
        }

        // Loop through all nodes need to be processed.
        bool done = false;
        vector<pair<int, list<OctreeNode>::iterator>> keypoint_count_and_node;
        keypoint_count_and_node.reserve ( nodes.size() * 4 );

        while ( !done )
        {
            int nodes_count_to_expand = 0;

            keypoint_count_and_node.clear();

            node_it = nodes.begin();

            while ( node_it != nodes.end() )
            {
                if ( node_it->no_more_child_ )
                {
                    // If node only contains one keypoint, don't divide
                    node_it++;
                }
                else
                {
                    // Divide node if it has more than 1 keypoints
                    OctreeNode node_ul, node_ur, node_bl, node_br;
                    node_it->DivideNode ( &node_ul, &node_ur, &node_bl, &node_br );

                    // Add child nodes if they contain keypoints.
                    if ( node_ul.keypoints_.size() > 0 )
                    {
                        nodes.push_front ( node_ul );

                        if ( node_ul.keypoints_.size() > 1 )
                        {
                            nodes_count_to_expand ++;
                            keypoint_count_and_node.push_back ( make_pair ( node_ul.keypoints_.size(), nodes.begin() ) );
                        }
                    }

                    if ( node_ur.keypoints_.size() > 0 )
                    {
                        nodes.push_front ( node_ur );

                        if ( node_ur.keypoints_.size() > 1 )
                        {
                            nodes_count_to_expand ++;
                            keypoint_count_and_node.push_back ( make_pair ( node_ur.keypoints_.size(), nodes.begin() ) );
                        }
                    }

                    if ( node_bl.keypoints_.size() > 0 )
                    {
                        nodes.push_front ( node_bl );

                        if ( node_bl.keypoints_.size() > 1 )
                        {
                            nodes_count_to_expand ++;
                            keypoint_count_and_node.push_back ( make_pair ( node_bl.keypoints_.size(), nodes.begin() ) );
                        }
                    }

                    if ( node_br.keypoints_.size() > 0 )
                    {
                        nodes.push_front ( node_br );

                        if ( node_br.keypoints_.size() > 1 )
                        {
                            nodes_count_to_expand ++;
                            keypoint_count_and_node.push_back ( make_pair ( node_br.keypoints_.size(), nodes.begin() ) );
                        }
                    }

                    node_it = nodes.erase ( node_it );
                }
            }

            if ( ( int ) nodes.size() >= level_feature_count_[level] || nodes_count_to_expand == 0 )
            {
                // Finish if there are more nodes than required features or no nodes to divide
                done = true;
            }
            else if ( ( int ) nodes.size() + 3 * nodes_count_to_expand > level_feature_count_[level] )
            {
                // If the next level potentially has more nodes than required, divide the nodes
                // in the descending order of number of keypoints it contains.
                while ( !done )
                {
                    nodes_count_to_expand = 0;
                    vector<pair<int, list<OctreeNode>::iterator>> keypoint_count_and_node_temp = keypoint_count_and_node;
                    keypoint_count_and_node.clear();

                    sort ( keypoint_count_and_node_temp.begin(), keypoint_count_and_node.end(),
                           [] ( pair<int, list<OctreeNode>::iterator> &left, pair<int, list<OctreeNode>::iterator> &right )
                    {
                        return left.first < right.first;
                    } );

                    for ( int i = keypoint_count_and_node_temp.size() - 1; i >= 0; i-- )
                    {
                        OctreeNode node_ul, node_ur, node_bl, node_br;
                        keypoint_count_and_node_temp[i].second->DivideNode ( &node_ul, &node_ur, &node_bl, &node_br );

                        // Add child nodes if they contain keypoints.
                        if ( node_ul.keypoints_.size() > 0 )
                        {
                            nodes.push_front ( node_ul );

                            if ( node_ul.keypoints_.size() > 1 )
                            {
                                nodes_count_to_expand ++;
                                keypoint_count_and_node.push_back ( make_pair ( node_ul.keypoints_.size(), nodes.begin() ) );
                            }
                        }

                        if ( node_ur.keypoints_.size() > 0 )
                        {
                            nodes.push_front ( node_ur );

                            if ( node_ur.keypoints_.size() > 1 )
                            {
                                nodes_count_to_expand ++;
                                keypoint_count_and_node.push_back ( make_pair ( node_ur.keypoints_.size(), nodes.begin() ) );
                            }
                        }

                        if ( node_bl.keypoints_.size() > 0 )
                        {
                            nodes.push_front ( node_bl );

                            if ( node_bl.keypoints_.size() > 1 )
                            {
                                nodes_count_to_expand ++;
                                keypoint_count_and_node.push_back ( make_pair ( node_bl.keypoints_.size(), nodes.begin() ) );
                            }
                        }

                        if ( node_br.keypoints_.size() > 0 )
                        {
                            nodes.push_front ( node_br );

                            if ( node_br.keypoints_.size() > 1 )
                            {
                                nodes_count_to_expand ++;
                                keypoint_count_and_node.push_back ( make_pair ( node_br.keypoints_.size(), nodes.begin() ) );
                            }
                        }

                        nodes.erase ( keypoint_count_and_node_temp[i].second );

                        if ( ( int ) nodes.size() >= level_feature_count_[level] )
                        {
                            break;
                        }
                    }

                    if ( ( int ) nodes.size() >= level_feature_count_[level] || nodes_count_to_expand == 0 )
                    {
                        done = true;
                    }
                }
            }
        }

        // Retain the best keypoint in each node.
        level_keypoints->reserve ( feature_count_ );

        for ( list<OctreeNode>::iterator node_it = nodes.begin(); node_it != nodes.end(); node_it++ )
        {
            vector<KeyPoint> &node_keypoints = node_it->keypoints_;
            KeyPoint *best_keypoint = &node_keypoints[0];
            float max_response = best_keypoint->response;

            for ( size_t i = 1; i < node_keypoints.size(); i++ )
            {
                if ( node_keypoints[i].response > max_response )
                {
                    best_keypoint = &node_keypoints[i];
                    max_response = node_keypoints[i].response;
                }
            }
        }
    }

    void FeatureExtractor::ComputeOrientation ( const Mat &image, vector< KeyPoint > *level_keypoints )
    {
        for ( vector<KeyPoint>::iterator keypoint = level_keypoints->begin();
              keypoint != level_keypoints->end(); keypoint++ )
        {
            int m_01 = 0, m_10 = 0;

            const uchar *center = &image.at<uchar> ( cvRound ( keypoint->pt.y ), cvRound ( keypoint->pt.x ) );

            // Treat the center line differently, v=0
            for ( int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u )
                m_10 += u * center[u];

            // Go line by line in the circuI853lar patch
            int step = ( int ) image.step1();

            for ( int v = 1; v <= HALF_PATCH_SIZE; ++v )
            {
                // Proceed over the two lines
                int v_sum = 0;
                int d = patch_bound_u_[v];

                for ( int u = -d; u <= d; ++u )
                {
                    int val_plus = center[u + v * step], val_minus = center[u - v * step];
                    v_sum += ( val_plus - val_minus );
                    m_10 += u * ( val_plus + val_minus );
                }

                m_01 += v * v_sum;
            }

            keypoint->angle = fastAtan2 ( ( float ) m_01, ( float ) m_10 );
        }
    }

    void FeatureExtractor::ComputeMaskedDistortedDescriptors ( const Mat &image, const KeyPoint &keypoint, const Point2d &undistorted_keypoint,
            const CameraModel &camera_model, uchar *descriptor, uchar *descriptor_mask )
    {
        vector<Point> rotated_distorted_pattern ( pattern_.size() );

        RotateAndDistortPattern ( &rotated_distorted_pattern, keypoint.angle, undistorted_keypoint, camera_model );

        vector<vector<Point>> mask_pattern ( 2, vector<Point> ( pattern_.size() ) );

        RotateAndDistortPattern ( &mask_pattern[0], keypoint.angle + 20.0, undistorted_keypoint, camera_model );
        RotateAndDistortPattern ( &mask_pattern[1], keypoint.angle - 20.0, undistorted_keypoint, camera_model );

        const Point *pattern = &rotated_distorted_pattern[0];
        const Point *mask_pattern_1 = &mask_pattern[0][0];
        const Point *mask_pattern_2 = &mask_pattern[1][0];
        const uchar *center = 0;

        int ix = 0, iy = 0;

        int row = cvRound ( keypoint.pt.y );
        int col = cvRound ( keypoint.pt.x );

#define GET_VALUE(idx) \
    ( ix = pattern[idx].x, \
      iy = pattern[idx].y, \
      center = image.ptr<uchar>(row+iy),\
      center[col+ix] )
#define GET_VALUE_MASK1(idx) \
    ( ix = mask_pattern_1[idx].x, \
      iy = mask_pattern_2[idx].y, \
      center = image.ptr<uchar>(row+iy),\
      center[col+ix] )
#define GET_VALUE_MASK2(idx) \
    ( ix = mask_pattern_2[idx].x, \
      iy = mask_pattern_2[idx].y, \
      center = image.ptr<uchar>(row+iy),\
      center[col+ix] )

        for ( int i = 0; i < descriptor_size_; ++i, pattern += 16, mask_pattern_1 += 16, mask_pattern_2 += 16 )
        {
            int temp_val;
            int t0, t1, val = 0, maskVal = 0;
            int mask1_1, mask1_2, mask2_1, mask2_2, stable_val = 0;

            for ( int j = 0; j < 8; j++ )
            {
                stable_val = 0;
                t0 = GET_VALUE ( j * 2 );
                t1 = GET_VALUE ( j * 2 + 1 );
                temp_val = t0 < t1;
                val |= temp_val << j;
                mask1_1 = GET_VALUE_MASK1 ( j * 2 );
                mask1_2 = GET_VALUE_MASK1 ( j * 2 + 1 );           // mask1
                mask2_1 = GET_VALUE_MASK2 ( j * 2 );
                mask2_2 = GET_VALUE_MASK2 ( j * 2 + 1 );           // mask2
                stable_val += ( mask1_1 < mask1_2 ) ^ temp_val;
                stable_val += ( mask2_1 < mask2_2 ) ^ temp_val;
                maskVal |= ( stable_val == 0 ) << j;
            }

            descriptor[i] = ( uchar ) val;
            descriptor_mask[i] = ( uchar ) maskVal;
        }

#undef GET_VALUE
#undef GET_VALUE_MASK1
#undef GET_VALUE_MASK2
    }

    void FeatureExtractor::ComputeDistortedDescriptors ( const Mat &image, const KeyPoint &keypoint, const Point2d &undistorted_keypoint,
            const CameraModel &camera_model, uchar *descriptor )
    {
        vector<Point> rotated_distorted_pattern ( pattern_.size() );

        RotateAndDistortPattern ( &rotated_distorted_pattern, keypoint.angle, undistorted_keypoint, camera_model );

        const Point *pattern = &rotated_distorted_pattern[0];
        const uchar *center = 0;

        int ix = 0, iy = 0;

        int row = cvRound ( keypoint.pt.y );
        int col = cvRound ( keypoint.pt.x );

#define GET_VALUE(idx) \
    ( ix = pattern[idx].x, \
      iy = pattern[idx].y, \
      center = image.ptr<uchar>(row+iy),\
      center[col+ix] )

        for ( int i = 0; i < descriptor_size_; ++i, pattern += 16 )
        {
            int t0, t1, val = 0;

            for ( int j = 0; j < 8; j++ )
            {
                t0 = GET_VALUE ( 2 * j );
                t1 = GET_VALUE ( 2 * j + 1 );
                val |= ( t0 < t1 ) << j;
            }

            descriptor[i] = ( uchar ) val;
        }

#undef GET_VALUE
    }

    void FeatureExtractor::RotateAndDistortPattern ( vector< Point > *rotated_distorted_pattern, const double angle_in_degree,
            const Point2d &undistort_keypoint, const CameraModel &camera_model )
    {
        double angle_in_rad = angle_in_degree * RAD_IN_DEGREE;
        double cos_value = cos ( angle_in_rad );
        double sin_value = sin ( angle_in_rad );

        vector<double> x_dist ( pattern_.size() );
        vector<double> y_dist ( pattern_.size() );
        double x_center = 0.0;
        double y_center = 0.0;

        for ( size_t p = 0; p < pattern_.size(); ++p )
        {
            // Coordinates of rotated samples at undistorted keypoint.
            double x_sample = pattern_[p].x * cos_value - pattern_[p].y * sin_value + undistort_keypoint.x;
            double y_sample = pattern_[p].x * sin_value + pattern_[p].y * cos_value + undistort_keypoint.y;
            // Distort rotated sample positions
            camera_model.DistortPoint ( x_sample, y_sample, &x_dist[p], &y_dist[p] );
            // Accumulate x and y for finding the center.
            x_center += x_dist[p];
            y_center += y_dist[p];
        }

        // Get center of distorted pattern.
        x_center /= static_cast<double> ( pattern_.size() );
        y_center /= static_cast<double> ( pattern_.size() );

        // Substract the center from each sample in distorted pattern to get it relative.
        for ( size_t p = 0; p < pattern_.size(); ++p )
        {
            ( *rotated_distorted_pattern ) [p].x = cvRound ( x_dist[p] - x_center );
            ( *rotated_distorted_pattern ) [p].y = cvRound ( y_dist[p] - y_center );
        }
    }

    void FeatureExtractor::ComputeORBDescriptors ( const Mat &image, const KeyPoint &keypoint,
            const CameraModel &camera_model, uchar *descriptor )
    {
        vector<Point> rotated_pattern ( pattern_.size() );

        RotatePattern ( &rotated_pattern, keypoint.angle );

        const Point *pattern = &rotated_pattern[0];
        const uchar *center = 0;

        int ix = 0, iy = 0;

        int row = cvRound ( keypoint.pt.y );
        int col = cvRound ( keypoint.pt.x );

#define GET_VALUE(idx) \
    ( ix = pattern[idx].x, \
      iy = pattern[idx].y, \
      center = image.ptr<uchar>(row+iy),\
      center[col+ix] )

        for ( int i = 0; i < descriptor_size_; ++i, pattern += 16 )
        {
            int t0, t1, val = 0;

            for ( int j = 0; j < 8; j++ )
            {
                t0 = GET_VALUE ( 2 * j );
                t1 = GET_VALUE ( 2 * j + 1 );
                val |= ( t0 < t1 ) << j;
            }

            descriptor[i] = ( uchar ) val;
        }

#undef GET_VALUE
    }

    void FeatureExtractor::RotatePattern ( vector< Point > *rotated_pattern, const double angle_in_degree )
    {
        double angle_in_rad = angle_in_degree * RAD_IN_DEGREE;
        double cos_value = cos ( angle_in_rad );
        double sin_value = sin ( angle_in_rad );

        for ( size_t p = 0; p < pattern_.size(); ++p )
        {
            // rotate pattern point
            ( *rotated_pattern )[p].x = cvRound ( pattern_[p].x * cos_value - pattern_[p].y * sin_value );
            ( *rotated_pattern )[p].y = cvRound ( pattern_[p].x * sin_value + pattern_[p].y * cos_value );
        }
    }

    void OctreeNode::DivideNode ( OctreeNode *node_ul, OctreeNode *node_ur, OctreeNode *node_bl, OctreeNode *node_br )
    {
        const int mid_x = left_ + ceil ( static_cast<double> ( right_ - left_ ) / 2.0 );
        const int mid_y = up_ + ceil ( static_cast<double> ( bottom_ - up_ ) / 2.0 );

        // Define boundary of upper left node.
        node_ul->left_ = left_;
        node_ul->up_ = up_;
        node_ul->right_ = mid_x;
        node_ul->bottom_ = mid_y;

        // Define boundary of upper right node.
        node_ur->up_ = up_;
        node_ur->right_ = right_;
        node_ur->left_ = mid_x;
        node_ur->bottom_ = mid_y;

        // Define boundary of bottom left node.
        node_bl->bottom_ = bottom_;
        node_bl->left_ = left_;
        node_bl->right_ = mid_x;
        node_bl->up_ = mid_y;

        // Define boundary of bottom right node.
        node_br->bottom_ = bottom_;
        node_br->right_ = right_;
        node_br->left_ = mid_x;
        node_br->up_ = mid_y;

        // Associate keypoints to children nodes.
        for ( size_t i = 0; i < keypoints_.size(); i++ )
        {
            KeyPoint &keypoint = keypoints_[i];

            if ( keypoint.pt.x < mid_x )
            {
                if ( keypoint.pt.y < mid_y )
                {
                    node_ul->keypoints_.push_back ( keypoint );
                }
                else
                {
                    node_bl->keypoints_.push_back ( keypoint );
                }
            }
            else
            {
                if ( keypoint.pt.y < mid_y )
                {
                    node_ur->keypoints_.push_back ( keypoint );
                }
                else
                {
                    node_br->keypoints_.push_back ( keypoint );
                }
            }
        }

        // Set child flag
        node_ul->no_more_child_ = node_ul->keypoints_.size() <= 1;
        node_ur->no_more_child_ = node_ur->keypoints_.size() <= 1;
        node_bl->no_more_child_ = node_bl->keypoints_.size() <= 1;
        node_br->no_more_child_ = node_br->keypoints_.size() <= 1;
    }

}


