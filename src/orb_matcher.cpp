#include "orb_matcher.h"

#include <math.h>

namespace MCVORBSLAM
{
    const int ORBMatcher::HISTO_LENGTH = 30;

    ORBMatcher::ORBMatcher ( const double nn_ratio, const int descriptor_size, const bool check_orientation, const bool use_mask )
        : nn_ratio_ ( nn_ratio ), descriptor_size_ ( descriptor_size ), check_orientation_ ( check_orientation ), use_mask_ ( use_mask )
    {
        // Change thresholds if we are using masks, as the hamming distance distribution
        // for matching and non-matching points differs
        if ( use_mask_ )
        {
            threshold_high_ = floor ( 1.5 * descriptor_size_ );
            threshold_low_ = floor ( descriptor_size_ );
        }
        else
        {
            threshold_high_ = 3 * descriptor_size_;
            threshold_low_ = 2 * descriptor_size_;
        }
    }

    int ORBMatcher::SearchForInitialization ( const MultiFrame &frame1, const MultiFrame &frame2, vector< Vec2d > *keypoint_positions, vector< int > *matches_12, const int window_size )
    {
        int nmatches = 0;

        matches_12->resize ( frame1.GetKeyPoints().size(), -1 );

        vector<int> orientation_histogram[HISTO_LENGTH];

        for ( int i = 0; i < HISTO_LENGTH; ++i ) {
            orientation_histogram[i].reserve ( 500 );
	}
	
        const double orientation_unit = 1.0 / HISTO_LENGTH;

        vector<int> matched_distance ( frame2.GetKeyPoints().size(), INT_MAX );
        vector<int> matches_21 ( frame2.GetKeyPoints().size(), -1 );

        for ( size_t keypoint_index_1 = 0, iend1 = frame1.GetKeyPoints().size(); keypoint_index_1 < iend1; ++keypoint_index_1 )
        {
            cv::KeyPoint keypoint_1 = frame1.GetKeyPoints()[keypoint_index_1];
            int level_1 = keypoint_1.octave;

            int camera_index_1 = frame1.keypoint_to_cam.find ( keypoint_index_1 )->second;

            vector<size_t> vIndices2 =
                F2.GetFeaturesInArea ( camera_index_1, vbPrevMatched[i1] ( 0 ),
                                       vbPrevMatched[i1] ( 1 ),
                                       windowSize, level_1, level_1 );

            //cout << "vIndices2: " << vIndices2.size() << endl;
            //vector<size_t> vIndices2 =
            //	F2.GetFeaturesInArea(camIdx1, vbPrevMatched[i1](0), vbPrevMatched[i1](1),
            //	windowSize);
            if ( vIndices2.empty() )
                continue;

            int descIdx1 = F1.cont_idx_to_local_cam_idx.find ( keypoint_index_1 )->second;

            const uint64_t *d1 = F1.mDescriptors[camIdx1].ptr<uint64_t> ( descIdx1 );
            const uint64_t *d1_mask = 0;

            if ( havingMasks )
                d1_mask = F1.mDescriptorMasks[camIdx1].ptr<uint64_t> ( descIdx1 );

            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            for ( vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); ++vit )
            {
                size_t i2 = *vit;

                int camIdx2 = F2.keypoint_to_cam.find ( i2 )->second;
                int descIdx2 = F2.cont_idx_to_local_cam_idx.find ( i2 )->second;

                const uint64_t *d2 = F2.mDescriptors[camIdx2].ptr<uint64_t> ( descIdx2 );
                int dist = 0;

                if ( havingMasks )
                {
                    const uint64_t *d2_mask = F2.mDescriptorMasks[camIdx2].ptr<uint64_t> ( descIdx2 );
                    dist = DescriptorDistance64Masked ( d1, d2, d1_mask, d2_mask, mbFeatDim );
                }
                else
                    dist = DescriptorDistance64 ( d1, d2, mbFeatDim );

                if ( matched_distance[i2] <= dist )
                    continue;

                if ( dist < bestDist )
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestIdx2 = i2;
                }
                else if ( dist < bestDist2 )
                {
                    bestDist2 = dist;
                }
            }

            if ( bestDist <= TH_LOW_ )
            {
                if ( bestDist < ( double ) bestDist2 * mfNNratio )
                {
                    if ( matches_21[bestIdx2] >= 0 )
                    {
                        vnMatches12[vnMatches21[bestIdx2]] = - 1;
                        nmatches--;
                    }

                    vnMatches12[i1] = bestIdx2;
                    matches_21[bestIdx2] = keypoint_index_1;
                    matched_distance[bestIdx2] = bestDist;
                    nmatches++;

                    if ( mbCheckOrientation )
                    {
                        float rot = F1.mvKeys[i1].angle - F2.mvKeys[bestIdx2].angle;

                        if ( rot < 0.0 )
                            rot += 360.0f;

                        int bin = round ( rot / ( 360 * orientation_unit ) );

                        if ( bin == HISTO_LENGTH )
                            bin = 0;

                        //ROS_ASSERT(bin>=0 && bin<HISTO_LENGTH);
                        orientation_histogram[bin].push_back ( keypoint_index_1 );
                    }
                }
            }

        }

        if ( mbCheckOrientation )
        {
            int ind1 = -1;
            int ind2 = -1;
            int ind3 = -1;

            ComputeThreeMaxima ( orientation_histogram, HISTO_LENGTH, ind1, ind2, ind3 );

            for ( int i = 0; i < HISTO_LENGTH; i++ )
            {
                if ( i == ind1 || i == ind2 || i == ind3 )
                    continue;

                for ( size_t j = 0, jend = orientation_histogram[i].size(); j < jend; j++ )
                {
                    int idx1 = orientation_histogram[i][j];

                    if ( vnMatches12[idx1] >= 0 )
                    {
                        vnMatches12[idx1] = -1;
                        --nmatches;
                    }
                }
            }

        }

        //Update prev matched
        for ( size_t i1 = 0, iend1 = vnMatches12.size(); i1 < iend1; ++i1 )
            if ( vnMatches12[i1] >= 0 )
                vbPrevMatched[i1] =
                    cv::Vec2d ( F2.mvKeys[vnMatches12[i1]].pt.x, F2.mvKeys[vnMatches12[i1]].pt.y );

        HResClk::time_point end = HResClk::now();
        cout << "---matching time (" << T_in_ms ( begin, end ) << ")--- nr:" << nmatches << " " << endl;
        return nmatches;
    }

}
