#include "system.h"
#include "utils.h"

#include <iomanip>

namespace MCVORBSLAM
{

    void System::Initialize ( const string &yaml_path, const bool init_tracking, const bool init_local_mapping,
                              const bool init_loop_closing, const bool init_viewer )
    {
        cout << endl << "Initializing SLAM system" << endl << endl;

        // Initialize map and keyframe database if either is NULL
        if ( map_ == NULL || keyframe_database_ == NULL )
        {
            map_ = new Map();
            keyframe_database_ = new KeyframeDatabase ( vocabulary_ );
            cout << "\tDefault Map and KeyframeDatabase are created" << endl;
        }

        
        // Create publishers used in viewer.
        map_publisher_ = new MapPublisher ( map_ );
        frame_publisher_ = new FramePublisher ( map_, yaml_path );
        
        // Create tracking
        tracking_ = new Tracking();
        
        // Create local mapping
        local_mapping_ = new LocalMapping();
        
        // Create loop closing
        loop_closing_ = new LoopClosing();
        
        // Create viewer
        viewer_ = new Viewer ( this, frame_publisher_, map_publisher_, tracking_, yaml_path );

        if ( init_local_mapping )
        {
            local_mapping_thread_ = new thread ( &LocalMapping::Run, local_mapping_ );
        }

        if ( init_loop_closing )
        {
            loop_closing_thread_ = new thread ( &LoopClosing::Run, loop_closing_ );
        }

        if ( init_viewer )
        {
            viewer_thread_ = new thread ( &Viewer::Run, viewer_ );
        }

        cout << "\tRunning local mapping thread: " << init_local_mapping << endl;
        cout << "\tRunning loop closing thread: " << init_loop_closing << endl;
        cout << "\tRunning viewer thread: " << init_viewer << endl;
    }

    void System::LoadExtractor ( const string &yaml_path )
    {
        // Check if setting file exists.
        if ( !Utils::FileExists ( yaml_path ) )
        {
            cerr << endl << "SLAM setting file doesn't exist." << endl;
            exit ( 0 );
        }

        // Load ORB extractor parameters from SLAM setting file.
        FileStorage file ( yaml_path, FileStorage::READ );

        int feature_count = static_cast<int> ( file["Extractor.FeatureCount"] );
        int level_count = static_cast<int> ( file["Extractor.LevelCount"] );
        float level_scale = static_cast<float> ( file["Extractor.LevelScale"] );
        bool learn_masks = static_cast<bool> ( ( int ) file["Extractor.LearnMasks"] );
        bool use_agast = static_cast<bool> ( ( int ) file["Extractor.UseAgast"] );
        int fast_agast_threshold = static_cast<int> ( file["Extractor.FastAgastThreshold"] );
        int fast_agast_type = static_cast<int> ( file["Extractor.FastAgastType"] );
        bool use_mdBRIEF = static_cast<bool> ( ( int ) file["Extractor.UsemdBRIEF"] );
        int descriptor_size = static_cast<int> ( file["Extractor.DescriptorSize"] );
        int score_type = static_cast<int> ( file["Extractor.ScoreType"] );

        assert ( descriptor_size == 16 || descriptor_size == 32 || descriptor_size == 64 );
        assert ( score_type == 0 || score_type == 1 );

        int camera_count = camera_system_->GetCameraCount();
        feature_extractors_.resize ( camera_count );

        for ( int c = 0; c < camera_count; c++ )
        {
            feature_extractors_[c] = new FeatureExtractor ( feature_count, level_count, level_scale, learn_masks,
                    use_agast, fast_agast_threshold, fast_agast_type,
                    use_mdBRIEF, descriptor_size, score_type );
        }

        cout << "Feature extractor parameters are loaded!" << endl << endl;

        cout << boolalpha;
        cout << "\tNumber of features: " << feature_count << " (" <<  2 * feature_count << " while initialization)" << endl;
        cout << "\tFeature levels: " << level_count << endl;
        cout << "\tLevel scale: " << level_scale << endl;
        cout << "\tLearn masks: " << learn_masks << endl;
        cout << "\tUse AGAST instead of FAST: " << use_agast << endl;
        cout << "\tFAST/AGAST threshold: " << fast_agast_threshold << endl;
        cout << "\tFAST/AGAST type (OpenCV): " << fast_agast_type << endl;
        cout << "\tUse mdBRIEF instead of ORB: " << use_mdBRIEF << endl;
        cout << "\tDescriptor size: " << descriptor_size << endl;
        cout << "\tFeature sorting score: " << ( score_type == 0 ? "Harris Score" : "FAST Score" ) << endl;
    }

    void System::LoadVocabulary ( const string &vocabulary_path )
    {
        // Check if vocabulary file exists.
        if ( !Utils::FileExists ( vocabulary_path ) )
        {
            cerr << endl << "ORB vocabulary file doesn't exist." << endl;
            exit ( 0 );
        }

        // Load ORB vocabulary
        cout << endl << "Loading ORB Vocabulary, it may take a few seconds ..." << endl;
        vocabulary_ = new ORBVocabulary();
        bool vocabulary_loaded = vocabulary_->loadFromTextFile ( vocabulary_path );

        if ( !vocabulary_loaded )
        {
            cerr << "Unable to load ORB vocabulary file, format issue?" << endl;
            exit ( 0 );
        }

        cout << "ORB vocabulary loaded!" << endl;
    }



}
