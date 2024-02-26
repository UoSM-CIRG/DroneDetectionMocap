#include <iostream>
#include <cmath>
#include "cuda_utils.h"

#include "GLViewer.hpp"
#include "ClientPublisher.hpp"

#include "logging.h"
#include "utils.h"
#include <NvInfer.h>

using namespace nvinfer1;

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        // need configurationn file for camera placement relative to world frame and the yolov8 engine
        std::cout << "Usage:\n multi_cam_obj_tracking <calib.json> <yolov8s.engine>" << std::endl;
        return 1;
    }

    // Defines the Coordinate system and unit
    constexpr sl::COORDINATE_SYSTEM COORDINATE_SYSTEM = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    constexpr sl::UNIT UNIT = sl::UNIT::METER;

    // Read json file containing the configuration of your multicamera setup.
    auto configurations = sl::readFusionConfigurationFile(argv[1], COORDINATE_SYSTEM, UNIT);
    if (configurations.empty())
    {
        std::cout << "Empty configuration File." << std::endl;
        return EXIT_FAILURE;
    }

    // Check if the ZED camera should run within the same process or if they are running on the edge.
    std::vector<ClientPublisher> clients(configurations.size());
    int id_ = 0;
    //    std::map<int, std::string> svo_files;
    for (auto conf : configurations)
    {
        // if the ZED camera should run locally, then start a thread to handle it
        if (conf.communication_parameters.getType() == sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS)
        {
            std::cout << "Try to open ZED " << conf.serial_number << ".." << std::flush;
            auto state = clients[id_].open(conf.input_type, argv[2]);
            if (!state)
            {
                std::cerr << "Could not open ZED: " << conf.input_type.getConfiguration() << ". Skipping..." << std::endl;
                continue;
            }

            // if (conf.input_type.getType() == sl::InputType::INPUT_TYPE::SVO_FILE)
            //     svo_files.insert(std::make_pair(id_, conf.input_type.getConfiguration()));

            // std::cout << ". ready !" << std::endl;

            id_++;
        }
    }

    // // Synchronize SVO files in SVO mode
    // bool enable_svo_sync = (svo_files.size() > 1);
    // if (enable_svo_sync) {
    //     std::cout << "Starting SVO sync process..." << std::endl;
    //     std::map<int, int> cam_idx_to_svo_frame_idx = syncDATA(svo_files);

    //     for (auto &it : cam_idx_to_svo_frame_idx) {
    //         std::cout << "Setting camera " << it.first << " to frame " << it.second << std::endl;
    //         clients[it.first].setStartSVOPosition(it.second);
    //     }
    // }

    // start the client
    for (auto &it : clients)
        it.start();

    // Now that the ZED camera are running, we need to initialize the fusion module
    sl::InitFusionParameters init_params;
    init_params.coordinate_units = UNIT;
    init_params.coordinate_system = COORDINATE_SYSTEM;
    init_params.verbose = true;

    // create and initialize it
    sl::Fusion fusion;
    fusion.init(init_params);

    // subscribe to every cameras of the setup to internally gather their data
    std::vector<sl::CameraIdentifier> cameras;
    for (auto &it : configurations)
    {
        sl::CameraIdentifier uuid(it.serial_number);
        // to subscribe to a camera you must give its serial number, the way to communicate with it (shared memory or local network), and its world pose in the setup.
        auto state = fusion.subscribe(uuid, it.communication_parameters, it.pose);
        if (state != sl::FUSION_ERROR_CODE::SUCCESS)
            std::cout << "Unable to subscribe to " << std::to_string(uuid.sn) << " . " << state << std::endl;
        else
            cameras.push_back(uuid);
    }

    // check that at least one camera is connected
    if (cameras.empty())
    {
        std::cout << "no connections " << std::endl;
        return EXIT_FAILURE;
    }

    // Start camera threads after creating a shared_ptr to the GLViewer
    std::unique_ptr<GLViewer> ptr_viewer = std::make_unique<GLViewer>();
    ptr_viewer->init(argc, argv);
    std::cout << "Initialized OpenGL Viewer!";

    // showing cmds
    std::cout << "Viewer Shortcuts\n"
              << "\t- 'r': swicth on/off for raw skeleton display\n"
              << "\t- 'p': swicth on/off for live point cloud display\n"
              << "\t- 'c': swicth on/off point cloud display with flat color\n"
              << std::endl;

    sl::FusionMetrics metrics;
    std::map<sl::CameraIdentifier, sl::Mat> views;
    std::map<sl::CameraIdentifier, sl::Mat> pointClouds;
    sl::Resolution low_res(512, 360);

    bool new_data = false;
    sl::Timestamp ts_new_data = sl::Timestamp(0);

    // Set up a callback function for each client
    for (auto &client : clients) {
        client.setCallback([&views](const uint64_t id, sl::Mat& updatedSl, sl::Objects& detectedObjects) {
            // std::cout << "Updating camera map from = "<< id << std::endl;
            views[id] = updatedSl;
        });
    }
    // run the fusion as long as the viewer is available.
    while (ptr_viewer->isAvailable())
    {
        // std::cout << "Processing main fusion loop" << std::endl;
        // run the fusion process (which gather data from all camera, sync them and process them)
        if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS)
        {
            // for debug, you can retrieve the data send by each camera
            for (auto &id : cameras)
            {
                sl::Pose pose;
                if (fusion.getPosition(pose, sl::REFERENCE_FRAME::WORLD, id, sl::POSITION_TYPE::RAW) == sl::POSITIONAL_TRACKING_STATE::OK)
                    ptr_viewer->setCameraPose(id.sn, pose.pose_data);

                // auto state_view = fusion.retrieveImage(views[id], id, low_res);
                // if (state_view == sl::FUSION_ERROR_CODE::SUCCESS)
                //     ptr_viewer->updateCamera(id.sn, views[id]);
                auto state_pc = fusion.retrieveMeasure(pointClouds[id], id, sl::MEASURE::XYZBGRA, low_res);
                if (state_pc == sl::FUSION_ERROR_CODE::SUCCESS)
                    ptr_viewer->updatePC(id.sn, pointClouds[id]);
                
                auto it = views.find(id);
                if (it != views.end() && it->second.isInit())
                    ptr_viewer->updateCamera(id.sn, views[id]);
            }

            // get metrics about the fusion process for monitoring purposes
            fusion.getProcessMetrics(metrics);
        }
        // update the 3D view
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    ptr_viewer->exit();

    for (auto &it : clients)
        it.stop();

    fusion.close();

    return EXIT_SUCCESS;
}