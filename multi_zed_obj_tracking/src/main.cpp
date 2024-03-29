#include <iostream>
#include <cmath>
#include "cuda_utils.h"

#include "GLViewer.hpp"
#include "ClientPublisher.hpp"

#include "logging.h"
#include "utils.h"
#include <NvInfer.h>

using namespace nvinfer1;

std::map<int, sl::Transform> initCameraTransformMatrices(std::map<int, sl::Transform> &tfs)
{
    std::map<int, sl::Transform> T_cam_world;

    bool is_world_set = false;
    int first_index = 0;
    // Iterate over camera poses
    for (auto &tf : tfs)
    {
        if (!is_world_set)
        {
            T_cam_world[tf.first] = sl::Transform::identity();
            first_index = tf.first;
            is_world_set = true;
        }
        else
        {
            T_cam_world[tf.first] = tf.second * T_cam_world[first_index];
        }
    }
    return T_cam_world;
}

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
    
    sl::InitParameters cam_params;
    cam_params.camera_resolution = sl::RESOLUTION::HD720;
    cam_params.camera_fps = 60U;
    cam_params.depth_mode = sl::DEPTH_MODE::NEURAL;
    cam_params.coordinate_units = UNIT;
    cam_params.coordinate_system = COORDINATE_SYSTEM;
    cam_params.depth_minimum_distance = 0.2f;
    cam_params.depth_maximum_distance = 9.0f;

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
    for (auto conf : configurations)
    {
        // if the ZED camera should run locally, then start a thread to handle it
        if (conf.communication_parameters.getType() == sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS)
        {
            std::cout << "Try to open ZED " << conf.serial_number << ".." << std::flush;
            cam_params.input = conf.input_type;
            auto state = clients[id_].open(cam_params, argv[2]);
            if (!state)
            {
                std::cerr << "Could not open ZED: " << conf.input_type.getConfiguration() << ". Skipping..." << std::endl;
                continue;
            }
            id_++;
        }
    }

    // start the client
    for (auto &it : clients)
        it.start();

    // Now that the ZED camera are running, we need to initialize the fusion module
    sl::InitFusionParameters fusion_params;
    fusion_params.coordinate_units = UNIT;
    fusion_params.coordinate_system = COORDINATE_SYSTEM;
    fusion_params.verbose = true;

    // create and initialize fusion module
    std::unique_ptr<sl::Fusion> ptr_fusion = std::make_unique<sl::Fusion>();
    ptr_fusion->init(fusion_params);

    // subscribe to every cameras of the setup to internally gather their data
    std::vector<sl::CameraIdentifier> cameras;
    for (auto &it : configurations)
    {
        sl::CameraIdentifier uuid(it.serial_number);
        // to subscribe to a camera you must give its serial number, the way to communicate with it (shared memory or local network), and its world pose in the setup.
        auto state = ptr_fusion->subscribe(uuid, it.communication_parameters, it.pose);
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
              << "\t- 's': swicth on/off for independent object bounding box display\n"
              << "\t- 'f': swicth on/off for fused object bounding box display\n"
              << "\t- 'p': swicth on/off for live point cloud display\n"
              << "\t- 'c': swicth on/off point cloud display with flat color\n"
              << "\t- 'q': quit\n"
              << std::endl;

    sl::FusionMetrics metrics;
    std::map<int, sl::Transform> poses;
    std::map<sl::CameraIdentifier, sl::Mat> views;
    std::map<sl::CameraIdentifier, sl::Objects> objects;
    std::map<sl::CameraIdentifier, sl::Mat> pointClouds;

    std::vector<ZedRenderData> renders;
    sl::Resolution low_res(512, 360);

    // assuming the first camera is world frame,
    // create transfomation matrix for remaining camera to first camera
    // first camera is identity
    std::map<int, sl::Transform> T_cam_world;
    bool isPoseSet = false;
    bool isTransformInit = false;

    // Set up a callback function for each client
    for (auto &client : clients)
    {
        client.setCallback([&views, &objects](const uint64_t id, sl::Mat &updatedSl, sl::Objects &detectedObjects)
                           {
            // std::cout << "Camera callback at  " << id << std::endl;
            views[id] = updatedSl;
            objects[id] = detectedObjects; });
    }

    while (ptr_viewer->isAvailable())
    {
        renders.clear();

        // run the fusion process (which gather data from all camera, sync them and process them)
        if (ptr_fusion->process() == sl::FUSION_ERROR_CODE::SUCCESS)
        {
            // set the camera pose and transformation matrices
            for (auto &id : cameras)
            {
                if (!isPoseSet)
                {
                    // camera static, set once only
                    sl::Pose pose;
                    if (ptr_fusion->getPosition(pose, sl::REFERENCE_FRAME::WORLD, id, sl::POSITION_TYPE::RAW) == sl::POSITIONAL_TRACKING_STATE::OK)
                    {
                        ptr_viewer->setCameraPose(id.sn, pose.pose_data);
                        poses[id.sn] = pose.pose_data;
                    }
                    // std::cout << "Adding camera poses " << id.sn << ", pose size = " << poses.size() << ", camera size = " << cameras.size() << std::endl;
                    isPoseSet = poses.size() > cameras.size() - 1;
                }
            }

            // render stream
            if (!isTransformInit && isPoseSet)
            {
                // std::cout << "Computing relative frame transform" << std::endl;
                T_cam_world = initCameraTransformMatrices(poses);
                isTransformInit = true;
            }

            if (isTransformInit)
            {
                for (auto &id : cameras)
                {
                    auto state_pc = ptr_fusion->retrieveMeasure(pointClouds[id], id, sl::MEASURE::XYZBGRA, low_res);
                    if (state_pc == sl::FUSION_ERROR_CODE::SUCCESS)
                    {
                        auto render = ZedRenderData(id.sn, views[id], objects[id], pointClouds[id]);
                        renders.emplace_back(render);
                    }
                }
                ptr_viewer->updateFusion(renders, T_cam_world);
                // get metrics about the fusion process for monitoring purposes
                // fusion.getProcessMetrics(metrics);
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    ptr_viewer->exit();

    for (auto &it : clients)
        it.stop();

    ptr_fusion->close();

    return EXIT_SUCCESS;
}
