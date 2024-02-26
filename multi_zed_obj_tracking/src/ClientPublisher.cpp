#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher() : running(false)
{
}

ClientPublisher::~ClientPublisher()
{
    zed.close();
}

bool ClientPublisher::open(sl::InputType input, const std::string engine_name)
{
    // already running
    if (runner.joinable())
        return false;

    sl::InitParameters init_parameters;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.input = input;
    if (input.getType() == sl::InputType::INPUT_TYPE::SVO_FILE)
        init_parameters.svo_real_time_mode = true;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
    auto state = zed.open(init_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    // in most cases in body tracking setup, the cameras are static
    sl::PositionalTrackingParameters positional_tracking_parameters;
    // in most cases for body detection application the camera is static:
    positional_tracking_parameters.set_as_static = true;
    state = zed.enablePositionalTracking(positional_tracking_parameters);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    // Custom OD
    sl::ObjectDetectionParameters detection_parameters;
    detection_parameters.enable_tracking = true;
    detection_parameters.enable_segmentation = true; // designed to give person pixel mask
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }

    if (detector.init(engine_name))
    {
        std::cerr << "Detector init failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // // define the body tracking parameters, as the fusion can does the tracking and fitting you don't need to enable them here, unless you need it for your app
    // sl::BodyTrackingParameters body_tracking_parameters;
    // body_tracking_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_MEDIUM;
    // body_tracking_parameters.body_format = sl::BODY_FORMAT::BODY_18;
    // body_tracking_parameters.enable_body_fitting = false;
    // body_tracking_parameters.enable_tracking = false;
    // state = zed.enableBodyTracking(body_tracking_parameters);
    // if (state != sl::ERROR_CODE::SUCCESS)
    // {
    //     std::cout << "Error: " << state << std::endl;
    //     return false;
    // }

    return true;
}

void ClientPublisher::start()
{
    if (zed.isOpened())
    {
        running = true;
        // the camera should stream its data so the fusion can subscibe to it to gather the detected body and others metadata needed for the process.
        zed.startPublishing();
        // the thread can start to process the camera grab in background
        runner = std::thread(&ClientPublisher::work, this);
    }
}

void ClientPublisher::stop()
{
    running = false;
    if (runner.joinable())
        runner.join();
    zed.close();
}

void ClientPublisher::work()
{

    auto display_resolution = zed.getCameraInformation().camera_configuration.resolution;
    sl::Mat left_sl;
    cv::Mat left_cv;
    sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
    sl::Objects objects;

    while (running)
    {
        if (zed.grab() == sl::ERROR_CODE::SUCCESS)
        {
            // std::cout << "processing camera = " << zed.getCameraInformation().serial_number << std::endl;
            // Get image for inference
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);

            // Running inference
            auto detections = detector.run(left_sl, display_resolution.height, display_resolution.width, CONF_THRESH);

            // Get image for display
            // No need to reverse convert as cv::Mat share same memory address as sl::Mat
            // anything drawn on cv::Mat will be reflected to sl::Mat
            left_cv = slMat2cvMat(left_sl);

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it : detections)
            {
                sl::CustomBoxObjectData tmp;
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.prob;
                tmp.label = (int)it.label;
                tmp.bounding_box_2d = cvt(it.box);
                tmp.is_grounded = ((int)it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                // others are tracked in full 3D space
                objects_in.push_back(tmp);
            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);

            // Displaying 'raw' objects
            for (size_t j = 0; j < detections.size(); j++)
            {
                cv::Rect r = get_rect(detections[j].box);
                cv::rectangle(left_cv, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(left_cv, "ID: " + std::to_string((int)detections[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }

            // Retrieve the tracked objects, with 2D and 3D attributes
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
            // Notify callback with updated data
            if (odCallback)
            {
                // std::cout << "sending od data from camera = " << zed.getCameraInformation().serial_number << std::endl;
                odCallback(zed.getCameraInformation().serial_number, left_sl, objects);
            }
        }
    }
    // sl::Bodies bodies;
    // sl::BodyTrackingRuntimeParameters body_runtime_parameters;
    // body_runtime_parameters.detection_confidence_threshold = 40;

    // // in this sample we use a dummy thread to process the ZED data.
    // // you can replace it by your own application and use the ZED like you use to, retrieve its images, depth, sensors data and so on.
    // // as long as you call the grab function and the retrieveBodies (which runs the detection) the camera will be able to seamlessly transmit the data to the fusion module.
    // while (running) {
    //     if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
    //         /*
    //         Your App

    //         */

    //         // just be sure to run the bodies detection
    //         zed.retrieveBodies(bodies, body_runtime_parameters);
    //     }
    // }
}

void ClientPublisher::setStartSVOPosition(unsigned pos)
{
    zed.setSVOPosition(pos);
    zed.grab();
}

std::vector<sl::uint2> ClientPublisher::cvt(const BBox &bbox_in)
{
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}
