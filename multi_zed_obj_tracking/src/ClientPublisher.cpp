#include "ClientPublisher.hpp"

ClientPublisher::ClientPublisher() : running(false)
{
}

ClientPublisher::~ClientPublisher()
{
    zed.close();
}

bool ClientPublisher::open(sl::InitParameters &param, const std::string engine_name)
{
    // already running
    if (runner.joinable())
        return false;

    auto state = zed.open(param);
    if (state != sl::ERROR_CODE::SUCCESS)
    {
        std::cout << "Error: " << state << std::endl;
        return false;
    }
    // application assumes the camera is static:
    sl::PositionalTrackingParameters positional_tracking_parameters;
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
    detection_parameters.enable_segmentation = true;
    detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
    // detection_parameters.instance_module_id = zed.getCameraInformation().serial_number;
    state = zed.enableObjectDetection(detection_parameters);
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

    return true;
}

void ClientPublisher::start()
{
    if (zed.isOpened())
    {
        running = true;
        zed.startPublishing();
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
                // tmp.is_grounded = ((int)it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
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
            // zed.retrieveObjects(objects, objectTracker_parameters_rt, zed.getCameraInformation().serial_number);
            zed.retrieveObjects(objects, objectTracker_parameters_rt);
            // Notify callback with updated data
            if (odCallback)
            {
                // std::cout << "sending od data from camera = " << zed.getCameraInformation().serial_number << std::endl;
                // std::cout << "objects size = " << objects.object_list.size() << std::endl;
                odCallback(zed.getCameraInformation().serial_number, left_sl, objects);
            }
        }
    }
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
