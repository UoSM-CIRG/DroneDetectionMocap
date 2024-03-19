#ifndef __SENDER_RUNNER_HDR__
#define __SENDER_RUNNER_HDR__

#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>

#include <thread>
#include "yolo.hpp"

#define CONF_THRESH 0.7

class ClientPublisher
{

public:
    ClientPublisher();
    ~ClientPublisher();

    bool open(sl::InitParameters &, const std::string);
    void start();
    void stop();
    void setStartSVOPosition(unsigned pos);
    bool isRunning()
    {
        return running;
    }

    // object detection callback
    using ODCallback = std::function<void(const uint64_t id, sl::Mat&, sl::Objects&)>;
    void setCallback(const ODCallback& cb) {
        odCallback = cb;
    }

private:
    sl::Camera zed;
    void work();
    std::thread runner;
    bool running;
    int serial;
    // for custom object detector
    Yolo detector;
    std::vector<sl::uint2> cvt(const BBox &bbox_in);
    inline cv::Rect get_rect(BBox box)
    {
        return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
    }
    ODCallback odCallback;
};

#endif // ! __SENDER_RUNNER_HDR__
