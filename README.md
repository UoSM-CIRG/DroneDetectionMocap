# DroneDetectionMocap

### Setup

The **calibrated_zed360.json** file provides important information for setup:

```
Zed2i -------------> Laptop <-----          
                               | |
                               | |
Zed2i ---> Jetson Orin Nano ---- |
        (10.100.26.64:30002)     |
                                 |
Zed2i ---> Jetson Orin Nano ------
        (10.100.27.205:30004)
```

Utilize the provided .onnx model, but conversion to TensorRT format is required. Note that this conversion depends on hardware and TensorRT version. Refer to zed_tensorrt_yolov8_onnx for guidance.

Refer to setup_ptp.txt for camera stream synchronization instructions.

Run the following commands in each build folder:


```
./yolo_onnx_zed -s ../../model/exported/real_and_synthetic.onnx mixed.engine

./multi_cam_obj_tracking ../../calibrated_zed360.json ../../zed_tensorrt_yolov8_onnx/build/mixed.engine 
```

### Experiments

Occasional glitches observed, not suitable for real-time applications.

After exporting raw data, removing outlier data points, and smoothing overall plots, there is quite a decent estimation.

Check the full video

![How it look like](/images/test1.png)

### TODO List:

Tasks according to priority:

    1. Deploy as Stereolab Edge Node and explore scalability.
    2. Optimize entire pipeline for GPU processing to maximize GPU load.
    3. Enhance Yolo object bounding box to include orientation in addition to position (x, y, z).
    4. Sync unique IDs for objects from different cameras; consider using a simple nearest neighbor approach.
    5. Explore detection using point clouds (consider Pointnet++).
    6. Implement multi-class labeling.