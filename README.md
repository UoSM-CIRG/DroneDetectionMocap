# DroneDetectionMocap

### Setup

The calibrated_zed360.json gives good info on it.

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

The is provided .onnx model, need to convert to tensorrt format yourself. Hardware and tensorrt version dependant.

Look into zed_tensorrt_yolov8_onnx.

Check out the setup_ptp.txt to sync camera stream.

Go to each build folder and run following command.
```
./yolo_onnx_zed -s ../../model/exported/real_and_synthetic.onnx mixed.engine

./multi_cam_obj_tracking ../../calibrated_zed360.json ../../zed_tensorrt_yolov8_onnx/build/mixed.engine 
```

### Experiments

Occasional glitches, clearly not good enough for real time application. 

Exported the raw data, removing the outlier fused data points and smoothen the overall plots give quite a good estimation. 

Check the full video

![How it look like](/images/test1.png)

### TODO List:

Lots of stuff but just summarize a few according to priority.

1. Stereolab Edge Deployment (how to scale it)
2. Figure out how to run entire pipeline on GPU, GPU load is not maximized.
3. Figure out Yolo obb, only position x,y,z now, need the orientation also.
4. Synching unique id for objects from different camera, not all cameras assiging the same objects. (maybe simple nearest neighhbour?)
5. Perform detection with pointclouds instead (thinking of Pointnet++)
6. Multiple class label
 