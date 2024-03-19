# DroneDetectionMocap

./multi_cam_obj_tracking ../../zed-aruco/multi-camera/build/MultiCamConfig.json ../../zed_tensorrt_yolov8_onnx/build/yolov8s.engine 

./yolo_onnx_zed -s ../../model/exported/real_and_synthetic.onnx mixed.engine

./multi_cam_obj_tracking ../../calibrated_zed360.json ../../zed_tensorrt_yolov8_onnx/build/mixed.engine 