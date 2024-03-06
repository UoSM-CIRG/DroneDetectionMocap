#ifndef O3D_UTILS_H_
#define O3D_UTILS_H_

#include <open3d/Open3D.h>
#include <sl/Camera.hpp>

// Define the inline function
inline open3d::geometry::PointCloud ConvertSLMatToO3DPointCloud(const sl::Mat& sl_pc) {
    int width = sl_pc.getResolution().width;
    int height = sl_pc.getResolution().height;

    // Create Open3D point cloud
    open3d::geometry::PointCloud pc_o3d;

    // Extract XYZ and color information from sl::Mat
    float* data_ptr = sl_pc.getPtr<float>(sl::MEM::CPU);
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            float x = data_ptr[(v * width + u) * 4 + 0];
            float y = data_ptr[(v * width + u) * 4 + 1];
            float z = data_ptr[(v * width + u) * 4 + 2];

            // Convert BGRA to RGB
            unsigned char b = static_cast<unsigned char>(data_ptr[(v * width + u) * 4 + 3]);
            unsigned char g = static_cast<unsigned char>(data_ptr[(v * width + u) * 4 + 2]);
            unsigned char r = static_cast<unsigned char>(data_ptr[(v * width + u) * 4 + 1]);

            // Add point to Open3D point cloud
            pc_o3d.points_.emplace_back(Eigen::Vector3d(x, y, z));
            pc_o3d.colors_.emplace_back(r / 255.0, g / 255.0, b / 255.0);
        }
    }

    return pc_o3d;
}

#endif  