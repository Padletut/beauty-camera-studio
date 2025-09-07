#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

namespace gpupixel {

struct CameraInfo {
    int id;
    std::string name;
    std::vector<cv::Size> supported_resolutions;
    cv::Size current_resolution;
    cv::Size default_resolution;
};

struct CameraSettings {
    float brightness = 0.0f;        // Camera brightness (-100 to 100)
    float contrast = 50.0f;         // Camera contrast (0 to 100)
    float saturation = 50.0f;       // Camera saturation (0 to 100)
    float gain = 50.0f;             // Camera gain (0 to 100)
    float sharpness = 50.0f;        // Camera sharpness (0 to 100)
    float zoom = 100.0f;            // Camera zoom (100 to 500)
    bool auto_focus = true;         // Camera auto focus
    float focus = 50.0f;            // Manual focus value (0 to 100)
    bool auto_gain = true;          // Camera auto gain (auto exposure)
    cv::Size resolution{1920, 1080}; // Default resolution
};

class CameraManager {
public:
    CameraManager();
    ~CameraManager();

    // Camera enumeration and selection
    void EnumerateAvailableCameras();
    void EnumerateResolutionsForCamera(int camera_id);
    bool SwitchCamera(int new_camera_id);
    
    // Camera settings
    void ApplyCameraSettings();
    void GetCurrentCameraSettings();
    void SetCameraSettings(const CameraSettings& settings);
    const CameraSettings& GetCameraSettings() const { return camera_settings_; }
    
    // Getters
    const std::vector<CameraInfo>& GetAvailableCameras() const { return available_cameras_; }
    int GetCurrentCameraId() const { return current_camera_id_; }
    bool HasCameraSettingsChanged() const { return camera_settings_changed_; }
    void ClearCameraSettingsChanged() { camera_settings_changed_ = false; }

private:
    std::vector<CameraInfo> available_cameras_;
    int current_camera_id_;
    CameraSettings camera_settings_;
    bool camera_settings_changed_;
    
    // Helper methods
    std::string GetCameraName(int camera_id);
    bool TestResolution(cv::VideoCapture& cap, int width, int height);
    void ApplyV4L2Settings();
};

} // namespace gpupixel
