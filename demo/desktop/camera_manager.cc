#include "camera_manager.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdlib>

namespace gpupixel {

CameraManager::CameraManager() 
    : current_camera_id_(0), camera_settings_changed_(false) {
}

CameraManager::~CameraManager() {
}

void CameraManager::EnumerateAvailableCameras() {
    std::cout << "Enumerating available cameras..." << std::endl;
    available_cameras_.clear();
    
    for (int i = 0; i < 10; ++i) {
        cv::VideoCapture cap(i);
        if (cap.isOpened()) {
            CameraInfo info;
            info.id = i;
            info.name = GetCameraName(i);
            
            // Get current/default resolution
            info.current_resolution.width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            info.current_resolution.height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            info.default_resolution = info.current_resolution;
            
            available_cameras_.push_back(info);
            cap.release();
            
            std::cout << "Found camera " << i << ": " << info.name << std::endl;
        }
    }
    
    std::cout << "Total cameras found: " << available_cameras_.size() << std::endl;
    
    // Enumerate resolutions for each camera
    for (auto& camera : available_cameras_) {
        EnumerateResolutionsForCamera(camera.id);
    }
}

void CameraManager::EnumerateResolutionsForCamera(int camera_id) {
    std::cout << "Testing resolutions for camera " << camera_id << "..." << std::endl;
    
    auto it = std::find_if(available_cameras_.begin(), available_cameras_.end(),
                          [camera_id](const CameraInfo& info) { return info.id == camera_id; });
    
    if (it == available_cameras_.end()) {
        std::cerr << "Camera " << camera_id << " not found" << std::endl;
        return;
    }
    
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << camera_id << std::endl;
        return;
    }
    
    std::cout << "  Current/Default: " << it->current_resolution.width << "x" << it->current_resolution.height << std::endl;
    
    // Test common resolutions
    std::vector<cv::Size> test_resolutions = {
        {160, 120}, {320, 240}, {352, 288}, {640, 360}, {640, 480},
        {704, 576}, {800, 448}, {800, 600}, {960, 720}, {1024, 576},
        {1152, 864}, {1280, 720}, {1440, 900}, {1600, 896}, {1680, 1050},
        {1920, 1080}, {2048, 1536}, {2304, 1536}
    };
    
    it->supported_resolutions.clear();
    
    for (const auto& res : test_resolutions) {
        if (TestResolution(cap, res.width, res.height)) {
            it->supported_resolutions.push_back(res);
            std::cout << "  Supported: " << res.width << "x" << res.height;
            
            // Check if this matches a test resolution
            for (const auto& test_res : test_resolutions) {
                if (res.width >= test_res.width && res.height >= test_res.height &&
                    (res.width != test_res.width || res.height != test_res.height)) {
                    std::cout << " (requested " << test_res.width << "x" << test_res.height << ")";
                    break;
                }
            }
            std::cout << std::endl;
        }
    }
    
    cap.release();
    std::cout << "Total supported resolutions: " << it->supported_resolutions.size() << std::endl;
}

bool CameraManager::SwitchCamera(int new_camera_id) {
    if (new_camera_id == current_camera_id_) {
        return true; // Already using this camera
    }
    
    auto it = std::find_if(available_cameras_.begin(), available_cameras_.end(),
                          [new_camera_id](const CameraInfo& info) { return info.id == new_camera_id; });
    
    if (it == available_cameras_.end()) {
        std::cerr << "Camera " << new_camera_id << " not available" << std::endl;
        return false;
    }
    
    current_camera_id_ = new_camera_id;
    std::cout << "Switched to camera " << new_camera_id << ": " << it->name << std::endl;
    return true;
}

void CameraManager::ApplyCameraSettings() {
    ApplyV4L2Settings();
}

void CameraManager::GetCurrentCameraSettings() {
    // This would query current v4l2 settings, but for now we keep the stored values
    std::cout << "Getting current camera settings..." << std::endl;
}

void CameraManager::SetCameraSettings(const CameraSettings& settings) {
    camera_settings_ = settings;
    camera_settings_changed_ = true;
}

std::string CameraManager::GetCameraName(int camera_id) {
    return "Camera " + std::to_string(camera_id);
}

bool CameraManager::TestResolution(cv::VideoCapture& cap, int width, int height) {
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    
    int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // Consider it supported if we got close to the requested resolution
    return (actual_width >= width * 0.9 && actual_height >= height * 0.9);
}

void CameraManager::ApplyV4L2Settings() {
    std::cout << "Applying camera settings using v4l2-ctl..." << std::endl;
    
    // Build v4l2-ctl commands
    std::ostringstream brightness_cmd, contrast_cmd, saturation_cmd;
    std::ostringstream gain_cmd, sharpness_cmd, zoom_cmd, focus_cmd;
    
    // Convert UI values to v4l2 ranges
    int v4l2_brightness = static_cast<int>((camera_settings_.brightness + 100) * 255 / 200);  // -100 to 100 -> 0 to 255
    int v4l2_contrast = static_cast<int>(camera_settings_.contrast * 255 / 100);              // 0 to 100 -> 0 to 255
    int v4l2_saturation = static_cast<int>(camera_settings_.saturation * 255 / 100);          // 0 to 100 -> 0 to 255
    int v4l2_sharpness = static_cast<int>(camera_settings_.sharpness * 255 / 100);            // 0 to 100 -> 0 to 255
    int v4l2_zoom = static_cast<int>(camera_settings_.zoom);                                   // 100 to 500 -> 100 to 500
    
    brightness_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=brightness=" << v4l2_brightness;
    contrast_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=contrast=" << v4l2_contrast;
    saturation_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=saturation=" << v4l2_saturation;
    sharpness_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=sharpness=" << v4l2_sharpness;
    zoom_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=zoom_absolute=" << v4l2_zoom;
    
    std::cout << "  Brightness: " << camera_settings_.brightness << " (v4l2: " << v4l2_brightness << ")" << std::endl;
    std::cout << "  Contrast: " << camera_settings_.contrast << " (v4l2: " << v4l2_contrast << ")" << std::endl;
    std::cout << "  Saturation: " << camera_settings_.saturation << " (v4l2: " << v4l2_saturation << ")" << std::endl;
    std::cout << "  Sharpness: " << camera_settings_.sharpness << " (v4l2: " << v4l2_sharpness << ")" << std::endl;
    std::cout << "  Zoom: " << camera_settings_.zoom << "% (v4l2: " << v4l2_zoom << ")" << std::endl;
    
    // Execute commands
    std::system(brightness_cmd.str().c_str());
    std::system(contrast_cmd.str().c_str());
    std::system(saturation_cmd.str().c_str());
    std::system(sharpness_cmd.str().c_str());
    std::system(zoom_cmd.str().c_str());
    
    // Handle auto/manual focus
    if (camera_settings_.auto_focus) {
        std::ostringstream auto_focus_cmd;
        auto_focus_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=focus_auto=1";
        std::system(auto_focus_cmd.str().c_str());
        std::cout << "  Auto Focus: ON" << std::endl;
    } else {
        std::ostringstream manual_focus_cmd;
        manual_focus_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=focus_auto=0";
        std::system(manual_focus_cmd.str().c_str());
        
        int v4l2_focus = static_cast<int>(camera_settings_.focus * 255 / 100);
        focus_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=focus_absolute=" << v4l2_focus;
        std::system(focus_cmd.str().c_str());
        std::cout << "  Manual Focus: " << camera_settings_.focus << " (v4l2: " << v4l2_focus << ")" << std::endl;
    }
    
    // Handle auto/manual gain (exposure)
    if (camera_settings_.auto_gain) {
        std::ostringstream auto_exposure_cmd;
        auto_exposure_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=exposure_auto=3";  // Aperture Priority Mode
        std::system(auto_exposure_cmd.str().c_str());
        std::cout << "  Auto Gain: ON (Aperture Priority Mode)" << std::endl;
    } else {
        std::ostringstream manual_exposure_cmd;
        manual_exposure_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=exposure_auto=1";  // Manual Mode
        std::system(manual_exposure_cmd.str().c_str());
        
        int v4l2_gain = static_cast<int>(camera_settings_.gain * 255 / 100);
        gain_cmd << "v4l2-ctl -d /dev/video" << current_camera_id_ << " --set-ctrl=gain=" << v4l2_gain;
        std::system(gain_cmd.str().c_str());
        std::cout << "  Manual Gain: " << camera_settings_.gain << " (v4l2: " << v4l2_gain << ")" << std::endl;
    }
    
    std::cout << "Camera settings applied successfully using v4l2-ctl" << std::endl;
}

} // namespace gpupixel
