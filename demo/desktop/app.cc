// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on
#include <iostream>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "ghc/filesystem.hpp"

namespace fs {
using namespace ghc::filesystem;
using ifstream = ghc::filesystem::ifstream;
using ofstream = ghc::filesystem::ofstream;
using fstream = ghc::filesystem::fstream;
}  // namespace fs

#ifdef _WIN32
#include <Shlwapi.h>
#include <delayimp.h>
#include <windows.h>
#pragma comment(lib, "Shlwapi.lib")
#elif defined(__linux__)
#include <limits.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <stdlib.h>
#endif

#include "gpupixel/gpupixel.h"
#include "gpupixel/source/source_camera.h"
#include "face_detector/opencv_face_detector.h"
#include "gpupixel/filter/lipstick_filter.h"
#include "imgui.h"
#include <opencv2/opencv.hpp>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <unistd.h>
#include <cstring>
#include <vector>

using namespace gpupixel;

// Filters
std::shared_ptr<BeautyFaceFilter> beauty_filter_;
std::shared_ptr<FaceReshapeFilter> reshape_filter_;
std::shared_ptr<LipstickFilter> lipstick_filter_;   // For lipstick effect using face landmarks
std::shared_ptr<SaturationFilter> saturation_filter_; // For blusher-like effect
std::shared_ptr<SourceCamera> source_camera_;
std::shared_ptr<SinkRawData> sink_raw_data_;
#ifdef GPUPIXEL_ENABLE_FACE_DETECTOR
std::shared_ptr<FaceDetector> face_detector_;
#endif

// OpenCV Face Detector
std::shared_ptr<OpenCVFaceDetector> opencv_face_detector_;

// Filter parameters
float beauty_strength_ = 0.0f;
float whitening_strength_ = 0.0f;
float face_slim_strength_ = 0.0f;
float eye_enlarge_strength_ = 0.0f;
float color_tint_strength_ = 0.0f;      // For lipstick-like effect
float warmth_strength_ = 0.0f;          // For blusher-like effect

// Debug options
bool show_face_detection_ = false;      // Show face detection rectangles

// Beauty Profile System
struct BeautyProfile {
  std::string name;
  // Beauty filter settings
  float beauty_strength;
  float whitening_strength;
  float face_slim_strength;
  float eye_enlarge_strength;
  float color_tint_strength;
  float warmth_strength;
  
  // Camera settings
  float camera_brightness;
  float camera_contrast;
  float camera_saturation;
  float camera_gain;
  float camera_sharpness;
  float camera_zoom;
  bool camera_auto_focus;
  bool camera_auto_gain;
  
  // Resolution settings
  int resolution_width;
  int resolution_height;
  
  BeautyProfile() : name("Default"), beauty_strength(0.0f), whitening_strength(0.0f),
                    face_slim_strength(0.0f), eye_enlarge_strength(0.0f),
                    color_tint_strength(0.0f), warmth_strength(0.0f),
                    camera_brightness(0.0f), camera_contrast(50.0f), camera_saturation(50.0f),
                    camera_gain(0.0f), camera_sharpness(50.0f), camera_zoom(100.0f),
                    camera_auto_focus(true), camera_auto_gain(true),
                    resolution_width(640), resolution_height(480) {}
                    
  BeautyProfile(const std::string& n, float beauty, float whitening, float slim,
                float eye, float tint, float warmth, float brightness, float contrast,
                float saturation, float gain, float sharpness, float zoom,
                bool auto_focus, bool auto_gain, int width, int height)
    : name(n), beauty_strength(beauty), whitening_strength(whitening),
      face_slim_strength(slim), eye_enlarge_strength(eye),
      color_tint_strength(tint), warmth_strength(warmth),
      camera_brightness(brightness), camera_contrast(contrast), camera_saturation(saturation),
      camera_gain(gain), camera_sharpness(sharpness), camera_zoom(zoom),
      camera_auto_focus(auto_focus), camera_auto_gain(auto_gain),
      resolution_width(width), resolution_height(height) {}
};

std::map<std::string, BeautyProfile> beauty_profiles_;
std::string default_profile_name_ = "Default";

// Virtual Camera System
bool virtual_camera_enabled_ = false;
int virtual_camera_fd_ = -1;
const char* virtual_camera_device_ = "/dev/video10";
cv::VideoWriter virtual_camera_writer_;
cv::Mat virtual_camera_frame_;
std::string current_profile_name_ = "";
char profile_name_buffer_[256] = "New Profile";
bool show_profile_save_popup_ = false;
bool show_profile_manager_ = false;

// Camera settings
float camera_brightness_ = 0.0f;        // Camera brightness (-100 to 100)
float camera_contrast_ = 50.0f;         // Camera contrast (0 to 100)
float camera_saturation_ = 50.0f;       // Camera saturation (0 to 100)
float camera_gain_ = 50.0f;             // Camera gain (0 to 100)
float camera_sharpness_ = 50.0f;        // Camera sharpness (0 to 100)
float camera_zoom_ = 100.0f;            // Camera zoom (100 to 500)
bool camera_auto_focus_ = true;         // Camera auto focus
bool camera_auto_gain_ = true;          // Camera auto gain (auto exposure)
bool camera_settings_changed_ = false;  // Flag for camera settings changes

// Webcam device selection
std::vector<std::string> available_cameras_;
int current_camera_id_ = 0;
int selected_camera_id_ = 0;
bool camera_changed_ = false;

// Resolution selection
struct Resolution {
  int width;
  int height;
  std::string name;
  
  Resolution(int w, int h) : width(w), height(h) {
    name = std::to_string(w) + "x" + std::to_string(h);
  }
};

std::vector<Resolution> available_resolutions_;
int current_resolution_index_ = 0;
int selected_resolution_index_ = 0;
bool resolution_changed_ = false;

// Frame update timing
bool need_initial_processing = true;

// GLFW window handle
GLFWwindow* main_window_ = nullptr;

// Forward declarations
void EnumerateResolutionsForCamera(int camera_id);
void EnumerateAvailableCameras();
void RecreateCamera(int camera_id, int width, int height);
bool SwitchCamera(int new_camera_id);
void ApplyCameraSettings();
void GetCurrentCameraSettings();

// Beauty Profile System Functions
void SaveBeautyProfile(const std::string& name);
void LoadBeautyProfile(const std::string& name);
void ApplyCurrentProfile();
void SaveProfilesToFile();
void LoadProfilesFromFile();
void SetDefaultProfile(const std::string& name);

// Virtual Camera Functions
bool InitVirtualCamera();
void CloseVirtualCamera();
void WriteFrameToVirtualCamera();
void ToggleVirtualCamera();
std::string GetProfilesDirectory();

// Get executable path
std::string GetExecutablePath() {
  std::string path;
#ifdef _WIN32
  // Windows 平台实现
  std::cout << "GLFW initialized" << std::endl;
  char buffer[MAX_PATH];
  GetModuleFileNameA(NULL, buffer, MAX_PATH);
  PathRemoveFileSpecA(buffer);
  std::cout << "GLFW platform hint set" << std::endl;
  path = buffer;
#elif defined(__APPLE__)
  // macOS 平台实现
  char buffer[PATH_MAX];
  uint32_t size = sizeof(buffer);
  if (_NSGetExecutablePath(buffer, &size) == 0) {
    char realPath[PATH_MAX];
    if (realpath(buffer, realPath)) {
      path = realPath;
      // 移除文件名部分，只保留目录
      size_t pos = path.find_last_of("/\\");
      if (pos != std::string::npos) {
        path = path.substr(0, pos);
      }
    }
  }
  std::cout << "GLFW window hints set" << std::endl;
#elif defined(__linux__)
  // Linux 平台实现
  std::cout << "glfwCreateWindow called" << std::endl;
  char buffer[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
  if (count != -1) {
    buffer[count] = '\0';
    path = buffer;
  std::cout << "main_window_ pointer: " << main_window_ << std::endl;
    // 移除文件名部分，只保留目录
    size_t pos = path.find_last_of("/\\");
  std::cout << "Calling glfwMakeContextCurrent" << std::endl;
    if (pos != std::string::npos) {
  std::cout << "glfwMakeContextCurrent finished" << std::endl;
      path = path.substr(0, pos);
    }
  }
#endif
  return path;
}

// Enumerate supported resolutions for a specific camera
void EnumerateResolutionsForCamera(int camera_id) {
  available_resolutions_.clear();
  
  // Comprehensive list of video resolutions to test (including Logitech C920 specific ones)
  std::vector<Resolution> test_resolutions = {
    {160, 120},   // QQVGA
    {320, 240},   // QVGA
    {352, 288},   // CIF
    {640, 360},   // nHD
    {640, 480},   // VGA
    {704, 576},   // 4CIF
    {720, 480},   // NTSC
    {720, 576},   // PAL
    {800, 448},   // Logitech C920 specific
    {800, 600},   // SVGA
    {960, 720},   // 960p
    {1024, 576},  // WSVGA
    {1024, 768},  // XGA
    {1152, 864},  // XGA+
    {1280, 720},  // HD 720p
    {1280, 800},  // WXGA
    {1280, 960},  // SXGA
    {1280, 1024}, // SXGA
    {1366, 768},  // WXGA
    {1440, 900},  // WXGA+
    {1600, 900},  // HD+
    {1600, 1200}, // UXGA
    {1680, 1050}, // WSXGA+
    {1920, 1080}, // Full HD 1080p
    {1920, 1200}, // WUXGA
    {2048, 1152}, // QWXGA
    {2048, 1536}, // QXGA
    {2560, 1440}, // QHD
    {2560, 1600}, // WQXGA
    {3840, 2160}, // 4K UHD
    {4096, 2160}  // DCI 4K
  };
  
  cv::VideoCapture test_cap(camera_id, cv::CAP_V4L2);  // Force V4L2 backend
  if (!test_cap.isOpened()) {
    std::cerr << "Failed to open camera " << camera_id << " for resolution testing" << std::endl;
    // Add a default resolution if camera can't be opened
    available_resolutions_.push_back(Resolution(640, 480));
    return;
  }
  
  std::cout << "Testing resolutions for camera " << camera_id << "..." << std::endl;
  
  // Try to get the camera's current/default resolution first
  double current_width = test_cap.get(cv::CAP_PROP_FRAME_WIDTH);
  double current_height = test_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  if (current_width > 0 && current_height > 0) {
    available_resolutions_.push_back(Resolution((int)current_width, (int)current_height));
    std::cout << "  Current/Default: " << (int)current_width << "x" << (int)current_height << std::endl;
  }
  
  for (const auto& res : test_resolutions) {
    // Set the resolution
    test_cap.set(cv::CAP_PROP_FRAME_WIDTH, res.width);
    test_cap.set(cv::CAP_PROP_FRAME_HEIGHT, res.height);
    
    // Small delay to allow camera to adjust
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Get the actual resolution (may be different from requested)
    double actual_width = test_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double actual_height = test_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    // Check if the resolution is supported (allow some tolerance for minor differences)
    if (actual_width > 0 && actual_height > 0) {
      // Check if we already have this resolution (with tolerance for small differences)
      bool already_added = false;
      for (const auto& existing : available_resolutions_) {
        if (abs(existing.width - (int)actual_width) <= 2 && 
            abs(existing.height - (int)actual_height) <= 2) {
          already_added = true;
          break;
        }
      }
      
      if (!already_added) {
        available_resolutions_.push_back(Resolution((int)actual_width, (int)actual_height));
        std::cout << "  Supported: " << (int)actual_width << "x" << (int)actual_height;
        if ((int)actual_width != res.width || (int)actual_height != res.height) {
          std::cout << " (requested " << res.width << "x" << res.height << ")";
        }
        std::cout << std::endl;
      }
    }
  }
  
  test_cap.release();
  
  // If no resolutions were found, add a default one
  if (available_resolutions_.empty()) {
    available_resolutions_.push_back(Resolution(640, 480));
    std::cout << "  No resolutions detected, using default 640x480" << std::endl;
  }
  
  // Sort resolutions by total pixels
  std::sort(available_resolutions_.begin(), available_resolutions_.end(),
            [](const Resolution& a, const Resolution& b) {
              return (a.width * a.height) < (b.width * b.height);
            });
  
  std::cout << "Total supported resolutions: " << available_resolutions_.size() << std::endl;
}

// Enumerate available camera devices
void EnumerateAvailableCameras() {
  available_cameras_.clear();
  
  // If we're refreshing and have an active camera, we need to temporarily stop it
  // to allow enumeration of the current camera device
  bool was_camera_running = false;
  int previous_camera_id = current_camera_id_;
  
  if (source_camera_) {
    std::cout << "Temporarily stopping camera " << current_camera_id_ << " for enumeration..." << std::endl;
    source_camera_->Stop();
    // Give the camera a moment to fully release
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    was_camera_running = true;
  }
  
  // Test camera devices 0-9 (most systems won't have more than 10 cameras)
  for (int i = 0; i < 10; i++) {
    cv::VideoCapture test_cap(i, cv::CAP_V4L2);  // Force V4L2 backend
    if (test_cap.isOpened()) {
      std::string camera_name = "Camera " + std::to_string(i);
      
      // Note: Resolution information is now shown in the dedicated Resolution Selection section
      available_cameras_.push_back(camera_name);
      test_cap.release();
      std::cout << "Found camera " << i << ": " << camera_name << std::endl;
    } else {
      test_cap.release();
    }
  }
  
    // Restart the camera if it was previously running
    if (was_camera_running) {
      std::cout << "Recreating camera " << previous_camera_id << "..." << std::endl;
      
      // Reset the old camera object
      if (source_camera_) {
        source_camera_.reset();
      }
      
      // Use current resolution if available, otherwise default
      int width = 640, height = 480;
      if (!available_resolutions_.empty() && current_resolution_index_ < available_resolutions_.size()) {
        width = available_resolutions_[current_resolution_index_].width;
        height = available_resolutions_[current_resolution_index_].height;
      }
      
      // Create a new camera instance
      source_camera_ = SourceCamera::Create(previous_camera_id, width, height);
      if (source_camera_) {
        // Apply camera settings after creation
        ApplyCameraSettings();
        
        // Reconnect the pipeline
        source_camera_->AddSink(beauty_filter_)
            ->AddSink(reshape_filter_)   // Re-enabled - testing shader compatibility
            // ->AddSink(lipstick_filter_)  // Disabled - Color Tint(Lipstick Effect) causing black output
            ->AddSink(saturation_filter_)
            ->AddSink(sink_raw_data_);
        
        // Start the camera
        source_camera_->Start();
        std::cout << "Camera " << previous_camera_id << " restarted successfully with " << width << "x" << height << std::endl;
      } else {
        std::cerr << "Failed to recreate camera " << previous_camera_id << std::endl;
      }
    }  if (available_cameras_.empty()) {
    available_cameras_.push_back("No cameras found");
    std::cout << "No cameras detected" << std::endl;
  } else {
    std::cout << "Total cameras found: " << available_cameras_.size() << std::endl;
    // Enumerate resolutions for the current camera
    EnumerateResolutionsForCamera(current_camera_id_);
  }
}

// Recreate camera with new resolution
void RecreateCamera(int camera_id, int width, int height) {
  std::cout << "Processing resolution change to " << width << "x" << height << std::endl;
  
  // Stop and destroy the current camera
  if (source_camera_) {
    source_camera_->Stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    source_camera_.reset();
  }
  
  // Create new camera with specified resolution
  source_camera_ = SourceCamera::Create(camera_id, width, height);
  if (source_camera_) {
    // Apply camera settings after creation
    ApplyCameraSettings();
    
    // Reconnect the pipeline
    source_camera_->AddSink(beauty_filter_)
        ->AddSink(reshape_filter_)   // Re-enabled - testing shader compatibility
        // ->AddSink(lipstick_filter_)  // Disabled - Color Tint(Lipstick Effect) causing black output
        ->AddSink(saturation_filter_)
        ->AddSink(sink_raw_data_);
    
    // Start the camera
    source_camera_->Start();
    current_camera_id_ = camera_id;
    std::cout << "Resolution change successful" << std::endl;
  } else {
    std::cerr << "Failed to create camera " << camera_id << " with resolution " << width << "x" << height << std::endl;
    std::cerr << "This resolution may not be supported by your camera. Falling back to default resolution." << std::endl;
    
    // Fallback to a known working resolution
    int fallback_width = 640;
    int fallback_height = 480;
    
    // Try to use the first available resolution as fallback
    if (!available_resolutions_.empty()) {
      fallback_width = available_resolutions_[0].width;
      fallback_height = available_resolutions_[0].height;
    }
    
    std::cout << "Attempting fallback to " << fallback_width << "x" << fallback_height << std::endl;
    source_camera_ = SourceCamera::Create(camera_id, fallback_width, fallback_height);
    if (source_camera_) {
      // Apply camera settings after creation
      ApplyCameraSettings();
      
      // Reconnect the pipeline
      source_camera_->AddSink(beauty_filter_)
          ->AddSink(reshape_filter_)   // Re-enabled - testing shader compatibility
          // ->AddSink(lipstick_filter_)  // Disabled - Color Tint(Lipstick Effect) causing black output
          ->AddSink(saturation_filter_)
          ->AddSink(sink_raw_data_);
      
      // Start the camera
      source_camera_->Start();
      current_camera_id_ = camera_id;
      std::cout << "Fallback successful - using " << fallback_width << "x" << fallback_height << std::endl;
      
      // Update current resolution index to match the fallback
      for (size_t i = 0; i < available_resolutions_.size(); ++i) {
        if (available_resolutions_[i].width == fallback_width && 
            available_resolutions_[i].height == fallback_height) {
          current_resolution_index_ = i;
          selected_resolution_index_ = i;
          break;
        }
      }
    } else {
      std::cerr << "Complete camera failure - unable to initialize camera with any resolution" << std::endl;
    }
  }
}

// Switch to a different camera device
bool SwitchCamera(int new_camera_id) {
  if (new_camera_id == current_camera_id_) {
    return true; // Already using this camera
  }
  
  std::cout << "Switching from camera " << current_camera_id_ << " to camera " << new_camera_id << std::endl;
  
  // Use current resolution for new camera
  Resolution current_res = available_resolutions_[current_resolution_index_];
  RecreateCamera(new_camera_id, current_res.width, current_res.height);
  
  // Enumerate resolutions for the new camera
  EnumerateResolutionsForCamera(new_camera_id);
  
  // Reset resolution selection to first available resolution
  current_resolution_index_ = 0;
  selected_resolution_index_ = 0;
  
  return (source_camera_ != nullptr);
}

// Apply camera settings to the current camera
void ApplyCameraSettings() {
  if (!source_camera_) {
    return;
  }
  
  std::cout << "Applying camera settings using v4l2-ctl..." << std::endl;
  
  // Use v4l2-ctl to directly set camera properties
  // This works even when the camera is in use by another process
  
  // Brightness (0-255 range for v4l2-ctl)
  int brightness_v4l2 = (int)((camera_brightness_ + 100.0) * 255.0 / 200.0);
  brightness_v4l2 = std::max(0, std::min(255, brightness_v4l2));
  std::string brightness_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                               " --set-ctrl=brightness=" + std::to_string(brightness_v4l2);
  if (system(brightness_cmd.c_str()) == 0) {
    std::cout << "  Brightness: " << camera_brightness_ << " (v4l2: " << brightness_v4l2 << ")" << std::endl;
  }
  
  // Contrast (0-255 range for v4l2-ctl)
  int contrast_v4l2 = (int)(camera_contrast_ * 255.0 / 100.0);
  contrast_v4l2 = std::max(0, std::min(255, contrast_v4l2));
  std::string contrast_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                             " --set-ctrl=contrast=" + std::to_string(contrast_v4l2);
  if (system(contrast_cmd.c_str()) == 0) {
    std::cout << "  Contrast: " << camera_contrast_ << " (v4l2: " << contrast_v4l2 << ")" << std::endl;
  }
  
  // Saturation (0-255 range for v4l2-ctl)
  int saturation_v4l2 = (int)(camera_saturation_ * 255.0 / 100.0);
  saturation_v4l2 = std::max(0, std::min(255, saturation_v4l2));
  std::string saturation_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                               " --set-ctrl=saturation=" + std::to_string(saturation_v4l2);
  if (system(saturation_cmd.c_str()) == 0) {
    std::cout << "  Saturation: " << camera_saturation_ << " (v4l2: " << saturation_v4l2 << ")" << std::endl;
  }
  
  // Gain control - choose between auto and manual
  if (camera_auto_gain_) {
    // Set auto exposure mode (3 = Aperture Priority Mode)
    std::string auto_exposure_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                                    " --set-ctrl=auto_exposure=3";
    if (system(auto_exposure_cmd.c_str()) == 0) {
      std::cout << "  Auto Gain: ON (Aperture Priority Mode)" << std::endl;
    }
  } else {
    // Set manual exposure mode (1) to enable manual gain control
    std::string auto_exposure_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                                    " --set-ctrl=auto_exposure=1";
    system(auto_exposure_cmd.c_str());  // Set manual exposure mode
    
    // Apply manual gain setting
    int gain_v4l2 = (int)(camera_gain_ * 255.0 / 100.0);
    gain_v4l2 = std::max(0, std::min(255, gain_v4l2));
    std::string gain_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                           " --set-ctrl=gain=" + std::to_string(gain_v4l2);
    if (system(gain_cmd.c_str()) == 0) {
      std::cout << "  Manual Gain: " << camera_gain_ << " (v4l2: " << gain_v4l2 << ")" << std::endl;
    }
  }
  
  // Sharpness (0-255 range for v4l2-ctl)
  int sharpness_v4l2 = (int)(camera_sharpness_ * 255.0 / 100.0);
  sharpness_v4l2 = std::max(0, std::min(255, sharpness_v4l2));
  std::string sharpness_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                              " --set-ctrl=sharpness=" + std::to_string(sharpness_v4l2);
  if (system(sharpness_cmd.c_str()) == 0) {
    std::cout << "  Sharpness: " << camera_sharpness_ << " (v4l2: " << sharpness_v4l2 << ")" << std::endl;
  }
  
  // Zoom (100-500 range matches v4l2-ctl zoom_absolute)
  int zoom_v4l2 = (int)camera_zoom_;
  zoom_v4l2 = std::max(100, std::min(500, zoom_v4l2));
  std::string zoom_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                         " --set-ctrl=zoom_absolute=" + std::to_string(zoom_v4l2);
  if (system(zoom_cmd.c_str()) == 0) {
    std::cout << "  Zoom: " << camera_zoom_ << "% (v4l2: " << zoom_v4l2 << ")" << std::endl;
  }
  
  // Auto Focus
  std::string autofocus_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                              " --set-ctrl=focus_automatic_continuous=" + (camera_auto_focus_ ? "1" : "0");
  if (system(autofocus_cmd.c_str()) == 0) {
    std::cout << "  Auto Focus: " << (camera_auto_focus_ ? "ON" : "OFF") << std::endl;
  }
  
  std::cout << "Camera settings applied successfully using v4l2-ctl" << std::endl;
}

// Get current camera settings
void GetCurrentCameraSettings() {
  if (!source_camera_) {
    return;
  }
  
  std::cout << "Reading current camera settings using v4l2-ctl..." << std::endl;
  
  // Read current values using v4l2-ctl and parse the output
  std::string device = "/dev/video" + std::to_string(current_camera_id_);
  
  // Helper lambda to get control value
  auto getControlValue = [&](const std::string& control) -> int {
    std::string cmd = "v4l2-ctl --device=" + device + " --get-ctrl=" + control + " 2>/dev/null | cut -d'=' -f2 | tr -d ' \\n'";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return -1;
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe)) {
      result += buffer;
    }
    pclose(pipe);
    
    // Remove any whitespace and newlines
    result.erase(result.find_last_not_of(" \n\r\t") + 1);
    
    if (!result.empty() && result != "0" && result.find_first_not_of("0123456789-") == std::string::npos) {
      try {
        return std::stoi(result);
      } catch (const std::exception& e) {
        std::cout << "    Error parsing " << control << " value: " << result << std::endl;
        return -1;
      }
    }
    return -1;
  };
  
  // Read current values and convert to our ranges
  int brightness_v4l2 = getControlValue("brightness");
  if (brightness_v4l2 >= 0) {
    camera_brightness_ = (brightness_v4l2 * 200.0 / 255.0) - 100.0;
    std::cout << "  Brightness: " << camera_brightness_ << " (v4l2: " << brightness_v4l2 << ")" << std::endl;
  }
  
  int contrast_v4l2 = getControlValue("contrast");
  if (contrast_v4l2 >= 0) {
    camera_contrast_ = contrast_v4l2 * 100.0 / 255.0;
    std::cout << "  Contrast: " << camera_contrast_ << " (v4l2: " << contrast_v4l2 << ")" << std::endl;
  }
  
  int saturation_v4l2 = getControlValue("saturation");
  if (saturation_v4l2 >= 0) {
    camera_saturation_ = saturation_v4l2 * 100.0 / 255.0;
    std::cout << "  Saturation: " << camera_saturation_ << " (v4l2: " << saturation_v4l2 << ")" << std::endl;
  }
  
  int gain_v4l2 = getControlValue("gain");
  if (gain_v4l2 >= 0) {
    camera_gain_ = gain_v4l2 * 100.0 / 255.0;
    std::cout << "  Gain: " << camera_gain_ << " (v4l2: " << gain_v4l2 << ")" << std::endl;
  }
  
  int sharpness_v4l2 = getControlValue("sharpness");
  if (sharpness_v4l2 >= 0) {
    camera_sharpness_ = sharpness_v4l2 * 100.0 / 255.0;
    std::cout << "  Sharpness: " << camera_sharpness_ << " (v4l2: " << sharpness_v4l2 << ")" << std::endl;
  }
  
  int zoom_v4l2 = getControlValue("zoom_absolute");
  if (zoom_v4l2 >= 0) {
    camera_zoom_ = zoom_v4l2;
    std::cout << "  Zoom: " << camera_zoom_ << "% (v4l2: " << zoom_v4l2 << ")" << std::endl;
  }
  
  int autofocus_v4l2 = getControlValue("focus_automatic_continuous");
  if (autofocus_v4l2 >= 0) {
    camera_auto_focus_ = (autofocus_v4l2 > 0);
    std::cout << "  Auto Focus: " << (camera_auto_focus_ ? "ON" : "OFF") << " (v4l2: " << autofocus_v4l2 << ")" << std::endl;
  }
  
  std::cout << "Current camera settings retrieved successfully" << std::endl;
}

// Check shader compilation/linking errors
bool CheckShaderErrors(GLuint shader, const char* type) {
  GLint success;
  GLchar infoLog[1024];

  if (strcmp(type, "PROGRAM") != 0) {
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(shader, 1024, NULL, infoLog);
      std::cerr << "Shader compilation error: " << type << "\n"
                << infoLog << std::endl;
      return false;
    }
  } else {
    glGetProgramiv(shader, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader, 1024, NULL, infoLog);
      std::cerr << "Program linking error: " << type << "\n"
                << infoLog << std::endl;
      return false;
    }
  }
  return true;
}

// GLFW framebuffer resize callback
void OnFramebufferResize(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

// GLFW error callback
void ErrorCallback(int error, const char* description) {
  std::cerr << "GLFW Error: " << description << std::endl;
}

// Initialize GLFW and create window
bool SetupGlfwWindow(GLFWwindow** outWindow, int width = 1280, int height = 720) {
  // Set GLFW error callback
  glfwSetErrorCallback(ErrorCallback);

  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }

#ifdef GLFW_PLATFORM
  glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
#endif

  // Detect platform and adapt OpenGL version settings
#ifdef __APPLE__
  // macOS requires Core Profile and higher version of OpenGL
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#else
  // Linux(Ubuntu) platform uses more compatible configuration
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

  glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

  main_window_ = glfwCreateWindow(1280, 720, "Beauty Camera Studio", NULL, NULL);
  if (main_window_ == NULL) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }
  std::cout << "main_window_ pointer: " << main_window_ << std::endl;
if (main_window_ == NULL) {
    std::cerr << "Error: main_window_ is NULL before glfwMakeContextCurrent!" << std::endl;
    glfwTerminate();
    return false;
}


  // Initialize GLAD and setup window parameters
  glfwMakeContextCurrent(main_window_);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    glfwDestroyWindow(main_window_);
    glfwTerminate();
    return false;
  }

  glfwSetFramebufferSizeCallback(main_window_, OnFramebufferResize);

  return true;
}

// Initialize ImGui
void SetupImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(main_window_, true);

  // Select appropriate GLSL version based on platform
#ifdef __APPLE__
  // Use GLSL 3.3 version for macOS Core Profile
  const char* glsl_version = "#version 330 core";
#else
  // Use more compatible GLSL 1.30 version for Linux(Ubuntu)
  const char* glsl_version = "#version 130";
#endif
  ImGui_ImplOpenGL3_Init(glsl_version);
}

// Initialize GPUPixel filters and pipeline
void SetupFilterPipeline() {
  // Check if running from AppImage (environment variable set by AppRun script)
  const char* appimage_resource_path = getenv("BEAUTY_CAMERA_RESOURCE_PATH");
  
  std::string resource_path;
  if (appimage_resource_path) {
    // Running from AppImage - use the provided resource path
    resource_path = appimage_resource_path;
    std::cout << "[App] Running from AppImage, resource path: " << resource_path << std::endl;
  } else {
    // Running from regular installation - use executable directory
    resource_path = fs::path(GetExecutablePath()).parent_path().string();
    std::cout << "[App] Running from regular installation, resource path: " << resource_path << std::endl;
  }
  
  GPUPixel::SetResourcePath(resource_path);

  // Create filters - Only use known working ones
  beauty_filter_ = BeautyFaceFilter::Create();
  
  // Keep others created but don't add to pipeline
  reshape_filter_ = FaceReshapeFilter::Create();
  lipstick_filter_ = LipstickFilter::Create();    // Lipstick effect using face landmarks
  saturation_filter_ = SaturationFilter::Create();

#ifdef GPUPIXEL_ENABLE_FACE_DETECTOR
  face_detector_ = FaceDetector::Create();
#endif

  // Create OpenCV Face Detector
  opencv_face_detector_ = std::make_shared<OpenCVFaceDetector>();
  if (!opencv_face_detector_->Init()) {
    std::cerr << "Failed to initialize OpenCV face detector" << std::endl;
  } else {
    std::cout << "OpenCV face detector initialized successfully" << std::endl;
  }

  // Load beauty profiles and apply default profile
  std::cout << "Loading beauty profiles..." << std::endl;
  LoadProfilesFromFile();
  if (!default_profile_name_.empty() && beauty_profiles_.find(default_profile_name_) != beauty_profiles_.end()) {
    LoadBeautyProfile(default_profile_name_);
    std::cout << "Applied default beauty profile: " << default_profile_name_ << std::endl;
  }

  // Enumerate available cameras
  std::cout << "Enumerating available cameras..." << std::endl;
  EnumerateAvailableCameras();

  // Create source camera and sink raw data with detected resolution
  int init_width = 640, init_height = 480;  // Default fallback
  
  // Check if a profile was loaded and has a resolution setting
  bool profile_resolution_found = false;
  if (!default_profile_name_.empty() && beauty_profiles_.find(default_profile_name_) != beauty_profiles_.end()) {
    const BeautyProfile& profile = beauty_profiles_[default_profile_name_];
    // Find the profile's resolution in available resolutions
    for (int i = 0; i < available_resolutions_.size(); i++) {
      if (available_resolutions_[i].width == profile.resolution_width &&
          available_resolutions_[i].height == profile.resolution_height) {
        init_width = profile.resolution_width;
        init_height = profile.resolution_height;
        selected_resolution_index_ = i;
        current_resolution_index_ = i;
        profile_resolution_found = true;
        std::cout << "Using profile resolution: " << init_width << "x" << init_height << std::endl;
        break;
      }
    }
  }
  
  if (!profile_resolution_found && !available_resolutions_.empty()) {
    init_width = available_resolutions_[0].width;
    init_height = available_resolutions_[0].height;
    std::cout << "Using detected resolution: " << init_width << "x" << init_height << std::endl;
  }
  source_camera_ = SourceCamera::Create(current_camera_id_, init_width, init_height);
  sink_raw_data_ = SinkRawData::Create();

  if (!source_camera_) {
    std::cerr << "Failed to create camera source. Check if a camera is connected." << std::endl;
    return;
  }
  
  // Get current camera settings and apply defaults
  // GetCurrentCameraSettings();  // Temporarily disabled due to parsing issues
  
  // Set default camera settings only if no profile was loaded
  if (default_profile_name_.empty() || beauty_profiles_.find(default_profile_name_) == beauty_profiles_.end()) {
    std::cout << "No profile loaded, using default camera settings" << std::endl;
    camera_brightness_ = 0.0f;    // Default brightness
    camera_contrast_ = 50.0f;     // Default contrast  
    camera_saturation_ = 50.0f;   // Default saturation
    camera_gain_ = 0.0f;          // Default gain
    camera_sharpness_ = 50.0f;    // Default sharpness
    camera_zoom_ = 100.0f;        // Default zoom (100%)
    camera_auto_focus_ = true;    // Default auto focus on
    camera_auto_gain_ = true;     // Default auto gain on
  } else {
    std::cout << "Profile loaded, camera settings already configured from profile" << std::endl;
  }
  
  ApplyCameraSettings();
  
  // Build complete filter pipeline with face detection
  // Now using 106-point landmark format compatible with GPUPixel
  source_camera_->AddSink(beauty_filter_)
      ->AddSink(reshape_filter_)   // Re-enabled - testing shader compatibility
      // ->AddSink(lipstick_filter_)  // Disabled - Color Tint(Lipstick Effect) causing black output
      ->AddSink(saturation_filter_)// Warmth/blusher effect
      ->AddSink(sink_raw_data_);
      
  // Don't start camera capture immediately - wait for UI to be ready
}

// Update filter parameters from ImGui controls
void UpdateFilterParametersFromUI() {
  ImGui::Begin("Beauty Control Panel", nullptr,
               ImGuiWindowFlags_AlwaysAutoResize);

  bool parameters_changed = false;

  if (ImGui::SliderFloat("Smoothing", &beauty_strength_, 0.0f, 20.0f)) {
    beauty_filter_->SetBlurAlpha(beauty_strength_ / 10.0f);  // Range 0.0-2.0 for more intense smoothing
    parameters_changed = true;
  }

  if (ImGui::SliderFloat("Whitening", &whitening_strength_, 0.0f, 10.0f)) {
    beauty_filter_->SetWhite(whitening_strength_ / 20.0f);
    parameters_changed = true;
  }

  if (ImGui::SliderFloat("Face Slimming", &face_slim_strength_, 0.0f, 10.0f)) {
    reshape_filter_->SetFaceSlimLevel(face_slim_strength_ / 200.0f);
    parameters_changed = true;
  }

  if (ImGui::SliderFloat("Eye Enlarging", &eye_enlarge_strength_, 0.0f,
                         10.0f)) {
    reshape_filter_->SetEyeZoomLevel(eye_enlarge_strength_ / 100.0f);
    parameters_changed = true;
  }

  // Temporarily disabled - Color Tint and Warmth effects need shader fixes
  /*
  if (ImGui::SliderFloat("Color Tint (Lipstick Effect)", &color_tint_strength_, 0.0f, 10.0f)) {
    parameters_changed = true;
  }

  if (ImGui::SliderFloat("Warmth/Saturation", &warmth_strength_, 0.0f, 10.0f)) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "(Not working - shader compatibility issue)");
    parameters_changed = true;
  }
  */

  ImGui::Separator();
  ImGui::Text("Camera Selection:");
  
  // Camera selection dropdown
  if (!available_cameras_.empty() && available_cameras_[0] != "No cameras found") {
    std::vector<const char*> camera_names;
    for (const auto& name : available_cameras_) {
      camera_names.push_back(name.c_str());
    }
    
    if (ImGui::Combo("Camera Device", &selected_camera_id_, camera_names.data(), camera_names.size())) {
      if (selected_camera_id_ != current_camera_id_) {
        camera_changed_ = true;
        std::cout << "Camera selection changed to: " << selected_camera_id_ << std::endl;
      }
    }
    
    // Show current camera info
    ImGui::Text("Current: %s", available_cameras_[current_camera_id_].c_str());
    
    // Refresh cameras button
    if (ImGui::Button("Refresh Cameras")) {
      std::cout << "Refreshing camera list..." << std::endl;
      EnumerateAvailableCameras();
      // Reset selection if current camera is no longer available
      if (current_camera_id_ >= available_cameras_.size()) {
        selected_camera_id_ = 0;
        camera_changed_ = true;
      }
    }
  } else {
    ImGui::Text("No cameras detected");
    if (ImGui::Button("Retry Camera Detection")) {
      EnumerateAvailableCameras();
    }
  }

  ImGui::Separator();
  ImGui::Text("Resolution Selection:");
  
  // Resolution selection dropdown
  if (!available_resolutions_.empty()) {
    std::vector<const char*> resolution_names;
    for (const auto& res : available_resolutions_) {
      resolution_names.push_back(res.name.c_str());
    }
    
    if (ImGui::Combo("Resolution", &selected_resolution_index_, resolution_names.data(), resolution_names.size())) {
      if (selected_resolution_index_ != current_resolution_index_) {
        resolution_changed_ = true;
        std::cout << "Resolution selection changed to: " << available_resolutions_[selected_resolution_index_].name << std::endl;
      }
    }
    
    // Show current resolution info
    if (current_resolution_index_ < available_resolutions_.size()) {
      ImGui::Text("Current: %s", available_resolutions_[current_resolution_index_].name.c_str());
    }
    
    // Refresh resolutions button
    if (ImGui::Button("Refresh Resolutions")) {
      std::cout << "Refreshing resolution list for camera " << current_camera_id_ << "..." << std::endl;
      EnumerateResolutionsForCamera(current_camera_id_);
      // Reset selection to first available resolution
      if (!available_resolutions_.empty()) {
        current_resolution_index_ = 0;
        selected_resolution_index_ = 0;
      }
    }
    
    // Add a button to add custom resolution
    ImGui::Separator();
    static int custom_width = 1280;
    static int custom_height = 720;
    
    ImGui::Text("Custom Resolution:");
    ImGui::InputInt("Width", &custom_width);
    ImGui::InputInt("Height", &custom_height);
    
    if (ImGui::Button("Add & Test Custom Resolution")) {
      if (custom_width > 0 && custom_height > 0) {
        // Check if this resolution already exists
        bool already_exists = false;
        for (const auto& res : available_resolutions_) {
          if (res.width == custom_width && res.height == custom_height) {
            already_exists = true;
            break;
          }
        }
        
        if (!already_exists) {
          std::cout << "Testing custom resolution: " << custom_width << "x" << custom_height << std::endl;
          
          // Temporarily stop the main camera to free up the device for testing
          bool camera_was_running = false;
          if (source_camera_) {
            std::cout << "Temporarily stopping camera for resolution testing..." << std::endl;
            source_camera_->Stop();
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Give camera time to release
            camera_was_running = true;
          }
          
          // Test if the resolution actually works by trying to open the camera with it
          cv::VideoCapture test_cap(current_camera_id_, cv::CAP_V4L2);  // Force V4L2 backend
          if (test_cap.isOpened()) {
            test_cap.set(cv::CAP_PROP_FRAME_WIDTH, custom_width);
            test_cap.set(cv::CAP_PROP_FRAME_HEIGHT, custom_height);
            
            // Small delay to allow camera to adjust
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Get the actual resolution
            double actual_width = test_cap.get(cv::CAP_PROP_FRAME_WIDTH);
            double actual_height = test_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            
            test_cap.release();
            
            // Check if the camera accepted the resolution (with some tolerance)
            if (actual_width > 0 && actual_height > 0 && 
                abs((int)actual_width - custom_width) <= 2 && 
                abs((int)actual_height - custom_height) <= 2) {
              
              available_resolutions_.push_back(Resolution(custom_width, custom_height));
              // Sort resolutions by total pixels
              std::sort(available_resolutions_.begin(), available_resolutions_.end(),
                        [](const Resolution& a, const Resolution& b) {
                          return (a.width * a.height) < (b.width * b.height);
                        });
              std::cout << "✓ Custom resolution " << custom_width << "x" << custom_height << " validated and added!" << std::endl;
            } else {
              std::cout << "✗ Resolution " << custom_width << "x" << custom_height << " not supported by camera" << std::endl;
              std::cout << "  Camera returned: " << (int)actual_width << "x" << (int)actual_height << std::endl;
            }
          } else {
            std::cout << "✗ Failed to test resolution - camera not accessible" << std::endl;
          }
          
          // Restart the main camera if it was running
          if (camera_was_running && source_camera_) {
            std::cout << "Restarting main camera..." << std::endl;
            source_camera_->Start();
          }
        } else {
          std::cout << "Resolution " << custom_width << "x" << custom_height << " already exists" << std::endl;
        }
      }
    }
    
    if (ImGui::Button("Apply Resolution")) {
      if (selected_resolution_index_ != current_resolution_index_ && 
          selected_resolution_index_ < available_resolutions_.size()) {
        Resolution new_res = available_resolutions_[selected_resolution_index_];
        std::cout << "Attempting to apply resolution: " << new_res.name << std::endl;
        RecreateCamera(current_camera_id_, new_res.width, new_res.height);
        
        // Only update indices if camera creation was successful
        if (source_camera_) {
          current_resolution_index_ = selected_resolution_index_;
          resolution_changed_ = false;
          std::cout << "Successfully applied resolution: " << new_res.name << std::endl;
          
          // Reinitialize virtual camera with new resolution if it was enabled
          if (virtual_camera_enabled_) {
            CloseVirtualCamera();
            InitVirtualCamera();
          }
        } else {
          std::cout << "Failed to apply resolution - keeping previous selection" << std::endl;
          // Reset selection back to current working resolution
          selected_resolution_index_ = current_resolution_index_;
        }
      } else if (selected_resolution_index_ == current_resolution_index_) {
        std::cout << "Resolution is already active: " << available_resolutions_[current_resolution_index_].name << std::endl;
      }
    }
  } else {
    ImGui::Text("No resolutions available");
  }

  ImGui::Separator();
  ImGui::Text("Camera Settings:");
  
  // Camera settings controls
  bool camera_settings_changed = false;
  
  if (ImGui::SliderFloat("Camera Brightness", &camera_brightness_, -100.0f, 100.0f)) {
    camera_settings_changed = true;
  }
  
  if (ImGui::SliderFloat("Camera Contrast", &camera_contrast_, 0.0f, 100.0f)) {
    camera_settings_changed = true;
  }
  
  if (ImGui::SliderFloat("Camera Saturation", &camera_saturation_, 0.0f, 100.0f)) {
    camera_settings_changed = true;
  }
  
  // Auto Gain checkbox
  if (ImGui::Checkbox("Auto Gain", &camera_auto_gain_)) {
    camera_settings_changed = true;
  }
  
  // Manual gain slider (only enabled when auto gain is off)
  if (!camera_auto_gain_) {
    if (ImGui::SliderFloat("Camera Gain", &camera_gain_, 0.0f, 100.0f)) {
      camera_settings_changed = true;
    }
  } else {
    // Show grayed out gain slider when auto gain is enabled
    ImGui::BeginDisabled();
    ImGui::SliderFloat("Camera Gain (Auto)", &camera_gain_, 0.0f, 100.0f);
    ImGui::EndDisabled();
  }
  
  if (ImGui::SliderFloat("Camera Sharpness", &camera_sharpness_, 0.0f, 100.0f)) {
    camera_settings_changed = true;
  }
  
  if (ImGui::SliderFloat("Camera Zoom", &camera_zoom_, 100.0f, 500.0f, "%.0f%%")) {
    camera_settings_changed = true;
  }
  
  if (ImGui::Checkbox("Auto Focus", &camera_auto_focus_)) {
    camera_settings_changed = true;
  }
  
  // Apply settings button
  if (ImGui::Button("Apply Camera Settings") || camera_settings_changed) {
    ApplyCameraSettings();
    camera_settings_changed_ = true;
  }
  
  ImGui::SameLine();
  if (ImGui::Button("Reset Camera Settings")) {
    camera_brightness_ = 0.0f;
    camera_contrast_ = 50.0f;
    camera_saturation_ = 50.0f;
    camera_gain_ = 50.0f;
    camera_sharpness_ = 50.0f;
    camera_zoom_ = 100.0f;
    camera_auto_focus_ = true;
    camera_auto_gain_ = true;  // Default to auto gain
    ApplyCameraSettings();
    camera_settings_changed_ = true;
  }
  
  ImGui::SameLine();
  if (ImGui::Button("Get Current Settings")) {
    GetCurrentCameraSettings();
  }

  ImGui::Separator();
  ImGui::Text("Debug Options:");
  if (ImGui::Checkbox("Show Face Detection", &show_face_detection_)) {
    if (opencv_face_detector_) {
      opencv_face_detector_->SetDebugMode(show_face_detection_);
    }
  }

  // Beauty Profile Management
  ImGui::Separator();
  ImGui::Text("Beauty Profiles:");
  
  // Current profile display
  if (!current_profile_name_.empty()) {
    ImGui::Text("Current: %s", current_profile_name_.c_str());
  } else {
    ImGui::Text("Current: None");
  }
  
  // Profile selection dropdown
  if (ImGui::BeginCombo("Load Profile", current_profile_name_.c_str())) {
    for (const auto& pair : beauty_profiles_) {
      bool is_selected = (current_profile_name_ == pair.first);
      if (ImGui::Selectable(pair.first.c_str(), is_selected)) {
        LoadBeautyProfile(pair.first);
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  
  // Profile management buttons
  if (ImGui::Button("Save Current as Profile")) {
    show_profile_save_popup_ = true;
  }
  
  ImGui::SameLine();
  if (ImGui::Button("Profile Manager")) {
    show_profile_manager_ = true;
  }
  
  // Save profile popup
  if (show_profile_save_popup_) {
    ImGui::OpenPopup("Save Profile");
  }
  
  if (ImGui::BeginPopupModal("Save Profile", &show_profile_save_popup_, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Enter profile name:");
    ImGui::InputText("##ProfileName", profile_name_buffer_, sizeof(profile_name_buffer_));
    
    if (ImGui::Button("Save")) {
      if (strlen(profile_name_buffer_) > 0) {
        SaveBeautyProfile(std::string(profile_name_buffer_));
        show_profile_save_popup_ = false;
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) {
      show_profile_save_popup_ = false;
    }
    
    ImGui::EndPopup();
  }
  
  // Profile manager popup
  if (show_profile_manager_) {
    ImGui::OpenPopup("Profile Manager");
  }
  
  if (ImGui::BeginPopupModal("Profile Manager", &show_profile_manager_, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::Text("Manage Beauty Profiles");
    ImGui::Separator();
    
    // Default profile selection
    ImGui::Text("Default Profile:");
    if (ImGui::BeginCombo("##DefaultProfile", default_profile_name_.c_str())) {
      for (const auto& pair : beauty_profiles_) {
        bool is_selected = (default_profile_name_ == pair.first);
        if (ImGui::Selectable(pair.first.c_str(), is_selected)) {
          SetDefaultProfile(pair.first);
        }
        if (is_selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    
    ImGui::Separator();
    ImGui::Text("Available Profiles:");
    
    // List all profiles with delete option
    std::string profile_to_delete = "";
    for (const auto& pair : beauty_profiles_) {
      ImGui::Text("- %s", pair.first.c_str());
      ImGui::SameLine();
      std::string delete_button_id = "Delete##" + pair.first;
      if (ImGui::Button(delete_button_id.c_str())) {
        if (pair.first != "Default") {  // Don't allow deleting the Default profile
          profile_to_delete = pair.first;
        }
      }
    }
    
    // Delete selected profile
    if (!profile_to_delete.empty()) {
      beauty_profiles_.erase(profile_to_delete);
      if (current_profile_name_ == profile_to_delete) {
        current_profile_name_ = "";
      }
      if (default_profile_name_ == profile_to_delete) {
        SetDefaultProfile("Default");
      }
      SaveProfilesToFile();
    }
    
    ImGui::Separator();
    if (ImGui::Button("Close")) {
      show_profile_manager_ = false;
    }
    
    ImGui::EndPopup();
  }

  // Virtual Camera Controls
  ImGui::Separator();
  ImGui::Text("Virtual Camera:");
  
  // Virtual camera toggle button
  if (virtual_camera_enabled_) {
    if (ImGui::Button("Stop Virtual Camera")) {
      ToggleVirtualCamera();
    }
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "ACTIVE");
    ImGui::Text("Other apps can use: %s", virtual_camera_device_);
  } else {
    if (ImGui::Button("Start Virtual Camera")) {
      ToggleVirtualCamera();
    }
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 0, 0, 1), "STOPPED");
    ImGui::Text("Virtual camera will be available at: %s", virtual_camera_device_);
  }
  
  ImGui::Text("Use in Discord, OBS, Zoom, etc.");

  ImGui::End();

  // Note: Pipeline processing is now handled continuously in the main render loop
  // No need to trigger here since we process every frame automatically
}

// Update filter parameters directly
void UpdateFilterParameters() {
  beauty_filter_->SetBlurAlpha(beauty_strength_ / 10.0f);  // Range 0.0-2.0 for more intense smoothing
  beauty_filter_->SetWhite(whitening_strength_ / 20.0f);

  // Face reshape filter controls
  reshape_filter_->SetFaceSlimLevel(face_slim_strength_ / 200.0f);
  reshape_filter_->SetEyeZoomLevel(eye_enlarge_strength_ / 100.0f);

  // Color tint and warmth filter controls
  float lipstick_intensity = color_tint_strength_ / 10.0f;
  lipstick_filter_->SetBlendLevel(lipstick_intensity * 0.5f);  // Scale to [0, 0.5] range
  
  float warmth = warmth_strength_ / 10.0f;
  saturation_filter_->setSaturation(1.0f + warmth * 0.5f);
}

// Render RGBA data to screen
void RenderRGBAToScreen(const uint8_t* rgba_data, int width, int height) {
  if (!rgba_data || width <= 0 || height <= 0) {
    return;
  }

  // Get current window size for viewport settings
  int window_width, window_height;
  glfwGetFramebufferSize(main_window_, &window_width, &window_height);

  // Render RGBA data to screen
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glViewport(0, 0, window_width, window_height);

  // Note: Don't clear framebuffer here, or it will overwrite ImGui rendering
  // glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  // glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Create a texture from the RGBA data
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, rgba_data);

  // Use appropriate shader code based on platform
#ifdef __APPLE__
  // macOS uses 330 core version
  const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
      gl_Position = vec4(aPos, 1.0);
      TexCoord = aTexCoord;
    }
  )";

  const char* fragmentShaderSource = R"(
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;
    uniform sampler2D texture1;
    void main() {
      FragColor = texture(texture1, TexCoord);
    }
  )";
#else
  // Linux(Ubuntu) uses more compatible 130 version
  const char* vertexShaderSource = R"(
    #version 130
    attribute vec3 aPos;
    attribute vec2 aTexCoord;
    varying vec2 TexCoord;
    void main() {
      gl_Position = vec4(aPos, 1.0);
      TexCoord = aTexCoord;
    }
  )";

  const char* fragmentShaderSource = R"(
    #version 130
    varying vec2 TexCoord;
    uniform sampler2D texture1;
    void main() {
      gl_FragColor = texture2D(texture1, TexCoord);
    }
  )";
#endif

  // Compile shaders
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  if (!CheckShaderErrors(vertexShader, "VERTEX")) {
    return;
  }

  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  if (!CheckShaderErrors(fragmentShader, "FRAGMENT")) {
    return;
  }

  // Create shader program
  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  if (!CheckShaderErrors(shaderProgram, "PROGRAM")) {
    return;
  }

  // Use shader program
  glUseProgram(shaderProgram);

  // Calculate aspect ratio and adjust vertices to maintain aspect ratio
  float imageAspectRatio = (float)width / (float)height;
  float windowAspectRatio = 1280.0f / 720.0f;

  float scaleX = 1.0f;
  float scaleY = 1.0f;

  if (imageAspectRatio > windowAspectRatio) {
    // Image is wider than window
    scaleY = windowAspectRatio / imageAspectRatio;
  } else {
    // Image is taller than window
    scaleX = imageAspectRatio / windowAspectRatio;
  }

  // Set up vertex data and buffers with corrected aspect ratio
  float vertices[] = {
      // positions        // texture coords
      -scaleX, -scaleY, 0.0f,
      0.0f,    1.0f,  // Bottom left (flipped Y texture coord)
      scaleX,  -scaleY, 0.0f,
      1.0f,    1.0f,  // Bottom right (flipped Y texture coord)
      scaleX,  scaleY,  0.0f,
      1.0f,    0.0f,  // Top right (flipped Y texture coord)
      -scaleX, scaleY,  0.0f,
      0.0f,    0.0f  // Top left (flipped Y texture coord)
  };

  unsigned int indices[] = {0, 1, 2, 2, 3, 0};

  // Draw
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture);

  // Create VAO first (required for Core Profile)
  GLuint VAO, VBO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // Position attribute (location = 0)
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // Texture coord attribute (location = 1)
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float),
                        (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Set texture uniform
  GLint texUniform = glGetUniformLocation(shaderProgram, "texture1");
  glUniform1i(texUniform, 0);

  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

  // Clean up
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glDeleteTextures(1, &texture);
  glDeleteProgram(shaderProgram);
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
}

// Render a single frame
void RenderFrame() {
  // Delay initial processing to give camera time to capture frames
  static int frame_count = 0;
  frame_count++;

  // Clear the framebuffer
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  // Start ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // Update filter parameters
  UpdateFilterParametersFromUI();
  static bool camera_started = false;
  
  // Start camera after UI is stable (around frame 10)
  if (!camera_started && frame_count > 10) {
    if (source_camera_) {
      source_camera_->Start();
      camera_started = true;
    }
  }
  
  // Handle camera switching
  if (camera_changed_ && camera_started) {
    std::cout << "Processing camera change from " << current_camera_id_ << " to " << selected_camera_id_ << std::endl;
    if (SwitchCamera(selected_camera_id_)) {
      std::cout << "Camera switch successful" << std::endl;
    } else {
      std::cout << "Camera switch failed, reverting selection" << std::endl;
      selected_camera_id_ = current_camera_id_;  // Revert UI selection
    }
    camera_changed_ = false;
  }
  
  // Handle resolution changes
  if (resolution_changed_ && camera_started) {
    std::cout << "Processing resolution change to " << available_resolutions_[selected_resolution_index_].name << std::endl;
    if (selected_resolution_index_ < available_resolutions_.size()) {
      Resolution new_res = available_resolutions_[selected_resolution_index_];
      RecreateCamera(current_camera_id_, new_res.width, new_res.height);
      current_resolution_index_ = selected_resolution_index_;
      std::cout << "Resolution change successful" << std::endl;
    } else {
      std::cout << "Invalid resolution index, reverting selection" << std::endl;
      selected_resolution_index_ = current_resolution_index_;  // Revert UI selection
    }
    resolution_changed_ = false;
  }
  
  // Only start camera processing after camera has been running for a while
  if (camera_started && frame_count > 10) {  // Reduced wait time for faster startup
    // Process latest camera frame in main thread and handle initial processing
    if (source_camera_) {
      source_camera_->ProcessLatestFrame();
    }

    if (need_initial_processing && frame_count > 15) {  // Reduced wait time for faster startup
      if (source_camera_) {
        source_camera_->TriggerPipelineProcessing();
        need_initial_processing = false;
      }
    } else if (!need_initial_processing && frame_count > 15) {
      // Continue triggering pipeline processing for every frame after initial setup
      if (source_camera_) {
        // Perform face detection on current frame
        if (opencv_face_detector_ && frame_count % 5 == 0) {  // Run face detection every 5 frames for performance
          cv::Mat current_frame = source_camera_->GetLatestFrame();
          if (!current_frame.empty()) {
            std::vector<float> landmarks = opencv_face_detector_->DetectFace(current_frame);
            
            if (!landmarks.empty()) {
              // Apply landmarks to face-dependent filters
              reshape_filter_->SetFaceLandmarks(landmarks);
              lipstick_filter_->SetFaceLandmarks(landmarks);  // Now working with face detection
            }
          }
        }
        
        source_camera_->TriggerPipelineProcessing();
      }
    }

    // Get the latest processed frame from sink
    const unsigned char* buffer = nullptr;
    int width = 0;
    int height = 0;
    
    // Only get buffer if sink is ready and has processed data
    // Check if sink actually has processed frames before calling GetRgbaBuffer
    if (sink_raw_data_ && !need_initial_processing && frame_count > 20) {
      // First check if the sink has received any input framebuffers
      // by checking its dimensions - they'll be > 0 only after processing
      if (sink_raw_data_->GetWidth() > 0 && sink_raw_data_->GetHeight() > 0) {
        // Add additional safety check - wait longer before first buffer access
        if (frame_count > 25) {  // Much earlier buffer access
          buffer = sink_raw_data_->GetRgbaBuffer();
          width = sink_raw_data_->GetWidth();
          height = sink_raw_data_->GetHeight();
        }
      }
    }

    if (buffer && width > 0 && height > 0) {
#ifdef GPUPIXEL_ENABLE_FACE_DETECTOR
      // Detect facial landmarks
      std::vector<float> landmarks = face_detector_->Detect(
          buffer, width, height, width * 4, GPUPIXEL_MODE_FMT_PICTURE,
          GPUPIXEL_FRAME_TYPE_RGBA);

      if (!landmarks.empty()) {
        lipstick_filter_->SetFaceLandmarks(landmarks);
        blusher_filter_->SetFaceLandmarks(landmarks);
        reshape_filter_->SetFaceLandmarks(landmarks);
      }
#endif

      // Render RGBA data to screen
      if (show_face_detection_ && opencv_face_detector_) {
        // Create a copy of the buffer for debug visualization
        std::vector<unsigned char> debug_buffer(buffer, buffer + (width * height * 4));
        
        // Convert to OpenCV Mat for drawing overlays
        cv::Mat debug_frame(height, width, CV_8UC4, debug_buffer.data());
        
        // Get the last detected faces and eyes from the face detector
        std::vector<cv::Rect> faces = opencv_face_detector_->GetLastDetectedFaces();
        std::vector<cv::Rect> eyes = opencv_face_detector_->GetLastDetectedEyes();
        
        if (!faces.empty()) {
          // Get camera frame dimensions for scaling
          cv::Mat current_frame = source_camera_->GetLatestFrame();
          if (!current_frame.empty()) {
            float scale_x = (float)width / current_frame.cols;
            float scale_y = (float)height / current_frame.rows;
            
            // Draw face rectangles
            for (const auto& face : faces) {
              cv::Rect scaled_face(
                (int)(face.x * scale_x), (int)(face.y * scale_y),
                (int)(face.width * scale_x), (int)(face.height * scale_y)
              );
              cv::rectangle(debug_frame, scaled_face, cv::Scalar(0, 255, 0, 255), 2);
              cv::putText(debug_frame, "Face", 
                         cv::Point(scaled_face.x, scaled_face.y - 10),
                         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0, 255), 1);
            }
            
            // Draw eye rectangles
            int eye_count = 0;
            for (const auto& eye : eyes) {
              cv::Rect scaled_eye(
                (int)(eye.x * scale_x), (int)(eye.y * scale_y),
                (int)(eye.width * scale_x), (int)(eye.height * scale_y)
              );
              cv::rectangle(debug_frame, scaled_eye, cv::Scalar(255, 0, 0, 255), 2);
              cv::putText(debug_frame, "Eye" + std::to_string(++eye_count),
                         cv::Point(scaled_eye.x, scaled_eye.y - 5),
                         cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0, 255), 1);
            }
          }
        }
        
        RenderRGBAToScreen(debug_buffer.data(), width, height);
      } else {
        RenderRGBAToScreen(buffer, width, height);
      }
    }
  }

  // Render ImGui
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

  // Write frame to virtual camera if enabled
  WriteFrameToVirtualCamera();

  // Swap buffers and poll events
  glfwSwapBuffers(main_window_);
  glfwPollEvents();
}

// Clean up resources
void CleanupResources() {
  // Close virtual camera first
  CloseVirtualCamera();
  
  // Stop camera first
  if (source_camera_) {
    source_camera_->Stop();
  }

  // Explicitly destroy GPUPixel objects before GLFW cleanup
  source_camera_.reset();
  sink_raw_data_.reset();
  lipstick_filter_.reset();
  saturation_filter_.reset();
  reshape_filter_.reset();
  beauty_filter_.reset();
  opencv_face_detector_.reset();
#ifdef GPUPIXEL_ENABLE_FACE_DETECTOR
  face_detector_.reset();
#endif

  // Cleanup ImGui
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  // Cleanup GLFW last
  if (main_window_) {
    glfwDestroyWindow(main_window_);
    main_window_ = nullptr;
  }
  glfwTerminate();
}

// Beauty Profile System Implementation
std::string GetProfilesDirectory() {
  // Use user's config directory instead of executable directory
  // This ensures profiles work correctly with AppImage and are user-specific
  const char* home = getenv("HOME");
  if (!home) {
    home = "/tmp"; // Fallback if HOME is not set
  }
  
  std::string profiles_dir = std::string(home) + "/.config/beauty-camera-studio";
  
  // Create profiles directory if it doesn't exist
  fs::create_directories(profiles_dir);
  
  return profiles_dir;
}

void SaveBeautyProfile(const std::string& name) {
  // Get current resolution
  int current_width = 640, current_height = 480;  // Default fallback
  if (current_resolution_index_ < available_resolutions_.size()) {
    current_width = available_resolutions_[current_resolution_index_].width;
    current_height = available_resolutions_[current_resolution_index_].height;
  }
  
  std::cout << "Saving profile '" << name << "' with values:" << std::endl;
  std::cout << "  Beauty: " << beauty_strength_ << std::endl;
  std::cout << "  Whitening: " << whitening_strength_ << std::endl;
  std::cout << "  Face Slim: " << face_slim_strength_ << std::endl;
  std::cout << "  Eye Enlarge: " << eye_enlarge_strength_ << std::endl;
  std::cout << "  Color Tint: " << color_tint_strength_ << std::endl;
  std::cout << "  Warmth: " << warmth_strength_ << std::endl;
  std::cout << "  Camera Brightness: " << camera_brightness_ << std::endl;
  std::cout << "  Camera Contrast: " << camera_contrast_ << std::endl;
  std::cout << "  Camera Saturation: " << camera_saturation_ << std::endl;
  std::cout << "  Camera Gain: " << camera_gain_ << std::endl;
  std::cout << "  Camera Sharpness: " << camera_sharpness_ << std::endl;
  std::cout << "  Camera Zoom: " << camera_zoom_ << std::endl;
  std::cout << "  Auto Focus: " << (camera_auto_focus_ ? "ON" : "OFF") << std::endl;
  std::cout << "  Auto Gain: " << (camera_auto_gain_ ? "ON" : "OFF") << std::endl;
  std::cout << "  Resolution: " << current_width << "x" << current_height << std::endl;
  
  BeautyProfile profile(name, beauty_strength_, whitening_strength_,
                       face_slim_strength_, eye_enlarge_strength_,
                       color_tint_strength_, warmth_strength_,
                       camera_brightness_, camera_contrast_, camera_saturation_,
                       camera_gain_, camera_sharpness_, camera_zoom_,
                       camera_auto_focus_, camera_auto_gain_,
                       current_width, current_height);
  
  beauty_profiles_[name] = profile;
  current_profile_name_ = name;
  
  SaveProfilesToFile();
  std::cout << "Beauty profile '" << name << "' saved successfully" << std::endl;
}

void LoadBeautyProfile(const std::string& name) {
  auto it = beauty_profiles_.find(name);
  if (it != beauty_profiles_.end()) {
    const BeautyProfile& profile = it->second;
    
    std::cout << "Loading profile '" << name << "' with values:" << std::endl;
    std::cout << "  Beauty: " << profile.beauty_strength << std::endl;
    std::cout << "  Whitening: " << profile.whitening_strength << std::endl;
    std::cout << "  Face Slim: " << profile.face_slim_strength << std::endl;
    std::cout << "  Eye Enlarge: " << profile.eye_enlarge_strength << std::endl;
    std::cout << "  Color Tint: " << profile.color_tint_strength << std::endl;
    std::cout << "  Warmth: " << profile.warmth_strength << std::endl;
    std::cout << "  Camera Brightness: " << profile.camera_brightness << std::endl;
    std::cout << "  Camera Contrast: " << profile.camera_contrast << std::endl;
    std::cout << "  Camera Saturation: " << profile.camera_saturation << std::endl;
    std::cout << "  Camera Gain: " << profile.camera_gain << std::endl;
    std::cout << "  Camera Sharpness: " << profile.camera_sharpness << std::endl;
    std::cout << "  Camera Zoom: " << profile.camera_zoom << std::endl;
    std::cout << "  Auto Focus: " << (profile.camera_auto_focus ? "ON" : "OFF") << std::endl;
    std::cout << "  Auto Gain: " << (profile.camera_auto_gain ? "ON" : "OFF") << std::endl;
    std::cout << "  Resolution: " << profile.resolution_width << "x" << profile.resolution_height << std::endl;
    
    // Restore beauty filter settings
    beauty_strength_ = profile.beauty_strength;
    whitening_strength_ = profile.whitening_strength;
    face_slim_strength_ = profile.face_slim_strength;
    eye_enlarge_strength_ = profile.eye_enlarge_strength;
    color_tint_strength_ = profile.color_tint_strength;
    warmth_strength_ = profile.warmth_strength;
    
    // Restore camera settings
    camera_brightness_ = profile.camera_brightness;
    camera_contrast_ = profile.camera_contrast;
    camera_saturation_ = profile.camera_saturation;
    camera_gain_ = profile.camera_gain;
    camera_sharpness_ = profile.camera_sharpness;
    camera_zoom_ = profile.camera_zoom;
    camera_auto_focus_ = profile.camera_auto_focus;
    camera_auto_gain_ = profile.camera_auto_gain;
    
    // Find and set the resolution
    for (int i = 0; i < available_resolutions_.size(); i++) {
      if (available_resolutions_[i].width == profile.resolution_width &&
          available_resolutions_[i].height == profile.resolution_height) {
        selected_resolution_index_ = i;
        current_resolution_index_ = i;
        resolution_changed_ = true;  // Trigger resolution change
        std::cout << "  Resolution restored to: " << available_resolutions_[i].name << std::endl;
        break;
      }
    }
    
    current_profile_name_ = name;
    
    // Apply the loaded settings to filters and camera
    ApplyCurrentProfile();
    ApplyCameraSettings();
    
    std::cout << "Beauty profile '" << name << "' loaded successfully" << std::endl;
  } else {
    std::cout << "Beauty profile '" << name << "' not found" << std::endl;
  }
}

void ApplyCurrentProfile() {
  // Update filter parameters when profile is loaded
  if (beauty_filter_) {
    beauty_filter_->SetBlurAlpha(beauty_strength_ / 10.0f);  // Range 0.0-2.0 for more intense smoothing
    beauty_filter_->SetWhite(whitening_strength_ / 20.0f);
  }
  
  // Update saturation filter for warmth effect
  if (saturation_filter_) {
    saturation_filter_->setSaturation(1.0f + (warmth_strength_ / 20.0f));
  }
  
  // Face reshape filter parameters would be applied here when enabled
  // Lipstick filter would be applied here when enabled
  
  std::cout << "Applied beauty profile settings to filters" << std::endl;
}

void SaveProfilesToFile() {
  std::string profiles_dir = GetProfilesDirectory();
  std::string profiles_file = profiles_dir + "/beauty_profiles.txt";
  std::string default_file = profiles_dir + "/default_profile.txt";
  
  // Save profiles
  std::ofstream file(profiles_file);
  if (file.is_open()) {
    for (const auto& pair : beauty_profiles_) {
      const BeautyProfile& profile = pair.second;
      file << profile.name << "|"
           << profile.beauty_strength << "|"
           << profile.whitening_strength << "|"
           << profile.face_slim_strength << "|"
           << profile.eye_enlarge_strength << "|"
           << profile.color_tint_strength << "|"
           << profile.warmth_strength << "|"
           << profile.camera_brightness << "|"
           << profile.camera_contrast << "|"
           << profile.camera_saturation << "|"
           << profile.camera_gain << "|"
           << profile.camera_sharpness << "|"
           << profile.camera_zoom << "|"
           << (profile.camera_auto_focus ? 1 : 0) << "|"
           << (profile.camera_auto_gain ? 1 : 0) << "|"
           << profile.resolution_width << "|"
           << profile.resolution_height << std::endl;
    }
    file.close();
  }
  
  // Save default profile name
  std::ofstream default_out(default_file);
  if (default_out.is_open()) {
    default_out << default_profile_name_ << std::endl;
    default_out.close();
  }
}

void LoadProfilesFromFile() {
  std::string profiles_dir = GetProfilesDirectory();
  std::string profiles_file = profiles_dir + "/beauty_profiles.txt";
  std::string default_file = profiles_dir + "/default_profile.txt";
  
  // Load profiles
  std::ifstream file(profiles_file);
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string name, token;
      
      if (std::getline(iss, name, '|')) {
        BeautyProfile profile;
        profile.name = name;
        
        if (std::getline(iss, token, '|')) profile.beauty_strength = std::stof(token);
        if (std::getline(iss, token, '|')) profile.whitening_strength = std::stof(token);
        if (std::getline(iss, token, '|')) profile.face_slim_strength = std::stof(token);
        if (std::getline(iss, token, '|')) profile.eye_enlarge_strength = std::stof(token);
        if (std::getline(iss, token, '|')) profile.color_tint_strength = std::stof(token);
        if (std::getline(iss, token, '|')) profile.warmth_strength = std::stof(token);
        
        // Load camera settings (with backwards compatibility for old profiles)
        if (std::getline(iss, token, '|')) profile.camera_brightness = std::stof(token);
        if (std::getline(iss, token, '|')) profile.camera_contrast = std::stof(token);
        if (std::getline(iss, token, '|')) profile.camera_saturation = std::stof(token);
        if (std::getline(iss, token, '|')) profile.camera_gain = std::stof(token);
        if (std::getline(iss, token, '|')) profile.camera_sharpness = std::stof(token);
        if (std::getline(iss, token, '|')) profile.camera_zoom = std::stof(token);
        if (std::getline(iss, token, '|')) profile.camera_auto_focus = (std::stoi(token) != 0);
        if (std::getline(iss, token, '|')) profile.camera_auto_gain = (std::stoi(token) != 0);
        if (std::getline(iss, token, '|')) profile.resolution_width = std::stoi(token);
        if (std::getline(iss, token, '|')) profile.resolution_height = std::stoi(token);
        
        beauty_profiles_[name] = profile;
      }
    }
    file.close();
    std::cout << "Loaded " << beauty_profiles_.size() << " beauty profiles" << std::endl;
  }
  
  // Load default profile name
  std::ifstream default_in(default_file);
  if (default_in.is_open()) {
    std::getline(default_in, default_profile_name_);
    default_in.close();
    std::cout << "Default profile: " << default_profile_name_ << std::endl;
  }
  
  // Create a "Default" profile if none exist
  if (beauty_profiles_.empty()) {
    SaveBeautyProfile("Default");
    SetDefaultProfile("Default");
  }
}

void SetDefaultProfile(const std::string& name) {
  default_profile_name_ = name;
  SaveProfilesToFile();
  std::cout << "Default profile set to: " << name << std::endl;
}

// Virtual Camera Functions
bool InitVirtualCamera() {
  if (virtual_camera_enabled_) {
    return true;  // Already initialized
  }
  
  std::cout << "Initializing virtual camera at " << virtual_camera_device_ << std::endl;
  
  // Check if virtual camera device exists
  virtual_camera_fd_ = open(virtual_camera_device_, O_RDWR);
  if (virtual_camera_fd_ < 0) {
    std::cerr << "Failed to open virtual camera device: " << virtual_camera_device_ << std::endl;
    std::cerr << "Make sure v4l2loopback is loaded: sudo modprobe v4l2loopback" << std::endl;
    return false;
  }
  
  // Get current camera resolution for virtual camera
  int virt_width = 640;  // Default fallback
  int virt_height = 480;
  
  // Try to get resolution from sink first
  if (sink_raw_data_) {
    virt_width = sink_raw_data_->GetWidth();
    virt_height = sink_raw_data_->GetHeight();
  }
  
  // If sink doesn't have valid dimensions, try to get from current resolution setting
  if (virt_width <= 0 || virt_height <= 0) {
    if (!available_resolutions_.empty() && current_resolution_index_ < available_resolutions_.size()) {
      virt_width = available_resolutions_[current_resolution_index_].width;
      virt_height = available_resolutions_[current_resolution_index_].height;
    } else {
      virt_width = 640;
      virt_height = 480;
    }
  }
  
  // Set up video format - use RGB24 which is easier to work with
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  fmt.fmt.pix.width = virt_width;
  fmt.fmt.pix.height = virt_height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;  // Changed to RGB24 for easier conversion
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  fmt.fmt.pix.bytesperline = virt_width * 3;  // RGB24 uses 3 bytes per pixel
  fmt.fmt.pix.sizeimage = virt_width * virt_height * 3;
  fmt.fmt.pix.colorspace = V4L2_COLORSPACE_SRGB;
  
  if (ioctl(virtual_camera_fd_, VIDIOC_S_FMT, &fmt) < 0) {
    std::cerr << "Failed to set virtual camera format: " << strerror(errno) << std::endl;
    close(virtual_camera_fd_);
    virtual_camera_fd_ = -1;
    return false;
  }
  
  virtual_camera_enabled_ = true;
  std::cout << "Virtual camera initialized successfully at " << virt_width << "x" << virt_height << " RGB24" << std::endl;
  return true;
}

void CloseVirtualCamera() {
  if (virtual_camera_fd_ >= 0) {
    close(virtual_camera_fd_);
    virtual_camera_fd_ = -1;
  }
  virtual_camera_enabled_ = false;
  std::cout << "Virtual camera closed" << std::endl;
}

void WriteFrameToVirtualCamera() {
  if (!virtual_camera_enabled_ || virtual_camera_fd_ < 0 || !sink_raw_data_) {
    return;
  }
  
  // Get the processed frame from the sink - use RGBA buffer
  const uint8_t* rgba_buffer = sink_raw_data_->GetRgbaBuffer();
  if (!rgba_buffer) {
    return;
  }
  
  int width = sink_raw_data_->GetWidth();
  int height = sink_raw_data_->GetHeight();
  
  if (width <= 0 || height <= 0) {
    return;
  }
  
  // Convert RGBA to RGB24 format (remove alpha channel)
  static std::vector<uint8_t> rgb24_buffer;
  int rgb24_size = width * height * 3;  // RGB24 uses 3 bytes per pixel
  rgb24_buffer.resize(rgb24_size);
  
  // Convert RGBA to RGB24 by removing alpha channel
  for (int i = 0; i < width * height; i++) {
    rgb24_buffer[i * 3] = rgba_buffer[i * 4];     // R
    rgb24_buffer[i * 3 + 1] = rgba_buffer[i * 4 + 1]; // G
    rgb24_buffer[i * 3 + 2] = rgba_buffer[i * 4 + 2]; // B
    // Skip alpha channel (rgba_buffer[i * 4 + 3])
  }
  
  // Write frame to virtual camera
  ssize_t bytes_written = write(virtual_camera_fd_, rgb24_buffer.data(), rgb24_size);
  if (bytes_written < 0) {
    std::cerr << "Failed to write frame to virtual camera: " << strerror(errno) << std::endl;
  } else if (bytes_written != rgb24_size) {
    std::cerr << "Partial write to virtual camera: " << bytes_written << "/" << rgb24_size << std::endl;
  }
}

void ToggleVirtualCamera() {
  if (virtual_camera_enabled_) {
    CloseVirtualCamera();
  } else {
    InitVirtualCamera();
  }
}

int main() {
  std::cout << "=== APP STARTING ===" << std::endl;
  std::cout.flush();

#ifdef _WIN32
  // Set DLL search path
  std::string exePath = GetExecutablePath();
  char dllDir[MAX_PATH];
  sprintf_s(dllDir, MAX_PATH, "%s\\..\\lib", exePath.c_str());
  SetDllDirectoryA(dllDir);
#endif

  // Initialize window and OpenGL context
  if (!SetupGlfwWindow(&main_window_)) {
    return -1;
  }

  // Setup ImGui interface
  SetupImGui();

  // Initialize filters and pipeline
  SetupFilterPipeline();

  std::cout << "=== ENTERING MAIN LOOP ===" << std::endl;
  std::cout.flush();

  // Main render loop
  while (!glfwWindowShouldClose(main_window_)) {
    // Render frame
    RenderFrame();
  }

  std::cout << "=== EXITING MAIN LOOP ===" << std::endl;
  std::cout.flush();

  // Cleanup
  CleanupResources();

  return 0;
}
