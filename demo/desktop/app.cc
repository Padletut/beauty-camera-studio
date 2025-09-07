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
#include "gpupixel/filter/blusher_filter.h"
#include "gpupixel/utils/mediapipe_segmentation.h"
#include "gpupixel/utils/rvm_processor.h"
#include "imgui.h"
#include <opencv2/opencv.hpp>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <fstream>
#include <cstdio>

using namespace gpupixel;

// Global MediaPipe segmentation instance
std::unique_ptr<MediaPipeSegmentation> mediapipe_segmentation;

// Filters
std::shared_ptr<BeautyFaceFilter> beauty_filter_;
std::shared_ptr<FaceReshapeFilter> reshape_filter_;
std::shared_ptr<LipstickFilter> lipstick_filter_;   // For lipstick effect using face landmarks
std::shared_ptr<BlusherFilter> blusher_filter_;     // For blusher effect using face landmarks
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

// MediaPipe AI Segmentation parameters
float ai_confidence_threshold_ = 0.1f;  // AI confidence threshold (0.3-0.8) - MediaPipe default
float ai_temporal_smoothing_ = 0.7f;    // AI temporal smoothing (0.3-0.9) - Optimized for streaming

// RVM Enhanced Processing parameters
bool enable_rvm_processing_ = false;    // Enable RVM-style processing
int rvm_temporal_buffer_size_ = 5;      // RVM temporal buffer size (3-10)
float rvm_temporal_weight_ = 0.3f;      // RVM temporal weight (0.1-0.8)
float rvm_motion_threshold_ = 0.1f;     // RVM motion threshold (0.05-0.3)
bool rvm_motion_compensation_ = true;   // RVM motion compensation
bool rvm_edge_refinement_ = true;       // RVM edge refinement

// Background Detection Configuration
int target_class_selection_ = 0;        // 0=AUTO_DETECT, 1=PERSON, 2=BACKGROUND
bool output_confidence_masks_ = true;   // Output confidence masks
bool output_category_mask_ = false;     // Output category masks

// Debug options
bool show_face_detection_ = false;      // Show face detection rectangles
bool show_body_detection_ = false;      // Show body/person detection mask

// For body detection visualization
cv::Mat last_person_mask_;

// Background Effects
enum class BackgroundMode {
  NONE = 0,
  BLUR,
  CUSTOM_IMAGE
};

BackgroundMode background_mode_ = BackgroundMode::BLUR;
float background_blur_strength_ = 15.0f;  // Blur kernel size (5-51, odd numbers)
std::string custom_background_path_ = "";
cv::Mat custom_background_image_;
bool background_image_loaded_ = false;

// MediaPipe Persistent Process - Removed unused code

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
  float camera_focus;
  bool camera_auto_gain;
  
  // Resolution settings
  int resolution_width;
  int resolution_height;
  
  BeautyProfile() : name("Default"), beauty_strength(0.0f), whitening_strength(0.0f),
                    face_slim_strength(0.0f), eye_enlarge_strength(0.0f),
                    color_tint_strength(0.0f), warmth_strength(0.0f),
                    camera_brightness(0.0f), camera_contrast(50.0f), camera_saturation(50.0f),
                    camera_gain(0.0f), camera_sharpness(50.0f), camera_zoom(100.0f),
                    camera_auto_focus(true), camera_focus(50.0f), camera_auto_gain(true),
                    resolution_width(640), resolution_height(480) {}
                    
  BeautyProfile(const std::string& n, float beauty, float whitening, float slim,
                float eye, float tint, float warmth, float brightness, float contrast,
                float saturation, float gain, float sharpness, float zoom,
                bool auto_focus, float focus, bool auto_gain, int width, int height)
    : name(n), beauty_strength(beauty), whitening_strength(whitening),
      face_slim_strength(slim), eye_enlarge_strength(eye),
      color_tint_strength(tint), warmth_strength(warmth),
      camera_brightness(brightness), camera_contrast(contrast), camera_saturation(saturation),
      camera_gain(gain), camera_sharpness(sharpness), camera_zoom(zoom),
      camera_auto_focus(auto_focus), camera_focus(focus), camera_auto_gain(auto_gain),
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
float camera_focus_ = 50.0f;            // Manual focus value (0 to 100)
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
  
  // Manual Focus (only when autofocus is disabled)
  if (!camera_auto_focus_) {
    // Focus absolute usually ranges from 0 to some max value (often 255 or 1023)
    // We'll map our 0-100% to a 0-255 range for most webcams
    int focus_v4l2 = (int)(camera_focus_ * 255.0 / 100.0);
    focus_v4l2 = std::max(0, std::min(255, focus_v4l2));
    std::string focus_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(current_camera_id_) + 
                            " --set-ctrl=focus_absolute=" + std::to_string(focus_v4l2);
    if (system(focus_cmd.c_str()) == 0) {
      std::cout << "  Manual Focus: " << camera_focus_ << "% (v4l2: " << focus_v4l2 << ")" << std::endl;
    }
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
  
  // Read manual focus value (only relevant when autofocus is off)
  int focus_v4l2 = getControlValue("focus_absolute");
  if (focus_v4l2 >= 0) {
    camera_focus_ = (float)(focus_v4l2 * 100.0 / 255.0);  // Convert back from v4l2 range to our 0-100%
    camera_focus_ = std::max(0.0f, std::min(100.0f, camera_focus_));
    std::cout << "  Manual Focus: " << camera_focus_ << "% (v4l2: " << focus_v4l2 << ")" << std::endl;
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
  blusher_filter_ = BlusherFilter::Create();      // Blusher effect using face landmarks
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
  
  // Manual focus slider (only when autofocus is disabled)
  if (!camera_auto_focus_) {
    if (ImGui::SliderFloat("Manual Focus", &camera_focus_, 0.0f, 100.0f, "%.0f%%")) {
      camera_settings_changed = true;
    }
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
    camera_focus_ = 50.0f;
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
  
  if (ImGui::Checkbox("Show Body Detection", &show_body_detection_)) {
    // Body detection visualization will be handled in rendering
  }

  ImGui::Separator();
  ImGui::Text("Background Effects:");
  
  // Background mode selection
  const char* background_modes[] = { "None", "Blur", "Custom Image" };
  int current_mode = static_cast<int>(background_mode_);
  
  if (ImGui::Combo("Background Mode", &current_mode, background_modes, 3)) {
    background_mode_ = static_cast<BackgroundMode>(current_mode);
  }
  
  // Blur strength slider (only show when Blur is selected)
  if (background_mode_ == BackgroundMode::BLUR) {
    if (ImGui::SliderFloat("Blur Strength", &background_blur_strength_, 5.0f, 51.0f, "%.0f")) {
      // Ensure odd kernel size for Gaussian blur
      int kernel_size = static_cast<int>(background_blur_strength_);
      if (kernel_size % 2 == 0) {
        background_blur_strength_ = static_cast<float>(kernel_size + 1);
      }
    }
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "(Higher = More Blur)");
  }
  
  // Custom image selection (only show when Custom Image is selected)
  if (background_mode_ == BackgroundMode::CUSTOM_IMAGE) {
    ImGui::Text("Custom Background Image:");
    
    // Display current image path
    if (!custom_background_path_.empty()) {
      ImGui::Text("Current: %s", custom_background_path_.c_str());
      if (background_image_loaded_) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✓ Image loaded successfully");
      } else {
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "✗ Failed to load image");
      }
    } else {
      ImGui::Text("No image selected");
    }
    
    // Browse button
    if (ImGui::Button("Browse Image...")) {
      // For now, show a text input - in a full implementation you'd use a file dialog
      ImGui::OpenPopup("Enter Image Path");
    }
    
    // Simple path input popup
    if (ImGui::BeginPopupModal("Enter Image Path", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      static char image_path_buffer[512] = "";
      
      ImGui::Text("Enter path to background image:");
      ImGui::InputText("Path", image_path_buffer, sizeof(image_path_buffer));
      
      if (ImGui::Button("Load")) {
        custom_background_path_ = std::string(image_path_buffer);
        
        // Try to load the image
        custom_background_image_ = cv::imread(custom_background_path_);
        background_image_loaded_ = !custom_background_image_.empty();
        
        if (background_image_loaded_) {
          std::cout << "Successfully loaded background image: " << custom_background_path_ << std::endl;
          std::cout << "Image size: " << custom_background_image_.cols << "x" << custom_background_image_.rows << std::endl;
        } else {
          std::cout << "Failed to load background image: " << custom_background_path_ << std::endl;
        }
        
        ImGui::CloseCurrentPopup();
      }
      
      ImGui::SameLine();
      if (ImGui::Button("Cancel")) {
        ImGui::CloseCurrentPopup();
      }
      
      ImGui::EndPopup();
    }
    
    // Clear button
    if (!custom_background_path_.empty()) {
      ImGui::SameLine();
      if (ImGui::Button("Clear")) {
        custom_background_path_.clear();
        custom_background_image_.release();
        background_image_loaded_ = false;
      }
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

  // AI Segmentation Controls
  ImGui::Separator();
  ImGui::Text("AI Segmentation Parameters:");
  
  if (ImGui::SliderFloat("AI Confidence Threshold", &ai_confidence_threshold_, 0.00f, 0.8f, "%.2f")) {
    if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
      mediapipe_segmentation->SetConfidenceThreshold(ai_confidence_threshold_);
      std::cout << "[AI] Updated confidence threshold to: " << ai_confidence_threshold_ << std::endl;
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("MediaPipe confidence: 0.3=sensitive, 0.5=balanced, 0.8=strict");
  }
  
  if (ImGui::SliderFloat("AI Temporal Smoothing", &ai_temporal_smoothing_, 0.00f, 0.9f, "%.2f")) {
    if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
      mediapipe_segmentation->SetTemporalSmoothing(true, ai_temporal_smoothing_);
      std::cout << "[AI] Updated temporal smoothing to: " << ai_temporal_smoothing_ << std::endl;
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Lower values = faster response to movement, higher values = smoother but slower response");
  }

  // RVM Enhanced Processing Controls
  ImGui::Separator();
  ImGui::Text("RVM Enhanced Processing:");
  
  if (ImGui::Checkbox("Enable RVM Processing", &enable_rvm_processing_)) {
    if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
      RVMProcessor::RVMConfig rvm_config;
      rvm_config.temporal_buffer_size = rvm_temporal_buffer_size_;
      rvm_config.temporal_weight = rvm_temporal_weight_;
      rvm_config.motion_threshold = rvm_motion_threshold_;
      rvm_config.enable_motion_compensation = rvm_motion_compensation_;
      rvm_config.enable_edge_refinement = rvm_edge_refinement_;
      
      mediapipe_segmentation->SetRVMProcessing(enable_rvm_processing_, rvm_config);
      std::cout << "[RVM] Enhanced processing: " << (enable_rvm_processing_ ? "enabled" : "disabled") << std::endl;
    }
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Enable RVM (Robust Video Matting) style temporal consistency and edge refinement");
  }
  
  if (enable_rvm_processing_) {
    if (ImGui::SliderInt("RVM Buffer Size", &rvm_temporal_buffer_size_, 3, 10)) {
      if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
        RVMProcessor::RVMConfig rvm_config;
        rvm_config.temporal_buffer_size = rvm_temporal_buffer_size_;
        rvm_config.temporal_weight = rvm_temporal_weight_;
        rvm_config.motion_threshold = rvm_motion_threshold_;
        rvm_config.enable_motion_compensation = rvm_motion_compensation_;
        rvm_config.enable_edge_refinement = rvm_edge_refinement_;
        mediapipe_segmentation->SetRVMProcessing(true, rvm_config);
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Number of frames used for temporal consistency (3-10)");
    }
    
    if (ImGui::SliderFloat("RVM Temporal Weight", &rvm_temporal_weight_, 0.1f, 0.8f, "%.2f")) {
      if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
        RVMProcessor::RVMConfig rvm_config;
        rvm_config.temporal_buffer_size = rvm_temporal_buffer_size_;
        rvm_config.temporal_weight = rvm_temporal_weight_;
        rvm_config.motion_threshold = rvm_motion_threshold_;
        rvm_config.enable_motion_compensation = rvm_motion_compensation_;
        rvm_config.enable_edge_refinement = rvm_edge_refinement_;
        mediapipe_segmentation->SetRVMProcessing(true, rvm_config);
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Strength of temporal smoothing (0.1=light, 0.8=strong)");
    }
    
    if (ImGui::SliderFloat("Motion Threshold", &rvm_motion_threshold_, 0.05f, 0.3f, "%.3f")) {
      if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
        RVMProcessor::RVMConfig rvm_config;
        rvm_config.temporal_buffer_size = rvm_temporal_buffer_size_;
        rvm_config.temporal_weight = rvm_temporal_weight_;
        rvm_config.motion_threshold = rvm_motion_threshold_;
        rvm_config.enable_motion_compensation = rvm_motion_compensation_;
        rvm_config.enable_edge_refinement = rvm_edge_refinement_;
        mediapipe_segmentation->SetRVMProcessing(true, rvm_config);
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Motion detection sensitivity (0.05=very sensitive, 0.3=less sensitive)");
    }
    
    if (ImGui::Checkbox("Motion Compensation", &rvm_motion_compensation_)) {
      if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
        RVMProcessor::RVMConfig rvm_config;
        rvm_config.temporal_buffer_size = rvm_temporal_buffer_size_;
        rvm_config.temporal_weight = rvm_temporal_weight_;
        rvm_config.motion_threshold = rvm_motion_threshold_;
        rvm_config.enable_motion_compensation = rvm_motion_compensation_;
        rvm_config.enable_edge_refinement = rvm_edge_refinement_;
        mediapipe_segmentation->SetRVMProcessing(true, rvm_config);
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Reduce temporal smoothing in high-motion areas");
    }
    
    if (ImGui::Checkbox("Edge Refinement", &rvm_edge_refinement_)) {
      if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
        RVMProcessor::RVMConfig rvm_config;
        rvm_config.temporal_buffer_size = rvm_temporal_buffer_size_;
        rvm_config.temporal_weight = rvm_temporal_weight_;
        rvm_config.motion_threshold = rvm_motion_threshold_;
        rvm_config.enable_motion_compensation = rvm_motion_compensation_;
        rvm_config.enable_edge_refinement = rvm_edge_refinement_;
        mediapipe_segmentation->SetRVMProcessing(true, rvm_config);
      }
    }
    if (ImGui::IsItemHovered()) {
      ImGui::SetTooltip("Enhanced edge quality using guided filtering");
    }
  }

  // Background Detection Configuration
  ImGui::Separator();
  ImGui::Text("Background Detection Configuration:");
  
  const char* target_class_items[] = { "Auto Detect", "Person Focus", "Background Focus" };
  if (ImGui::Combo("Target Class", &target_class_selection_, target_class_items, IM_ARRAYSIZE(target_class_items))) {
    std::cout << "[AI] Target class changed to: " << target_class_items[target_class_selection_] << std::endl;
    // Note: Configuration will be applied on next initialization
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Auto Detect: Standard segmentation\nPerson Focus: Optimize for person detection\nBackground Focus: Optimize for background detection");
  }
  
  if (ImGui::Checkbox("Output Confidence Masks", &output_confidence_masks_)) {
    std::cout << "[AI] Confidence masks: " << (output_confidence_masks_ ? "enabled" : "disabled") << std::endl;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Enable confidence-based mask output for fine-grained control");
  }
  
  if (ImGui::Checkbox("Output Category Masks", &output_category_mask_)) {
    std::cout << "[AI] Category masks: " << (output_category_mask_ ? "enabled" : "disabled") << std::endl;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("Enable categorical mask output for class-specific processing");
  }

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

// Load and use MediaPipe AI segmentation for real-time streaming
std::vector<unsigned char> ApplyAISegmentation(const unsigned char* rgba_data, int width, int height) {
  if (!rgba_data || width <= 0 || height <= 0) {
    return std::vector<unsigned char>(rgba_data, rgba_data + (width * height * 4));
  }
  
  static bool segmentation_initialized = false;
  static cv::Mat cached_mask;
  static int frame_counter = 0;
  static auto last_segmentation_time = std::chrono::high_resolution_clock::now();
  frame_counter++;
  
  // Initialize MediaPipe segmentation once
  if (!segmentation_initialized) {
    mediapipe_segmentation = std::make_unique<MediaPipeSegmentation>();
    
    // Create MediaPipe configuration with new background detection options
    MediaPipeConfig config;
    config.running_mode = MediaPipeConfig::LIVE_STREAM;
    
    // Map UI selection to target class
    switch (target_class_selection_) {
      case 0: config.target_class = MediaPipeConfig::AUTO_DETECT; break;
      case 1: config.target_class = MediaPipeConfig::PERSON; break;
      case 2: config.target_class = MediaPipeConfig::BACKGROUND; break;
      default: config.target_class = MediaPipeConfig::AUTO_DETECT; break;
    }
    
    config.output_confidence_masks = output_confidence_masks_;
    config.output_category_mask = output_category_mask_;
    config.confidence_threshold = ai_confidence_threshold_;
    config.temporal_smoothing = true;
    
    // Test different models with OpenCV 4.12.0 (prioritize MobileNet-v2 for compatibility)
    std::vector<std::string> test_models = {
        "/home/padletut/gpupixel/models/mobilenet_v2.onnx",                           // MobileNet-v2: Good OpenCV compatibility
        "/home/padletut/gpupixel/models/modnet.onnx",                                 // MODNet: High-quality portrait matting
        "/home/padletut/gpupixel/models/mediapipe_landscape_segmentation.onnx",
        "/home/padletut/gpupixel/models/mediapipe_selfie_segmentation.onnx",
        "/home/padletut/gpupixel/models/selfie_multiclass.onnx"
    };
    
    bool model_loaded = false;
    for (const auto& model_path : test_models) {
        std::cout << "[TEST] Testing model: " << model_path << std::endl;
        if (mediapipe_segmentation->Initialize(model_path, config)) {
            std::cout << "[SUCCESS] Model loaded with background detection config: " << model_path << std::endl;
            const char* target_names[] = { "AUTO_DETECT", "PERSON", "BACKGROUND" };
            std::cout << "[CONFIG] Target Class: " << target_names[target_class_selection_] << std::endl;
            std::cout << "[CONFIG] Confidence Masks: " << (output_confidence_masks_ ? "enabled" : "disabled") << std::endl;
            std::cout << "[CONFIG] Category Masks: " << (output_category_mask_ ? "enabled" : "disabled") << std::endl;
            model_loaded = true;
            break;
        } else {
            std::cout << "[FAILED] Model failed to load: " << model_path << std::endl;
        }
    }
    
    if (model_loaded) {
      // Initialize with current UI values
      mediapipe_segmentation->SetConfidenceThreshold(ai_confidence_threshold_);
      mediapipe_segmentation->SetTemporalSmoothing(true, ai_temporal_smoothing_);
      segmentation_initialized = true;
      std::cout << "[AI] MediaPipe C++ Segmentation initialized for real-time streaming" << std::endl;
    } else {
      std::cout << "[AI] Failed to initialize MediaPipe segmentation" << std::endl;
      mediapipe_segmentation.reset();
    }
  }
  
  // Check if background effects are enabled
  bool segmentation_enabled = (background_mode_ != BackgroundMode::NONE);
  
  // Get raw camera input for MediaPipe (unfiltered for better detection)
  cv::Mat raw_camera_frame;
  cv::Mat bgr_mat;
  if (source_camera_ && segmentation_enabled) {
    raw_camera_frame = source_camera_->GetLatestFrame();
    if (!raw_camera_frame.empty()) {
      // Convert raw camera BGR to our expected format
      if (raw_camera_frame.channels() == 3) {
        bgr_mat = raw_camera_frame.clone();
      } else if (raw_camera_frame.channels() == 4) {
        cv::cvtColor(raw_camera_frame, bgr_mat, cv::COLOR_BGRA2BGR);
      }
      // Flip to match display orientation
      cv::flip(bgr_mat, bgr_mat, 0);
    }
  }
  
  // Fallback: Convert processed RGBA to BGR for visualization (when raw camera unavailable)
  if (bgr_mat.empty()) {
    cv::Mat rgba_mat(height, width, CV_8UC4, const_cast<unsigned char*>(rgba_data));
    cv::Mat flipped_rgba;
    cv::flip(rgba_mat, flipped_rgba, 0);  // Flip for OpenGL coordinate system
    cv::cvtColor(flipped_rgba, bgr_mat, cv::COLOR_RGBA2BGR);
  }
  
  cv::Mat person_mask = cv::Mat::zeros(bgr_mat.size(), CV_8UC1);
  
  // Only run segmentation if background effects are enabled and MediaPipe is ready
  if (segmentation_enabled && segmentation_initialized && mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
    // Smart timing for real-time streaming: update every 3-5 frames (12-20 FPS at 60 FPS input)
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_segmentation_time);
    
    bool should_run_segmentation = (frame_counter % 4 == 0) && (time_since_last.count() >= 50);  // Max 20 FPS segmentation
    
    if (should_run_segmentation) {
      try {
        // Use new advanced processing with background detection configuration
        SegmentationResult result = mediapipe_segmentation->ProcessFrameAdvanced(bgr_mat);
        
        if (!result.confidence_mask.empty()) {
          // Successfully got new segmentation with enhanced configuration
          cv::Mat current_mask;
          
          // Ensure the confidence mask is in the right format [0,1] float
          if (result.confidence_mask.type() != CV_32F) {
            result.confidence_mask.convertTo(current_mask, CV_32F, 1.0/255.0);
          } else {
            result.confidence_mask.copyTo(current_mask);
          }
          
          // Apply target class selection
          if (target_class_selection_ == 2) {  // Background Focus
            // For background focus, invert the mask (low confidence = background)
            cv::Mat background_mask;
            cv::threshold(current_mask, background_mask, ai_confidence_threshold_, 1.0, cv::THRESH_BINARY_INV);
            background_mask.copyTo(current_mask);
            std::cout << "[DEBUG] Background focus mode: inverted mask" << std::endl;
          } else {
            // For person focus or auto detect, apply threshold to confidence mask
            cv::threshold(current_mask, current_mask, ai_confidence_threshold_, 1.0, cv::THRESH_BINARY);
            std::cout << "[DEBUG] Person/Auto focus mode: thresholded mask" << std::endl;
          }
          
          current_mask.copyTo(person_mask);
          current_mask.copyTo(cached_mask);
          last_segmentation_time = current_time;
          
          // Store for visualization (flip back to match display orientation)
          cv::Mat display_mask;
          cv::flip(person_mask, display_mask, 0);
          
          // Convert to proper format for visualization (CV_8UC1 [0,255])
          cv::Mat viz_mask;
          if (display_mask.type() == CV_32F) {
            display_mask.convertTo(viz_mask, CV_8UC1, 255.0);
          } else {
            display_mask.copyTo(viz_mask);
          }
          viz_mask.copyTo(last_person_mask_);
          
          // Debug output every second
          if (frame_counter % 60 == 0) {
            double coverage = cv::sum(person_mask)[0] / (person_mask.type() == CV_32F ? 
                                                        (1.0 * bgr_mat.rows * bgr_mat.cols) : 
                                                        (255.0 * bgr_mat.rows * bgr_mat.cols));
            std::cout << "[AI] MediaPipe Real-time: " << std::fixed << std::setprecision(1) 
                      << coverage * 100.0 << "% person detected (streaming)" << std::endl;
          }
        } else {
          // MediaPipe failed, use cached mask if available
          if (!cached_mask.empty() && cached_mask.size() == bgr_mat.size()) {
            cached_mask.copyTo(person_mask);
          } else {
            // No cached mask, fill as person for continuity
            person_mask.setTo(255);
          }
        }
      } catch (const std::exception& e) {
        std::cout << "[AI] MediaPipe processing error: " << e.what() << std::endl;
        // Use cached mask on error
        if (!cached_mask.empty() && cached_mask.size() == bgr_mat.size()) {
          cached_mask.copyTo(person_mask);
        } else {
          person_mask.setTo(255);
        }
      }
    } else {
      // Use cached mask for smooth streaming between updates
      if (!cached_mask.empty() && cached_mask.size() == bgr_mat.size()) {
        cached_mask.copyTo(person_mask);
      } else {
        // No cached mask available, assume person for continuity
        person_mask.setTo(255);
      }
    }
  } else if (segmentation_enabled) {
    // MediaPipe not available but effects enabled - use simple fallback
    // For streaming, better to assume person presence than heavy CV processing
    person_mask.setTo(255);  
  } else {
    // No background effects - assume full person
    person_mask.setTo(255);
  }
  
  // Apply background effects based on current mode
  cv::Mat processed_frame;
  
  // Convert beauty-filtered RGBA input to BGR for composition
  cv::Mat beauty_rgba_mat(height, width, CV_8UC4, const_cast<unsigned char*>(rgba_data));
  cv::Mat beauty_flipped_rgba;
  cv::flip(beauty_rgba_mat, beauty_flipped_rgba, 0);  // Flip for OpenGL coordinate system
  cv::Mat beauty_bgr_mat;
  cv::cvtColor(beauty_flipped_rgba, beauty_bgr_mat, cv::COLOR_RGBA2BGR);
  
  if (background_mode_ == BackgroundMode::BLUR) {
    // Apply background blur to beauty-filtered content
    cv::Mat blurred_frame;
    int kernel_size = static_cast<int>(background_blur_strength_);
    if (kernel_size % 2 == 0) kernel_size++;
    kernel_size = std::max(5, std::min(51, kernel_size));
    cv::GaussianBlur(beauty_bgr_mat, blurred_frame, cv::Size(kernel_size, kernel_size), 0);
    
    // Blend person (beauty-filtered) with background (blurred beauty-filtered)
    processed_frame = cv::Mat::zeros(beauty_bgr_mat.size(), CV_8UC3);
    
    // Handle different mask types (CV_32F [0,1] or CV_8UC1 [0,255])
    bool is_float_mask = (person_mask.type() == CV_32F);
    
    for (int y = 0; y < beauty_bgr_mat.rows; y++) {
      for (int x = 0; x < beauty_bgr_mat.cols; x++) {
        float mask_val;
        if (is_float_mask) {
          mask_val = person_mask.at<float>(y, x);  // Already [0,1]
        } else {
          mask_val = person_mask.at<unsigned char>(y, x) / 255.0f;  // Convert [0,255] to [0,1]
        }
        
        cv::Vec3b original = beauty_bgr_mat.at<cv::Vec3b>(y, x);
        cv::Vec3b blurred = blurred_frame.at<cv::Vec3b>(y, x);
        
        for (int c = 0; c < 3; c++) {
          processed_frame.at<cv::Vec3b>(y, x)[c] = 
            cv::saturate_cast<unsigned char>(mask_val * original[c] + (1.0f - mask_val) * blurred[c]);
        }
      }
    }
  } else {
    // No background effect or other modes - return beauty-filtered original
    processed_frame = beauty_bgr_mat.clone();
  }
  
  // Convert back to RGBA
  cv::Mat processed_rgba;
  cv::cvtColor(processed_frame, processed_rgba, cv::COLOR_BGR2RGBA);
  cv::Mat final_rgba;
  cv::flip(processed_rgba, final_rgba, 0);  // Flip back for OpenGL
  
  std::vector<unsigned char> result(final_rgba.total() * final_rgba.elemSize());
  std::memcpy(result.data(), final_rgba.data, result.size());
  
  return result;
}

// Apply custom background replacement using MediaPipe segmentation
std::vector<unsigned char> ApplyCustomBackground(const unsigned char* rgba_data, int width, int height) {
  if (!rgba_data || width <= 0 || height <= 0 || !background_image_loaded_) {
    // Return original if invalid data or no background loaded
    std::vector<unsigned char> result(rgba_data, rgba_data + (width * height * 4));
    return result;
  }

  
  // Convert RGBA to BGR for OpenCV processing
  cv::Mat rgba_mat(height, width, CV_8UC4, const_cast<unsigned char*>(rgba_data));
  cv::Mat flipped_rgba;
  cv::flip(rgba_mat, flipped_rgba, 0);  // Flip for OpenGL coordinate system
  
  cv::Mat bgr_mat;
  cv::cvtColor(flipped_rgba, bgr_mat, cv::COLOR_RGBA2BGR);
  
  // Generate person mask using AI segmentation or advanced CV fallback
  static cv::Mat cached_mask;
  static int frame_skip_counter = 0;
  
  cv::Mat person_mask;
  
  if (frame_skip_counter % 4 == 0 || cached_mask.empty()) {
    // Use MediaPipe segmentation (primary method) or fallback to GrabCut
    bool segmentation_success = false;
    
    // Try MediaPipe segmentation first if available
    if (mediapipe_segmentation && mediapipe_segmentation->IsReady()) {
      // Use new advanced processing with background detection configuration
      SegmentationResult result = mediapipe_segmentation->ProcessFrameAdvanced(bgr_mat);
      
      if (!result.confidence_mask.empty()) {
        // Ensure the confidence mask is in the right format [0,1] float  
        cv::Mat temp_mask;
        if (result.confidence_mask.type() != CV_32F) {
          result.confidence_mask.convertTo(temp_mask, CV_32F, 1.0/255.0);
        } else {
          result.confidence_mask.copyTo(temp_mask);
        }
        
        // Apply target class selection for body detection
        if (target_class_selection_ == 2) {  // Background Focus
          // For background focus, invert the mask (low confidence = background)
          cv::threshold(temp_mask, person_mask, ai_confidence_threshold_, 1.0, cv::THRESH_BINARY_INV);
        } else {
          // For person focus or auto detect, apply threshold to confidence mask
          cv::threshold(temp_mask, person_mask, ai_confidence_threshold_, 1.0, cv::THRESH_BINARY);
        }
        segmentation_success = true;
      }
    }
    
    if (!segmentation_success) {
      // Advanced fallback using GrabCut
      cv::Mat small_frame;
      cv::resize(bgr_mat, small_frame, cv::Size(256, 256));
      
      cv::Mat grabcut_mask = cv::Mat::zeros(256, 256, CV_8UC1);
      cv::Rect person_rect(64, 32, 128, 192);
      cv::Mat bgd_model, fgd_model;
      
      try {
        cv::grabCut(small_frame, grabcut_mask, person_rect, bgd_model, fgd_model, 2, cv::GC_INIT_WITH_RECT);
        
        cv::Mat result_mask = cv::Mat::zeros(256, 256, CV_8UC1);
        for (int y = 0; y < 256; y++) {
          for (int x = 0; x < 256; x++) {
            if (grabcut_mask.at<uint8_t>(y, x) == cv::GC_FGD || grabcut_mask.at<uint8_t>(y, x) == cv::GC_PR_FGD) {
              result_mask.at<uint8_t>(y, x) = 255;
            }
          }
        }
        
        cv::resize(result_mask, person_mask, bgr_mat.size());
        
      } catch (const cv::Exception& e) {
        // Final fallback
        person_mask = cv::Mat::zeros(bgr_mat.size(), CV_8UC1);
        cv::Point center(bgr_mat.cols/2, bgr_mat.rows/2);
        cv::Size axes(bgr_mat.cols/4, bgr_mat.rows/3);
        cv::ellipse(person_mask, center, axes, 0, 0, 360, cv::Scalar(255), -1);
      }
    }
    
    cached_mask = person_mask.clone();
  } else {
    if (!cached_mask.empty() && cached_mask.size() == bgr_mat.size()) {
      cached_mask.copyTo(person_mask);
    }
  }
  frame_skip_counter++;
  
  // Prepare custom background
  cv::Mat background_resized;
  cv::resize(custom_background_image_, background_resized, cv::Size(width, height));
  
  // Blend person (original) with custom background
  cv::Mat processed_frame = cv::Mat::zeros(bgr_mat.size(), CV_8UC3);
  for (int y = 0; y < bgr_mat.rows; y++) {
    for (int x = 0; x < bgr_mat.cols; x++) {
      float mask_val = person_mask.at<unsigned char>(y, x) / 255.0f;
      cv::Vec3b original = bgr_mat.at<cv::Vec3b>(y, x);
      cv::Vec3b background = background_resized.at<cv::Vec3b>(y, x);
      
      for (int c = 0; c < 3; c++) {
        processed_frame.at<cv::Vec3b>(y, x)[c] = 
          cv::saturate_cast<unsigned char>(mask_val * original[c] + (1.0f - mask_val) * background[c]);
      }
    }
  }
  
  // Convert back to RGBA
  cv::Mat processed_rgba;
  cv::cvtColor(processed_frame, processed_rgba, cv::COLOR_BGR2RGBA);
  cv::Mat final_rgba;
  cv::flip(processed_rgba, final_rgba, 0);  // Flip back for OpenGL
  
  std::vector<unsigned char> result(final_rgba.total() * final_rgba.elemSize());
  std::memcpy(result.data(), final_rgba.data, result.size());
  
  return result;
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

      // Apply MediaPipe background effects first, then face detection overlay
      std::vector<unsigned char> final_buffer;
      
      // Apply background effect based on selected mode
      switch (background_mode_) {
        case BackgroundMode::NONE:
          // No background effect - use original frame
          final_buffer.assign(buffer, buffer + (width * height * 4));
          break;
          
        case BackgroundMode::BLUR:
          // Apply AI-based blur segmentation
          final_buffer = ApplyAISegmentation(buffer, width, height);
          break;
          
        case BackgroundMode::CUSTOM_IMAGE:
          // Apply custom background replacement
          if (background_image_loaded_) {
            final_buffer = ApplyCustomBackground(buffer, width, height);
          } else {
            // Fallback to original if no custom image loaded
            final_buffer.assign(buffer, buffer + (width * height * 4));
          }
          break;
          
        default:
          final_buffer.assign(buffer, buffer + (width * height * 4));
          break;
      }
      
      // Add overlays if enabled
      bool has_overlays = (show_face_detection_ && opencv_face_detector_) || (show_body_detection_ && !last_person_mask_.empty());
      
      if (has_overlays) {
        cv::Mat debug_frame(height, width, CV_8UC4, final_buffer.data());
        
        // Add face detection overlay if enabled
        if (show_face_detection_ && opencv_face_detector_) {
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
        }        // Add body detection overlay if enabled
        if (show_body_detection_ && !last_person_mask_.empty()) {
          // Scale the stored mask to current frame size if needed
          cv::Mat display_mask;
          if (last_person_mask_.size() != cv::Size(width, height)) {
            cv::resize(last_person_mask_, display_mask, cv::Size(width, height));
          } else {
            display_mask = last_person_mask_;
          }
          
          // Create efficient colored overlay using OpenCV operations
          cv::Mat mask_3channel;
          cv::cvtColor(display_mask, mask_3channel, cv::COLOR_GRAY2BGR);
          
          // Create green overlay only where mask > 128
          cv::Mat green_overlay = cv::Mat::zeros(height, width, CV_8UC3);
          green_overlay.setTo(cv::Scalar(0, 255, 0), display_mask > 128);
          
          // Convert debug_frame to 3 channel for blending
          cv::Mat debug_3channel;
          cv::cvtColor(debug_frame, debug_3channel, cv::COLOR_BGRA2BGR);
          
          // Efficient alpha blending: frame * 0.7 + overlay * 0.3 where mask > 128
          cv::Mat blended;
          cv::addWeighted(debug_3channel, 0.7, green_overlay, 0.3, 0, blended, CV_8UC3);
          
          // Copy blended result back only where mask is active
          cv::Mat mask_bool = display_mask > 128;
          blended.copyTo(debug_3channel, mask_bool);
          
          // Convert back to BGRA
          cv::cvtColor(debug_3channel, debug_frame, cv::COLOR_BGR2BGRA);
        }
        
        RenderRGBAToScreen(final_buffer.data(), width, height);
      } else {
        // No overlays, just render with background effects
        RenderRGBAToScreen(final_buffer.data(), width, height);
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
                       camera_auto_focus_, camera_focus_, camera_auto_gain_,
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
    camera_focus_ = profile.camera_focus;
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
           << profile.camera_focus << "|"
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
        if (std::getline(iss, token, '|')) profile.camera_focus = std::stof(token);  // Optional field for backward compatibility
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
