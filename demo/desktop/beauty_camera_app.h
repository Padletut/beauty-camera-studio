#pragma once

#include "camera_manager.h"
#include "beauty_profile_manager.h"
#include "filter_pipeline.h"
#include "virtual_camera.h"
#include "gpupixel/utils/mediapipe_segmentation.h"
#include "gpupixel/utils/rvm_processor.h"
#include "face_detector/opencv_face_detector.h"
#include <GLFW/glfw3.h>
#include <memory>

namespace gpupixel {

class BeautyCameraApp {
public:
    BeautyCameraApp();
    ~BeautyCameraApp();
    
    // Application lifecycle
    bool Initialize();
    void Run();
    void Shutdown();
    
    // Main loop
    void Update();
    void Render();
    
private:
    // Core components
    std::unique_ptr<CameraManager> camera_manager_;
    std::unique_ptr<BeautyProfileManager> profile_manager_;
    std::unique_ptr<FilterPipeline> filter_pipeline_;
    std::unique_ptr<VirtualCamera> virtual_camera_;
    
    // AI/Detection components
    std::unique_ptr<MediaPipeSegmentation> mediapipe_segmentation_;
    std::unique_ptr<RVMProcessor> rvm_processor_;
    std::shared_ptr<OpenCVFaceDetector> face_detector_;
    
    // Window and rendering
    GLFWwindow* main_window_;
    bool initialized_;
    bool should_close_;
    
    // Current state
    FilterParameters current_filter_params_;
    std::string current_profile_name_;
    bool show_face_detection_;
    bool show_body_detection_;
    bool enable_rvm_processing_;
    
    // UI state
    bool show_profile_save_popup_;
    bool show_profile_manager_;
    char profile_name_buffer_[256];
    
    // Initialization helpers
    bool SetupWindow();
    bool SetupImGui();
    bool SetupComponents();
    
    // Update helpers
    void UpdateCameraSettings();
    void UpdateFilterParameters();
    void UpdateAIProcessing();
    void ProcessFrame();
    
    // UI rendering
    void RenderMainUI();
    void RenderBeautyControls();
    void RenderCameraControls();
    void RenderAIControls();
    void RenderProfileManager();
    void RenderProfileSavePopup();
    
    // Event handlers
    void OnProfileChanged();
    void OnCameraChanged();
    
    // Utility
    std::string GetResourcePath();
};

} // namespace gpupixel
