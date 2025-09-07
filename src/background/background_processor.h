#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include "person_segmentator.h"

enum class BackgroundMode {
    NONE = 0,
    BLUR = 1,
    CUSTOM_IMAGE = 2
};

class BackgroundProcessor {
public:
    BackgroundProcessor();
    ~BackgroundProcessor();
    
    bool Initialize();
    cv::Mat ProcessFrame(const cv::Mat& input_frame);
    
    // Background mode control
    void SetBackgroundMode(BackgroundMode mode) { background_mode_ = mode; }
    BackgroundMode GetBackgroundMode() const { return background_mode_; }
    
    // Blur settings
    void SetBlurStrength(float strength) { blur_strength_ = strength; }
    float GetBlurStrength() const { return blur_strength_; }
    
    // Custom background
    bool LoadCustomBackground(const std::string& image_path);
    void ClearCustomBackground();
    bool HasCustomBackground() const { return !custom_background_.empty(); }
    
    // Segmentation settings
    void SetSegmentationSmoothness(float smoothness) { segmentation_smoothness_ = smoothness; }
    
private:
    std::unique_ptr<PersonSegmentator> segmentator_;
    BackgroundMode background_mode_;
    
    // Blur effect
    float blur_strength_;
    
    // Custom background
    cv::Mat custom_background_;
    cv::Mat scaled_background_;
    
    // Segmentation
    float segmentation_smoothness_;
    
    // Helper methods
    cv::Mat ApplyBlurBackground(const cv::Mat& frame, const cv::Mat& mask);
    cv::Mat ApplyCustomBackground(const cv::Mat& frame, const cv::Mat& mask);
    cv::Mat ScaleBackgroundToFrame(const cv::Mat& background, const cv::Size& frame_size);
    cv::Mat SmoothMask(const cv::Mat& mask, float smoothness);
};
