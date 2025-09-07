#include "background_processor.h"
#include <iostream>

BackgroundProcessor::BackgroundProcessor() 
    : background_mode_(BackgroundMode::NONE), blur_strength_(15.0f), 
      segmentation_smoothness_(3.0f) {
    segmentator_ = std::make_unique<PersonSegmentator>();
}

BackgroundProcessor::~BackgroundProcessor() {
}

bool BackgroundProcessor::Initialize() {
    if (!segmentator_->Initialize()) {
        std::cerr << "Failed to initialize person segmentator" << std::endl;
        return false;
    }
    
    std::cout << "Background processor initialized successfully" << std::endl;
    return true;
}

cv::Mat BackgroundProcessor::ProcessFrame(const cv::Mat& input_frame) {
    if (background_mode_ == BackgroundMode::NONE) {
        return input_frame;  // No processing needed
    }
    
    // Get person segmentation mask
    cv::Mat person_mask = segmentator_->SegmentPerson(input_frame);
    if (person_mask.empty()) {
        return input_frame;  // Fallback to original frame
    }
    
    // Smooth the mask for better blending
    cv::Mat smooth_mask = SmoothMask(person_mask, segmentation_smoothness_);
    
    // Apply the selected background effect
    switch (background_mode_) {
        case BackgroundMode::BLUR:
            return ApplyBlurBackground(input_frame, smooth_mask);
        case BackgroundMode::CUSTOM_IMAGE:
            return ApplyCustomBackground(input_frame, smooth_mask);
        default:
            return input_frame;
    }
}

cv::Mat BackgroundProcessor::ApplyBlurBackground(const cv::Mat& frame, const cv::Mat& mask) {
    // Create blurred version of the entire frame
    cv::Mat blurred_frame;
    int kernel_size = static_cast<int>(blur_strength_);
    if (kernel_size % 2 == 0) kernel_size++; // Ensure odd kernel size
    kernel_size = std::max(3, std::min(51, kernel_size)); // Clamp to reasonable range
    
    cv::GaussianBlur(frame, blurred_frame, cv::Size(kernel_size, kernel_size), 0);
    
    // DEBUG: Check color values before blending
    static bool debug_printed = false;
    if (!debug_printed && frame.rows > frame.rows/2 && frame.cols > frame.cols/2) {
        cv::Vec3b orig_px = frame.at<cv::Vec3b>(frame.rows/2, frame.cols/2);
        cv::Vec3b blur_px = blurred_frame.at<cv::Vec3b>(frame.rows/2, frame.cols/2);
        std::cout << "[BLUR DEBUG] Original frame BGR: B=" << (int)orig_px[0] << " G=" << (int)orig_px[1] << " R=" << (int)orig_px[2] << std::endl;
        std::cout << "[BLUR DEBUG] Blurred frame BGR: B=" << (int)blur_px[0] << " G=" << (int)blur_px[1] << " R=" << (int)blur_px[2] << std::endl;
    }
    
    // Blend person (original) with blurred background
    cv::Mat result;
    cv::Mat mask_3ch;
    cv::cvtColor(mask, mask_3ch, cv::COLOR_GRAY2BGR);
    mask_3ch.convertTo(mask_3ch, CV_32F, 1.0/255.0);
    
    frame.convertTo(result, CV_32F);
    blurred_frame.convertTo(blurred_frame, CV_32F);
    
    // Person = mask * original + (1-mask) * blurred
    cv::multiply(mask_3ch, result, result);
    cv::Mat inv_mask = 1.0 - mask_3ch;
    cv::Mat blurred_bg;
    cv::multiply(inv_mask, blurred_frame, blurred_bg);
    result += blurred_bg;
    
    result.convertTo(result, CV_8U);
    
    // DEBUG: Check final result
    if (!debug_printed && result.rows > result.rows/2 && result.cols > result.cols/2) {
        cv::Vec3b final_px = result.at<cv::Vec3b>(result.rows/2, result.cols/2);
        std::cout << "[BLUR DEBUG] Final result BGR: B=" << (int)final_px[0] << " G=" << (int)final_px[1] << " R=" << (int)final_px[2] << std::endl;
        debug_printed = true;
    }
    
    return result;
}

cv::Mat BackgroundProcessor::ApplyCustomBackground(const cv::Mat& frame, const cv::Mat& mask) {
    if (custom_background_.empty()) {
        return frame;  // No custom background loaded
    }
    
    // Scale background to match frame size if needed
    cv::Mat background = ScaleBackgroundToFrame(custom_background_, frame.size());
    
    // Blend person (original) with custom background
    cv::Mat result;
    cv::Mat mask_3ch;
    cv::cvtColor(mask, mask_3ch, cv::COLOR_GRAY2BGR);
    mask_3ch.convertTo(mask_3ch, CV_32F, 1.0/255.0);
    
    frame.convertTo(result, CV_32F);
    background.convertTo(background, CV_32F);
    
    // Person = mask * original + (1-mask) * background
    cv::multiply(mask_3ch, result, result);
    cv::Mat inv_mask = 1.0 - mask_3ch;
    cv::Mat custom_bg;
    cv::multiply(inv_mask, background, custom_bg);
    result += custom_bg;
    
    result.convertTo(result, CV_8U);
    return result;
}

cv::Mat BackgroundProcessor::ScaleBackgroundToFrame(const cv::Mat& background, const cv::Size& frame_size) {
    if (background.size() == frame_size) {
        return background;
    }
    
    cv::Mat scaled;
    cv::resize(background, scaled, frame_size);
    return scaled;
}

cv::Mat BackgroundProcessor::SmoothMask(const cv::Mat& mask, float smoothness) {
    if (smoothness <= 0) {
        return mask;
    }
    
    cv::Mat smooth_mask;
    int kernel_size = static_cast<int>(smoothness * 2) + 1;
    cv::GaussianBlur(mask, smooth_mask, cv::Size(kernel_size, kernel_size), smoothness);
    return smooth_mask;
}

bool BackgroundProcessor::LoadCustomBackground(const std::string& image_path) {
    custom_background_ = cv::imread(image_path, cv::IMREAD_COLOR);
    if (custom_background_.empty()) {
        std::cerr << "Failed to load custom background image: " << image_path << std::endl;
        return false;
    }
    
    std::cout << "Custom background loaded: " << image_path << std::endl;
    return true;
}

void BackgroundProcessor::ClearCustomBackground() {
    custom_background_.release();
    scaled_background_.release();
}
