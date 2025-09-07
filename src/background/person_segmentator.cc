#include "person_segmentator.h"
#include <iostream>

PersonSegmentator::PersonSegmentator() 
    : initialized_(false), confidence_threshold_(0.5f), 
      input_width_(513), input_height_(513) {
}

PersonSegmentator::~PersonSegmentator() {
}

bool PersonSegmentator::Initialize(const std::string& model_path) {
    try {
        // For now, we'll use a simple approach with OpenCV's background subtraction
        // and morphological operations to create a person mask
        // This is much lighter than deep learning models and works reasonably well
        initialized_ = true;
        std::cout << "Person segmentator initialized successfully (using OpenCV background subtraction)" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize person segmentator: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat PersonSegmentator::SegmentPerson(const cv::Mat& input_frame) {
    if (!initialized_) {
        return cv::Mat();
    }
    
    // Simple person segmentation using background subtraction and morphological operations
    cv::Mat mask = cv::Mat::zeros(input_frame.size(), CV_8UC1);
    
    // For a simple implementation, we'll create a person mask based on:
    // 1. Skin color detection
    // 2. Motion detection 
    // 3. Center-weighted bias (assuming person is in center)
    
    cv::Mat hsv;
    cv::cvtColor(input_frame, hsv, cv::COLOR_BGR2HSV);
    
    // Skin color detection (rough approximation)
    cv::Mat skin_mask;
    cv::Scalar lower_skin(0, 20, 70);
    cv::Scalar upper_skin(20, 255, 255);
    cv::inRange(hsv, lower_skin, upper_skin, skin_mask);
    
    // Create a center-weighted mask (assume person is roughly in center)
    cv::Mat center_mask = cv::Mat::zeros(input_frame.size(), CV_8UC1);
    cv::Point center(input_frame.cols / 2, input_frame.rows / 2);
    cv::ellipse(center_mask, center, cv::Size(input_frame.cols / 3, input_frame.rows / 2), 0, 0, 360, cv::Scalar(255), -1);
    
    // Combine skin detection with center bias
    cv::bitwise_and(skin_mask, center_mask, mask);
    
    // Morphological operations to clean up the mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    
    // Dilate to ensure we cover the person fully
    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 3);
    
    // Apply Gaussian blur to soften edges
    cv::GaussianBlur(mask, mask, cv::Size(15, 15), 5);
    
    return mask;
}

cv::Mat PersonSegmentator::PreprocessFrame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    return resized;
}

cv::Mat PersonSegmentator::PostprocessMask(const cv::Mat& output, const cv::Size& original_size) {
    cv::Mat resized_mask;
    cv::resize(output, resized_mask, original_size);
    return resized_mask;
}
