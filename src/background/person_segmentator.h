#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <string>

class PersonSegmentator {
public:
    PersonSegmentator();
    ~PersonSegmentator();
    
    bool Initialize(const std::string& model_path = "");
    cv::Mat SegmentPerson(const cv::Mat& input_frame);
    bool IsInitialized() const { return initialized_; }
    
    // Settings
    void SetConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }
    void SetInputSize(int width, int height) { input_width_ = width; input_height_ = height; }
    
private:
    bool initialized_;
    cv::dnn::Net net_;
    float confidence_threshold_;
    int input_width_;
    int input_height_;
    
    // Model configuration
    std::vector<std::string> output_names_;
    
    // Helper methods
    cv::Mat PreprocessFrame(const cv::Mat& frame);
    cv::Mat PostprocessMask(const cv::Mat& output, const cv::Size& original_size);
};
