/*
 * GPUPixel OpenCV Face Detector
 * 
 * Alternative face detector using OpenCV instead of mars-face-kit
 * for Linux compatibility
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <memory>

namespace gpupixel {

class OpenCVFaceDetector {
public:
    OpenCVFaceDetector();
    ~OpenCVFaceDetector();
    
    bool Init();
    
    // Detect faces and return landmarks in GPUPixel format
    // Returns normalized coordinates (-1 to 1) as expected by GPUPixel filters
    std::vector<float> DetectFace(const cv::Mat& frame);
    
    // Enable/disable debug visualization
    void SetDebugMode(bool enabled);
    
    // Get last detected faces and eyes for debug visualization
    std::vector<cv::Rect> GetLastDetectedFaces() const { return last_detected_faces_; }
    std::vector<cv::Rect> GetLastDetectedEyes() const { return last_detected_eyes_; }
    
private:
    cv::CascadeClassifier face_cascade_;
    cv::CascadeClassifier eye_cascade_;
    bool initialized_;
    bool debug_mode_;
    
    // Store last detected faces and eyes for debug rendering
    std::vector<cv::Rect> last_detected_faces_;
    std::vector<cv::Rect> last_detected_eyes_;
    
    // Convert OpenCV face rectangle to GPUPixel landmark format
    std::vector<float> GenerateLandmarksFromFace(const cv::Rect& face, 
                                               const std::vector<cv::Rect>& eyes,
                                               int frame_width, int frame_height);
};

} // namespace gpupixel
