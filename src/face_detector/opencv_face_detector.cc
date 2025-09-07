/*
 * GPUPixel OpenCV Face Detector Implementation
 */

#include "opencv_face_detector.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace gpupixel {

OpenCVFaceDetector::OpenCVFaceDetector() : initialized_(false), debug_mode_(false) {
}

OpenCVFaceDetector::~OpenCVFaceDetector() {
}

bool OpenCVFaceDetector::Init() {
    // Load Haar cascade classifiers - try multiple paths
    std::string face_cascade_path = "models/haarcascade_frontalface_alt.xml";
    if (!face_cascade_.load(face_cascade_path)) {
        // Try system installation path
        face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
        if (!face_cascade_.load(face_cascade_path)) {
            // Try local path
            face_cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
            if (!face_cascade_.load(face_cascade_path)) {
                std::cerr << "Error loading face cascade from: " << face_cascade_path << std::endl;
                std::cerr << "Failed to initialize OpenCV face detector" << std::endl;
                return false;
            }
        }
    }
    
    std::string eye_cascade_path = "models/haarcascade_eye_tree_eyeglasses.xml";
    if (!eye_cascade_.load(eye_cascade_path)) {
        // Try system installation path
        eye_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
        if (!eye_cascade_.load(eye_cascade_path)) {
            // Try local path  
            eye_cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
            if (!eye_cascade_.load(eye_cascade_path)) {
                std::cerr << "Warning: Could not load eye cascade from: " << eye_cascade_path << std::endl;
                // Eye detection is optional, continue without it
            }
        }
    }
    
    initialized_ = true;
    std::cout << "OpenCV Face Detector initialized successfully" << std::endl;
    return true;
}

std::vector<float> OpenCVFaceDetector::DetectFace(const cv::Mat& frame) {
    if (!initialized_) {
        return std::vector<float>();
    }
    
    // Convert to grayscale for detection (frame is now RGBA from camera)
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_RGBA2GRAY);
    
    // Detect faces
    std::vector<cv::Rect> faces;
    face_cascade_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    
    // Store detected faces for debug rendering
    last_detected_faces_ = faces;
    
    if (faces.empty()) {
        last_detected_eyes_.clear();
        return std::vector<float>();
    }
    
    // Use the largest face
    cv::Rect largest_face = faces[0];
    for (const auto& face : faces) {
        if (face.area() > largest_face.area()) {
            largest_face = face;
        }
    }
    
    // Detect eyes within the face region
    cv::Mat face_roi = gray(largest_face);
    std::vector<cv::Rect> eyes;
    eye_cascade_.detectMultiScale(face_roi, eyes, 1.1, 3, 0, cv::Size(10, 10));
    
    // Adjust eye coordinates to be relative to full frame
    for (auto& eye : eyes) {
        eye.x += largest_face.x;
        eye.y += largest_face.y;
    }
    
    // Store detected eyes for debug rendering
    last_detected_eyes_ = eyes;
    
    // Draw debug rectangles if debug mode is enabled
    if (debug_mode_) {
        cv::Mat& debug_frame = const_cast<cv::Mat&>(frame);
        
        // Draw face rectangle in green
        cv::rectangle(debug_frame, largest_face, cv::Scalar(0, 255, 0), 2);
        
        // Draw eye rectangles in blue
        for (const auto& eye : eyes) {
            cv::rectangle(debug_frame, eye, cv::Scalar(255, 0, 0), 2);
        }
        
        // Add text labels
        cv::putText(debug_frame, "Face", 
                   cv::Point(largest_face.x, largest_face.y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        int eye_count = 0;
        for (const auto& eye : eyes) {
            cv::putText(debug_frame, "Eye" + std::to_string(++eye_count), 
                       cv::Point(eye.x, eye.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 0, 0), 1);
        }
    }
    
    return GenerateLandmarksFromFace(largest_face, eyes, frame.cols, frame.rows);
}

std::vector<float> OpenCVFaceDetector::GenerateLandmarksFromFace(const cv::Rect& face, 
                                                                    const std::vector<cv::Rect>& eyes,
                                                                    int frame_width, int frame_height) {
    std::vector<float> landmarks;
    
    // GPUPixel expects 106 landmarks (212 float values) in normalized coordinates (0 to 1)
    // Based on the shader, we need specific indices: 3,7,10,14,16,18,22,25,29 for thinFace
    // and indices 72,74,75,77 for bigEye
    
    // Calculate normalized coordinates (0 to 1, not -1 to 1)
    float face_left = (float)face.x / frame_width;
    float face_right = (float)(face.x + face.width) / frame_width;
    float face_top = (float)face.y / frame_height;
    float face_bottom = (float)(face.y + face.height) / frame_height;
    
    float face_center_x = (face_left + face_right) / 2.0f;
    float face_center_y = (face_top + face_bottom) / 2.0f;
    float face_width = face_right - face_left;
    float face_height = face_bottom - face_top;
    
    // Initialize landmarks array with 106 points (212 values)
    landmarks.resize(212, 0.0f);
    
    // Fill key landmarks needed by shader:
    // Index 3: Left jaw point
    landmarks[3*2] = face_left + face_width * 0.1f;     // x
    landmarks[3*2+1] = face_bottom - face_height * 0.2f; // y
    
    // Index 7: Lower left jaw
    landmarks[7*2] = face_left + face_width * 0.2f;
    landmarks[7*2+1] = face_bottom - face_height * 0.1f;
    
    // Index 10: Left side face
    landmarks[10*2] = face_left + face_width * 0.15f;
    landmarks[10*2+1] = face_center_y;
    
    // Index 14: Left center face
    landmarks[14*2] = face_left + face_width * 0.3f;
    landmarks[14*2+1] = face_center_y + face_height * 0.1f;
    
    // Index 16: Face center
    landmarks[16*2] = face_center_x;
    landmarks[16*2+1] = face_center_y + face_height * 0.1f;
    
    // Index 18: Right center face
    landmarks[18*2] = face_right - face_width * 0.3f;
    landmarks[18*2+1] = face_center_y + face_height * 0.1f;
    
    // Index 22: Right side face
    landmarks[22*2] = face_right - face_width * 0.15f;
    landmarks[22*2+1] = face_center_y;
    
    // Index 25: Lower right jaw
    landmarks[25*2] = face_right - face_width * 0.2f;
    landmarks[25*2+1] = face_bottom - face_height * 0.1f;
    
    // Index 29: Right jaw point
    landmarks[29*2] = face_right - face_width * 0.1f;
    landmarks[29*2+1] = face_bottom - face_height * 0.2f;
    
    // Target points for face slimming (indices 44,45,46,49)
    landmarks[44*2] = face_left + face_width * 0.2f;     // target for index 3
    landmarks[44*2+1] = face_bottom - face_height * 0.15f;
    
    landmarks[45*2] = face_left + face_width * 0.25f;    // target for index 7
    landmarks[45*2+1] = face_bottom - face_height * 0.05f;
    
    landmarks[46*2] = face_left + face_width * 0.25f;    // target for index 10
    landmarks[46*2+1] = face_center_y;
    
    landmarks[49*2] = face_center_x;                     // target for center points
    landmarks[49*2+1] = face_center_y + face_height * 0.05f;
    
    // Eye landmarks for big eye effect
    if (eyes.size() >= 2) {
        // Sort eyes by x position (left, right)
        std::vector<cv::Rect> sorted_eyes = eyes;
        std::sort(sorted_eyes.begin(), sorted_eyes.end(), 
                 [](const cv::Rect& a, const cv::Rect& b) { return a.x < b.x; });
        
        // Left eye center (index 74) and outer corner (index 72)
        cv::Rect left_eye = sorted_eyes[0];
        float left_eye_center_x = (float)(left_eye.x + left_eye.width/2) / frame_width;
        float left_eye_center_y = (float)(left_eye.y + left_eye.height/2) / frame_height;
        
        landmarks[74*2] = left_eye_center_x;         // Left eye center
        landmarks[74*2+1] = left_eye_center_y;
        
        landmarks[72*2] = left_eye_center_x - face_width * 0.05f;  // Left eye outer
        landmarks[72*2+1] = left_eye_center_y;
        
        // Right eye center (index 77) and outer corner (index 75)
        cv::Rect right_eye = sorted_eyes[1];
        float right_eye_center_x = (float)(right_eye.x + right_eye.width/2) / frame_width;
        float right_eye_center_y = (float)(right_eye.y + right_eye.height/2) / frame_height;
        
        landmarks[77*2] = right_eye_center_x;        // Right eye center
        landmarks[77*2+1] = right_eye_center_y;
        
        landmarks[75*2] = right_eye_center_x + face_width * 0.05f; // Right eye outer
        landmarks[75*2+1] = right_eye_center_y;
    } else {
        // No eyes detected, use approximate positions
        float eye_y = face_top + face_height * 0.4f;
        float left_eye_x = face_left + face_width * 0.3f;
        float right_eye_x = face_right - face_width * 0.3f;
        
        // Left eye landmarks
        landmarks[74*2] = left_eye_x;
        landmarks[74*2+1] = eye_y;
        landmarks[72*2] = left_eye_x - face_width * 0.05f;
        landmarks[72*2+1] = eye_y;
        
        // Right eye landmarks
        landmarks[77*2] = right_eye_x;
        landmarks[77*2+1] = eye_y;
        landmarks[75*2] = right_eye_x + face_width * 0.05f;
        landmarks[75*2+1] = eye_y;
    }
    
    return landmarks;
}

void OpenCVFaceDetector::SetDebugMode(bool enabled) {
    debug_mode_ = enabled;
}

} // namespace gpupixel
