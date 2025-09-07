/*
 * GPUPixel OpenCV Face Detector Implementation
 * 
 * Alternative face detector using OpenCV instead of mars-face-kit
 * for Linux compatibility
 */

#include "opencv_face_detector.h"
#include <iostream>
#include <algorithm>

namespace gpupixel {

OpenCVFaceDetector::OpenCVFaceDetector() 
    : initialized_(false), debug_mode_(false) {
}

OpenCVFaceDetector::~OpenCVFaceDetector() {
}

bool OpenCVFaceDetector::Init() {
    std::cout << "[OpenCVFaceDetector] Initializing OpenCV face detector..." << std::endl;
    
    // Try to load face cascade classifier
    std::string face_cascade_path = "models/haarcascade_frontalface_alt.xml";
    if (!face_cascade_.load(face_cascade_path)) {
        // Try alternative path
        face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
        if (!face_cascade_.load(face_cascade_path)) {
            // Try another alternative
            face_cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
            if (!face_cascade_.load(face_cascade_path)) {
                std::cerr << "[OpenCVFaceDetector] Failed to load face cascade classifier" << std::endl;
                std::cerr << "[OpenCVFaceDetector] Tried paths:" << std::endl;
                std::cerr << "  - models/haarcascade_frontalface_alt.xml" << std::endl;
                std::cerr << "  - /usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml" << std::endl;
                std::cerr << "  - /usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml" << std::endl;
                return false;
            }
        }
    }
    
    std::cout << "[OpenCVFaceDetector] Face cascade loaded from: " << face_cascade_path << std::endl;
    
    // Try to load eye cascade classifier (optional)
    std::string eye_cascade_path = "models/haarcascade_eye_tree_eyeglasses.xml";
    if (!eye_cascade_.load(eye_cascade_path)) {
        eye_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
        if (!eye_cascade_.load(eye_cascade_path)) {
            eye_cascade_path = "/usr/local/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
            if (!eye_cascade_.load(eye_cascade_path)) {
                std::cout << "[OpenCVFaceDetector] Eye cascade not found, face detection only" << std::endl;
            } else {
                std::cout << "[OpenCVFaceDetector] Eye cascade loaded from: " << eye_cascade_path << std::endl;
            }
        } else {
            std::cout << "[OpenCVFaceDetector] Eye cascade loaded from: " << eye_cascade_path << std::endl;
        }
    } else {
        std::cout << "[OpenCVFaceDetector] Eye cascade loaded from: " << eye_cascade_path << std::endl;
    }
    
    initialized_ = true;
    std::cout << "[OpenCVFaceDetector] Initialization complete!" << std::endl;
    return true;
}

std::vector<float> OpenCVFaceDetector::DetectFace(const cv::Mat& frame) {
    last_detected_faces_.clear();
    last_detected_eyes_.clear();
    
    if (!initialized_) {
        return std::vector<float>();
    }
    
    if (frame.empty()) {
        return std::vector<float>();
    }
    
    // Convert to grayscale for detection
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.clone();
    }
    
    // Detect faces
    std::vector<cv::Rect> faces;
    face_cascade_.detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));
    
    if (faces.empty()) {
        return std::vector<float>();
    }
    
    // Store detected faces
    last_detected_faces_ = faces;
    
    // Use the largest face
    cv::Rect largest_face = *std::max_element(faces.begin(), faces.end(), 
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.area() < b.area();
        });
    
    // Detect eyes within the face region if eye cascade is available
    std::vector<cv::Rect> eyes;
    if (!eye_cascade_.empty()) {
        cv::Mat face_roi = gray(largest_face);
        std::vector<cv::Rect> face_eyes;
        eye_cascade_.detectMultiScale(face_roi, face_eyes, 1.1, 3, 0, cv::Size(10, 10));
        
        // Convert eye coordinates to full frame coordinates
        for (const auto& eye : face_eyes) {
            cv::Rect global_eye(eye.x + largest_face.x, eye.y + largest_face.y, eye.width, eye.height);
            eyes.push_back(global_eye);
        }
    }
    
    last_detected_eyes_ = eyes;
    
    // Generate landmarks from face and eyes
    return GenerateLandmarksFromFace(largest_face, eyes, frame.cols, frame.rows);
}

void OpenCVFaceDetector::SetDebugMode(bool enabled) {
    debug_mode_ = enabled;
    std::cout << "[OpenCVFaceDetector] Debug mode: " << (enabled ? "enabled" : "disabled") << std::endl;
}

std::vector<float> OpenCVFaceDetector::GenerateLandmarksFromFace(const cv::Rect& face, 
                                                                const std::vector<cv::Rect>& eyes,
                                                                int frame_width, int frame_height) {
    // Generate a simplified set of landmarks based on face rectangle and eyes
    // GPUPixel expects normalized coordinates (-1 to 1)
    std::vector<float> landmarks;
    
    // Normalize face coordinates
    float face_center_x = (face.x + face.width / 2.0f) / frame_width * 2.0f - 1.0f;
    float face_center_y = (face.y + face.height / 2.0f) / frame_height * 2.0f - 1.0f;
    float face_width_norm = face.width / (float)frame_width * 2.0f;
    float face_height_norm = face.height / (float)frame_height * 2.0f;
    
    // Generate basic face landmarks (simplified 68-point model)
    // Face outline (17 points)
    for (int i = 0; i < 17; ++i) {
        float t = i / 16.0f; // Parameter from 0 to 1
        float x = face_center_x + (t - 0.5f) * face_width_norm * 0.8f;
        float y = face_center_y + 0.3f * face_height_norm; // Bottom of face
        landmarks.push_back(x);
        landmarks.push_back(y);
    }
    
    // Eye landmarks
    if (eyes.size() >= 2) {
        // Left eye (6 points)
        cv::Rect left_eye = eyes[0];
        float left_eye_x = (left_eye.x + left_eye.width / 2.0f) / frame_width * 2.0f - 1.0f;
        float left_eye_y = (left_eye.y + left_eye.height / 2.0f) / frame_height * 2.0f - 1.0f;
        for (int i = 0; i < 6; ++i) {
            landmarks.push_back(left_eye_x);
            landmarks.push_back(left_eye_y);
        }
        
        // Right eye (6 points)
        cv::Rect right_eye = eyes[1];
        float right_eye_x = (right_eye.x + right_eye.width / 2.0f) / frame_width * 2.0f - 1.0f;
        float right_eye_y = (right_eye.y + right_eye.height / 2.0f) / frame_height * 2.0f - 1.0f;
        for (int i = 0; i < 6; ++i) {
            landmarks.push_back(right_eye_x);
            landmarks.push_back(right_eye_y);
        }
    } else {
        // Estimate eye positions from face rectangle
        float left_eye_x = face_center_x - 0.2f * face_width_norm;
        float right_eye_x = face_center_x + 0.2f * face_width_norm;
        float eye_y = face_center_y - 0.1f * face_height_norm;
        
        // Left eye (6 points)
        for (int i = 0; i < 6; ++i) {
            landmarks.push_back(left_eye_x);
            landmarks.push_back(eye_y);
        }
        
        // Right eye (6 points)
        for (int i = 0; i < 6; ++i) {
            landmarks.push_back(right_eye_x);
            landmarks.push_back(eye_y);
        }
    }
    
    // Nose landmarks (9 points) - estimated
    for (int i = 0; i < 9; ++i) {
        landmarks.push_back(face_center_x);
        landmarks.push_back(face_center_y);
    }
    
    // Mouth landmarks (20 points) - estimated
    for (int i = 0; i < 20; ++i) {
        landmarks.push_back(face_center_x);
        landmarks.push_back(face_center_y + 0.2f * face_height_norm);
    }
    
    // Eyebrow landmarks (10 points) - estimated
    for (int i = 0; i < 10; ++i) {
        float t = i / 9.0f;
        float x = face_center_x + (t - 0.5f) * face_width_norm * 0.6f;
        float y = face_center_y - 0.3f * face_height_norm;
        landmarks.push_back(x);
        landmarks.push_back(y);
    }
    
    return landmarks;
}

} // namespace gpupixel
