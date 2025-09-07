/*
 * GPUPixel
 *
 * Created by PixPark on 2021/6/24.
 * Copyright © 2021 PixPark. All rights reserved.
 */

#include "gpupixel/face_detector/face_detector.h"
#include "opencv_face_detector.h"
#include <cassert>
#include <memory>
#include "utils/util.h"

namespace gpupixel {

std::shared_ptr<FaceDetector> FaceDetector::Create() {
  return std::shared_ptr<FaceDetector>(new FaceDetector());
}

FaceDetector::FaceDetector() : opencv_detector_(nullptr) {
  // Initialize OpenCV face detector
  opencv_detector_ = std::make_shared<OpenCVFaceDetector>();
  if (!opencv_detector_->Init()) {
    std::cerr << "Failed to initialize OpenCV face detector" << std::endl;
    opencv_detector_.reset();
  }
}

FaceDetector::~FaceDetector() {
  opencv_detector_.reset();
}

std::vector<float> FaceDetector::Detect(const uint8_t* data, int width,
                                      int height, int stride,
                                      GPUPIXEL_MODE_FMT fmt,
                                      GPUPIXEL_FRAME_TYPE type) {
  if (!opencv_detector_) {
    return std::vector<float>();
  }
  
  // Convert input data to OpenCV Mat
  cv::Mat frame;
  if (type == GPUPIXEL_FRAME_TYPE_RGBA) {
    cv::Mat rgba_frame(height, width, CV_8UC4, (void*)data, stride);
    cv::cvtColor(rgba_frame, frame, cv::COLOR_RGBA2BGR);
  } else if (type == GPUPIXEL_FRAME_TYPE_BGRA) {
    cv::Mat bgra_frame(height, width, CV_8UC4, (void*)data, stride);
    cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
  } else {
    // Assume BGR format
    frame = cv::Mat(height, width, CV_8UC3, (void*)data, stride);
  }
  
  // Use OpenCV face detector
  return opencv_detector_->DetectFace(frame);
}

}  // namespace gpupixel
