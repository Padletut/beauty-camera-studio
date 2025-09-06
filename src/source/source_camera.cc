/*
 * GPUPixel
 *
 * Created by PixPark on 2021/6/24.
 * Copyright © 2021 PixPark. All rights reserved.
 */

#include "gpupixel/source/source_camera.h"
#include "core/gpupixel_context.h"
#include "utils/logging.h"
#include <opencv2/imgproc.hpp>
#include <chrono>

namespace gpupixel {

std::shared_ptr<SourceCamera> SourceCamera::Create(int camera_id, int width, int height) {
  auto source_camera = std::shared_ptr<SourceCamera>(new SourceCamera(camera_id, width, height));
  if (source_camera && source_camera->Init()) {
    return source_camera;
  }
  return nullptr;
}

SourceCamera::SourceCamera(int camera_id, int width, int height)
    : camera_id_(camera_id), width_(width), height_(height), running_(false), frame_available_(false) {
  LOG_DEBUG("SourceCamera constructor");
}

SourceCamera::~SourceCamera() {
  LOG_DEBUG("SourceCamera destructor");
  Stop();
}

bool SourceCamera::Init() {
  LOG_DEBUG("SourceCamera Init");
  
  // Open camera with V4L2 backend for better compatibility
  camera_.open(camera_id_, cv::CAP_V4L2);
  if (!camera_.isOpened()) {
    LOG_ERROR("Failed to open camera " + std::to_string(camera_id_));
    return false;
  }

  // Set camera properties
  camera_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
  camera_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
  camera_.set(cv::CAP_PROP_FPS, 30);

  // Get actual resolution
  int actual_width = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH));
  int actual_height = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT));
  
  LOG_INFO("Camera opened: " + std::to_string(actual_width) + "x" + std::to_string(actual_height));
  
  return true;
}

bool SourceCamera::Start() {
  if (running_.load()) {
    LOG_WARN("Camera already running");
    return true;
  }

  if (!camera_.isOpened()) {
    LOG_ERROR("Camera not opened");
    return false;
  }

  running_.store(true);
  capture_thread_ = std::make_unique<std::thread>(&SourceCamera::CaptureLoop, this);
  
  LOG_INFO("Camera capture started");
  return true;
}

void SourceCamera::Stop() {
  if (!running_.load()) {
    return;
  }

  running_.store(false);
  
  if (capture_thread_ && capture_thread_->joinable()) {
    capture_thread_->join();
  }
  
  camera_.release();
  LOG_INFO("Camera capture stopped");
}

void SourceCamera::CaptureLoop() {
  cv::Mat frame;
  
  while (running_.load()) {
    if (!camera_.read(frame)) {
      LOG_ERROR("Failed to read frame from camera");
      break;
    }
    
    if (!frame.empty()) {
      ProcessFrame(frame);
    }
    
    // Small delay to control frame rate
    std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
  }
}

void SourceCamera::ProcessFrame(const cv::Mat& frame) {
  // Convert BGR to RGB
  cv::Mat rgb_frame;
  cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
  
  // Store frame for main thread processing
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ = rgb_frame.clone();
    frame_available_ = true;
  }
}

void SourceCamera::ProcessLatestFrame() {
  cv::Mat frame_copy;
  bool has_frame = false;
  
  // Get the latest frame
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (frame_available_) {
      frame_copy = latest_frame_.clone();
      has_frame = true;
      // Keep frame_available_ = true so we can process it multiple times
    }
  }
  
  if (!has_frame) {
    return;
  }
  
  // Store the frame for later processing, don't do OpenGL operations here
  current_processing_frame_ = frame_copy;
}

void SourceCamera::TriggerPipelineProcessing() {
  // Only process if we have a frame ready
  if (current_processing_frame_.empty()) {
    return;
  }
  
  int width = current_processing_frame_.cols;
  int height = current_processing_frame_.rows;
  
  // Create framebuffer and upload texture data like SourceImage does
  auto framebuffer = GPUPixelContext::GetInstance()->GetFramebufferFactory()->CreateFramebuffer(width, height, true);
  
  // Convert RGB to RGBA since OpenGL texture expects RGBA
  cv::Mat rgba_frame;
  cv::cvtColor(current_processing_frame_, rgba_frame, cv::COLOR_RGB2RGBA);
  
  // Upload RGBA data to the framebuffer's texture
  glBindTexture(GL_TEXTURE_2D, framebuffer->GetTexture());
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba_frame.data);
  glBindTexture(GL_TEXTURE_2D, 0);
  
  // Set as input framebuffer and trigger rendering
  SetFramebuffer(framebuffer);
  DoRender();
}

cv::Mat SourceCamera::GetLatestFrame() {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  if (frame_available_) {
    return latest_frame_.clone();
  }
  return cv::Mat();
}

void SourceCamera::Proceed(bool auto_proceed) {
  // This method can be used to manually trigger frame processing
  // For now, it's handled automatically in the capture loop
}

} // namespace gpupixel
