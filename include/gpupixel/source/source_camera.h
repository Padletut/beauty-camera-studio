/*
 * GPUPixel
 *
 * Created by PixPark on 2021/6/24.
 * Copyright © 2021 PixPark. All rights reserved.
 */

#pragma once

#include "gpupixel/source/source.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>

namespace gpupixel {

class GPUPIXEL_API SourceCamera : public Source {
 public:
  static std::shared_ptr<SourceCamera> Create(int camera_id = 0, int width = 640, int height = 480);
  
  ~SourceCamera();

  // Start/stop camera capture
  bool Start();
  void Stop();
  
  // Public method to process the latest frame in main thread
  void ProcessLatestFrame();
  
  // Manually trigger pipeline processing
  void TriggerPipelineProcessing();
  
  // Get latest frame for face detection (thread-safe)
  cv::Mat GetLatestFrame();
  
  // Source interface
  virtual bool Init();
  virtual void Proceed(bool auto_proceed = true);

 protected:
  SourceCamera(int camera_id, int width, int height);

 private:
  void CaptureLoop();
  void ProcessFrame(const cv::Mat& frame);
  
  int camera_id_;
  int width_;
  int height_;
  
  cv::VideoCapture camera_;
  std::unique_ptr<std::thread> capture_thread_;
  std::atomic<bool> running_;
  
  // Latest frame data (protected by mutex)
  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  cv::Mat current_processing_frame_;
  bool frame_available_;
};

} // namespace gpupixel
