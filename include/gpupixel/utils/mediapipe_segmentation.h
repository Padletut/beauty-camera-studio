#pragma once

#include "gpupixel/gpupixel_define.h"
#include "gpupixel/utils/rvm_processor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace gpupixel {

/**
 * @brief MediaPipe-style configuration options for segmentation
 */
struct MediaPipeConfig {
    // Running modes (similar to MediaPipe framework)
    enum RunningMode {
        IMAGE,        // Single image processing
        VIDEO,        // Video frame processing  
        LIVE_STREAM   // Real-time camera stream
    };
    
    // Target class for segmentation
    enum TargetClass {
        PERSON = 0,      // Focus on person detection (default)
        BACKGROUND = 1,  // Focus on background detection
        AUTO_DETECT = 2  // Auto-detect based on model type
    };
    
    RunningMode running_mode = LIVE_STREAM;
    TargetClass target_class = PERSON;        // NEW: Class selection
    bool output_category_mask = false;        // Output uint8 category mask
    bool output_confidence_masks = true;      // Output float confidence map
    std::string display_names_locale = "en";
    float confidence_threshold = 0.7f;
    bool temporal_smoothing = true;
    float temporal_alpha = 0.8f;
    
    // RVM-style processing options
    bool enable_rvm_processing = false;       // Enable RVM-style temporal processing
    RVMProcessor::RVMConfig rvm_config;       // RVM processor configuration
    
    // Result callback for async processing (LIVE_STREAM mode)
    std::function<void(const cv::Mat& mask, float person_percentage)> result_callback;
};

/**
 * @brief MediaPipe segmentation result structure
 */
struct SegmentationResult {
    cv::Mat confidence_mask;     // Float confidence map [0.0-1.0]
    cv::Mat category_mask;       // Uint8 category mask (0=bg, 1=person)
    float person_percentage;     // Percentage of person pixels
    bool success;                // Processing success flag
    int64_t timestamp_ms;        // Processing timestamp
};

/**
 * @brief MediaPipe selfie segmentation with framework-like configuration
 * 
 * This class mimics MediaPipe framework configuration options while using
 * ONNX models with OpenCV DNN for cross-platform compatibility.
 */
class GPUPIXEL_API MediaPipeSegmentation {
public:
    MediaPipeSegmentation();
    ~MediaPipeSegmentation();

    /**
     * @brief Initialize with MediaPipe-style configuration
     * @param model_path Path to the ONNX model file
     * @param config MediaPipe configuration options
     * @return true if initialization successful, false otherwise
     */
    bool Initialize(const std::string& model_path, const MediaPipeConfig& config);
    
    /**
     * @brief Initialize with default configuration (backward compatibility)
     * @param model_path Path to the ONNX model file
     * @return true if initialization successful, false otherwise  
     */
    bool Initialize(const std::string& model_path = "");

    /**
     * @brief Process frame with MediaPipe-style result structure
     * @param input_frame Input RGB image (OpenCV Mat in BGR format)
     * @return SegmentationResult with all output options
     */
    SegmentationResult ProcessFrameAdvanced(const cv::Mat& input_frame);

    /**
     * @brief Process a frame (backward compatibility)
     * @param input_frame Input RGB image (OpenCV Mat in BGR format)
     * @param output_mask Output binary mask (0=background, 255=person)
     * @return true if processing successful, false otherwise
     */
    bool ProcessFrame(const cv::Mat& input_frame, cv::Mat& output_mask);

    /**
     * @brief Update configuration at runtime
     * @param config New configuration options
     */
    void UpdateConfig(const MediaPipeConfig& config);

    /**
     * @brief Check if the segmentation system is ready for processing
     * @return true if initialized and ready, false otherwise
     */
    bool IsReady() const { return is_initialized_; }

    /**
     * @brief Set confidence threshold for segmentation (backward compatibility)
     * @param threshold Confidence threshold (0.0 to 1.0, default 0.7)
     */
    void SetConfidenceThreshold(float threshold) { 
        config_.confidence_threshold = threshold; 
    }

    /**
     * @brief Get current configuration
     * @return Current MediaPipe configuration
     */
    const MediaPipeConfig& GetConfig() const { return config_; }

    /**
     * @brief Enable/disable temporal smoothing for stable results
     * @param enable Whether to enable temporal smoothing
     * @param alpha Smoothing factor (0.0 to 1.0, default 0.85)
     */
    void SetTemporalSmoothing(bool enable, float alpha = 0.85f);

    /**
     * @brief Enable/disable RVM-style processing
     * @param enable Whether to enable RVM processing
     * @param rvm_config RVM processor configuration
     */
    void SetRVMProcessing(bool enable, const RVMProcessor::RVMConfig& rvm_config = RVMProcessor::RVMConfig());

private:
    bool is_initialized_;
    MediaPipeConfig config_;      // MediaPipe-style configuration
    int frame_count_;
    
    // RVM processor for enhanced temporal processing
    std::unique_ptr<RVMProcessor> rvm_processor_;
    
    // Model dimensions and type (auto-detected)
    int model_width_;
    int model_height_;
    int model_channels_;
    bool use_float16_;
    std::string model_type_;
    
    // OpenCV DNN components
    cv::dnn::Net dnn_net_;
    
    // Temporal smoothing state
    cv::Mat previous_mask_;
    cv::Mat previous_confidence_;
    
    // Performance tracking
    int64_t last_process_time_ms_;
    
    // Helper methods
    void ApplyTemporalSmoothing(cv::Mat& current_mask, cv::Mat& confidence_map);
    void DetectModelType(const std::string& model_path);
    SegmentationResult CreateResult(const cv::Mat& confidence_map, bool success = true);
};

} // namespace gpupixel
