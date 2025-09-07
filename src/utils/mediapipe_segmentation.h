#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>
#include <string>

namespace gpupixel {

/**
 * @brief TensorFlow Lite-based real-time person segmentation
 * 
 * This class provides high-quality person segmentation using Google's
 * LiteRT (TensorFlow Lite) selfie segmentation models via OpenCV DNN.
 * Accurate person detection without overly aggressive background removal.
 */
class MediaPipeSegmentation {
public:
    MediaPipeSegmentation();
    ~MediaPipeSegmentation();

    /**
     * @brief Initialize the TensorFlow Lite segmentation model
     * @param model_path Path to the TensorFlow Lite model (.tflite)
     * @return true if initialization successful, false otherwise
     */
    bool Initialize(const std::string& model_path);

    /**
     * @brief Process a frame and generate person segmentation mask
     * @param input_frame Input RGB image (OpenCV Mat in BGR format)
     * @param output_mask Output binary mask (0=background, 255=person)
     * @return true if processing successful, false otherwise
     */
    bool ProcessFrame(const cv::Mat& input_frame, cv::Mat& output_mask);

    /**
     * @brief Check if the segmentation system is ready for processing
     * @return true if initialized and ready, false otherwise
     */
    bool IsReady() const { return is_initialized_; }

    /**
     * @brief Set confidence threshold for segmentation
     * @param threshold Confidence threshold (0.0 to 1.0, default 0.5)
     */
    void SetConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }

    /**
     * @brief Enable/disable temporal smoothing for stable results
     * @param enable Whether to enable temporal smoothing
     * @param alpha Smoothing factor (0.0 to 1.0, default 0.7)
     */
    void SetTemporalSmoothing(bool enable, float alpha = 0.7f);

private:
    bool is_initialized_;
    float confidence_threshold_;
    bool temporal_smoothing_enabled_;
    float temporal_alpha_;
    int model_input_size_;
    
    // OpenCV DNN for TensorFlow Lite inference
    cv::dnn::Net dnn_net_;
    
    // Temporal smoothing
    cv::Mat previous_mask_;
    
    // Model preprocessing/postprocessing
    cv::Mat PreprocessFrame(const cv::Mat& input_frame);
    cv::Mat PostprocessMask(const cv::Mat& model_output, const cv::Size& original_size);
    void ApplyTemporalSmoothing(cv::Mat& current_mask);
    void ApplyMorphologicalOperations(cv::Mat& mask);
};

} // namespace gpupixel
