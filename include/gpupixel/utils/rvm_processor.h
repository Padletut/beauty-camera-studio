#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <memory>

namespace gpupixel {

/**
 * @brief RVM-style temporal video matting processor
 * Implements RVM (Robust Video Matting) techniques on top of existing segmentation models
 */
class RVMProcessor {
public:
    struct RVMConfig {
        int temporal_buffer_size;          // Number of frames to use for temporal consistency
        float temporal_weight;             // Weight for temporal smoothing
        float motion_threshold;            // Threshold for motion detection
        bool enable_motion_compensation;   // Enable motion-aware processing
        bool enable_edge_refinement;       // Enable edge refinement
        float edge_threshold;              // Threshold for edge detection
        
        // Constructor with default values
        RVMConfig() : 
            temporal_buffer_size(5),
            temporal_weight(0.3f),
            motion_threshold(0.1f),
            enable_motion_compensation(true),
            enable_edge_refinement(true),
            edge_threshold(0.5f) {}
    };

private:
    RVMConfig config_;
    std::deque<cv::Mat> frame_buffer_;        // Temporal frame buffer
    std::deque<cv::Mat> mask_buffer_;         // Temporal mask buffer
    cv::Mat previous_mask_;                   // Previous frame mask
    cv::Mat motion_map_;                      // Motion detection map
    bool initialized_ = false;

public:
    RVMProcessor() = default;
    ~RVMProcessor() = default;

    /**
     * @brief Initialize RVM processor with configuration
     */
    bool Initialize(const RVMConfig& config = RVMConfig());

    /**
     * @brief Process frame with RVM-style temporal matting
     * @param input_frame Current input frame
     * @param initial_mask Initial segmentation mask from base model
     * @return Refined mask with temporal consistency
     */
    cv::Mat ProcessFrame(const cv::Mat& input_frame, const cv::Mat& initial_mask);

    /**
     * @brief Detect motion between current and previous frame
     */
    cv::Mat DetectMotion(const cv::Mat& current_frame, const cv::Mat& previous_frame);

    /**
     * @brief Apply temporal smoothing using buffer history
     */
    cv::Mat ApplyTemporalSmoothing(const cv::Mat& current_mask);

    /**
     * @brief Refine mask edges for better quality
     */
    cv::Mat RefineEdges(const cv::Mat& mask, const cv::Mat& frame);

    /**
     * @brief Reset temporal buffer (useful for scene changes)
     */
    void ResetBuffer();

    /**
     * @brief Check if processor is initialized
     */
    bool IsInitialized() const { return initialized_; }
};

} // namespace gpupixel
