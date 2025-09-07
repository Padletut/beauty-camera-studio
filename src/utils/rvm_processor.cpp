#include "gpupixel/utils/rvm_processor.h"
#include <iostream>
#include <cmath>

namespace gpupixel {

bool RVMProcessor::Initialize(const RVMConfig& config) {
    config_ = config;
    
    // Validate configuration
    if (config_.temporal_buffer_size <= 0 || config_.temporal_buffer_size > 20) {
        std::cerr << "Invalid temporal buffer size: " << config_.temporal_buffer_size << std::endl;
        return false;
    }
    
    if (config_.temporal_weight < 0.0f || config_.temporal_weight > 1.0f) {
        std::cerr << "Invalid temporal weight: " << config_.temporal_weight << std::endl;
        return false;
    }
    
    // Clear buffers
    frame_buffer_.clear();
    mask_buffer_.clear();
    previous_mask_ = cv::Mat();
    motion_map_ = cv::Mat();
    
    initialized_ = true;
    std::cout << "RVM Processor initialized with temporal buffer size: " 
              << config_.temporal_buffer_size << std::endl;
    
    return true;
}

cv::Mat RVMProcessor::ProcessFrame(const cv::Mat& input_frame, const cv::Mat& initial_mask) {
    if (!initialized_) {
        std::cerr << "RVM Processor not initialized!" << std::endl;
        return initial_mask.clone();
    }
    
    if (input_frame.empty() || initial_mask.empty()) {
        std::cerr << "Empty input frame or mask!" << std::endl;
        return initial_mask.clone();
    }
    
    cv::Mat processed_mask = initial_mask.clone();
    
    // Convert mask to single channel float if needed
    if (processed_mask.channels() > 1) {
        cv::cvtColor(processed_mask, processed_mask, cv::COLOR_BGR2GRAY);
    }
    if (processed_mask.type() != CV_32F) {
        processed_mask.convertTo(processed_mask, CV_32F, 1.0/255.0);
    }
    
    // Update frame buffer
    frame_buffer_.push_back(input_frame.clone());
    if (frame_buffer_.size() > static_cast<size_t>(config_.temporal_buffer_size)) {
        frame_buffer_.pop_front();
    }
    
    // Motion compensation if enabled and we have previous frames
    if (config_.enable_motion_compensation && frame_buffer_.size() >= 2) {
        motion_map_ = DetectMotion(frame_buffer_.back(), frame_buffer_[frame_buffer_.size()-2]);
        
        // Reduce temporal smoothing in high-motion areas
        cv::Mat motion_weight;
        cv::threshold(motion_map_, motion_weight, config_.motion_threshold, 1.0, cv::THRESH_BINARY);
        motion_weight = 1.0f - motion_weight * 0.7f; // Reduce smoothing by 70% in motion areas
        
        // Apply motion-aware processing
        if (!previous_mask_.empty() && previous_mask_.size() == processed_mask.size()) {
            cv::Mat motion_adjusted_mask;
            cv::multiply(processed_mask, motion_weight, motion_adjusted_mask);
            cv::Mat prev_contribution;
            cv::multiply(previous_mask_, (1.0f - motion_weight), prev_contribution);
            processed_mask = motion_adjusted_mask + prev_contribution * config_.temporal_weight;
        }
    }
    
    // Update mask buffer
    mask_buffer_.push_back(processed_mask.clone());
    if (mask_buffer_.size() > static_cast<size_t>(config_.temporal_buffer_size)) {
        mask_buffer_.pop_front();
    }
    
    // Apply temporal smoothing
    if (mask_buffer_.size() > 1) {
        processed_mask = ApplyTemporalSmoothing(processed_mask);
    }
    
    // Edge refinement
    if (config_.enable_edge_refinement) {
        processed_mask = RefineEdges(processed_mask, input_frame);
    }
    
    // Store for next frame
    previous_mask_ = processed_mask.clone();
    
    return processed_mask;
}

cv::Mat RVMProcessor::DetectMotion(const cv::Mat& current_frame, const cv::Mat& previous_frame) {
    if (current_frame.size() != previous_frame.size()) {
        return cv::Mat::zeros(current_frame.size(), CV_32F);
    }
    
    cv::Mat current_gray, previous_gray;
    
    // Convert to grayscale
    if (current_frame.channels() == 3) {
        cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
    } else {
        current_gray = current_frame.clone();
    }
    
    if (previous_frame.channels() == 3) {
        cv::cvtColor(previous_frame, previous_gray, cv::COLOR_BGR2GRAY);
    } else {
        previous_gray = previous_frame.clone();
    }
    
    // Ensure same type
    if (current_gray.type() != CV_32F) {
        current_gray.convertTo(current_gray, CV_32F, 1.0/255.0);
    }
    if (previous_gray.type() != CV_32F) {
        previous_gray.convertTo(previous_gray, CV_32F, 1.0/255.0);
    }
    
    // Compute frame difference
    cv::Mat diff;
    cv::absdiff(current_gray, previous_gray, diff);
    
    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(diff, diff, cv::Size(5, 5), 1.0);
    
    // Normalize to [0, 1]
    cv::normalize(diff, diff, 0.0, 1.0, cv::NORM_MINMAX);
    
    return diff;
}

cv::Mat RVMProcessor::ApplyTemporalSmoothing(const cv::Mat& current_mask) {
    if (mask_buffer_.empty()) {
        return current_mask.clone();
    }
    
    cv::Mat smoothed_mask = cv::Mat::zeros(current_mask.size(), CV_32F);
    float total_weight = 0.0f;
    
    // Weighted average of masks in buffer
    for (size_t i = 0; i < mask_buffer_.size(); ++i) {
        if (mask_buffer_[i].size() != current_mask.size()) {
            continue; // Skip masks with different sizes
        }
        
        // More weight to recent frames
        float weight = std::exp(-static_cast<float>(mask_buffer_.size() - 1 - i) * 0.5f);
        
        cv::Mat mask_32f;
        if (mask_buffer_[i].type() != CV_32F) {
            mask_buffer_[i].convertTo(mask_32f, CV_32F, 1.0/255.0);
        } else {
            mask_32f = mask_buffer_[i];
        }
        
        smoothed_mask += mask_32f * weight;
        total_weight += weight;
    }
    
    if (total_weight > 0) {
        smoothed_mask /= total_weight;
    } else {
        smoothed_mask = current_mask.clone();
    }
    
    // Blend with current mask
    cv::Mat result;
    cv::addWeighted(current_mask, 1.0f - config_.temporal_weight, 
                   smoothed_mask, config_.temporal_weight, 0.0, result);
    
    return result;
}

cv::Mat RVMProcessor::RefineEdges(const cv::Mat& mask, const cv::Mat& frame) {
    cv::Mat refined_mask = mask.clone();
    
    // Convert mask to appropriate format
    if (refined_mask.type() != CV_32F) {
        refined_mask.convertTo(refined_mask, CV_32F, 1.0/255.0);
    }
    
    // Detect edges in the original frame
    cv::Mat frame_gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    } else {
        frame_gray = frame.clone();
    }
    
    cv::Mat edges;
    cv::Canny(frame_gray, edges, 50, 150);
    edges.convertTo(edges, CV_32F, 1.0/255.0);
    
    // Find mask edges
    cv::Mat mask_edges;
    cv::Mat mask_8u;
    refined_mask.convertTo(mask_8u, CV_8U, 255.0);
    cv::Canny(mask_8u, mask_edges, 50, 150);
    mask_edges.convertTo(mask_edges, CV_32F, 1.0/255.0);
    
    // Refine mask near edges
    cv::Mat edge_influence;
    cv::multiply(edges, mask_edges, edge_influence);
    
    // Apply guided filter-like refinement
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat refined_edges;
    cv::morphologyEx(edge_influence, refined_edges, cv::MORPH_CLOSE, kernel);
    
    // Blend refined edges with original mask
    cv::Mat result;
    cv::addWeighted(refined_mask, 0.8f, refined_edges, 0.2f, 0.0, result);
    
    // Ensure values are in [0, 1] range
    cv::threshold(result, result, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(result, result, 0.0, 0.0, cv::THRESH_TOZERO);
    
    return result;
}

void RVMProcessor::ResetBuffer() {
    frame_buffer_.clear();
    mask_buffer_.clear();
    previous_mask_ = cv::Mat();
    motion_map_ = cv::Mat();
    
    std::cout << "RVM Processor buffer reset" << std::endl;
}

} // namespace gpupixel
