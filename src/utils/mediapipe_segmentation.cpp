#include "gpupixel/utils/mediapipe_segmentation.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <opencv2/dnn.hpp>

namespace gpupixel {

MediaPipeSegmentation::MediaPipeSegmentation()
    : is_initialized_(false),
      frame_count_(0),
      model_width_(256),
      model_height_(256), 
      model_channels_(3),
      use_float16_(false),
      last_process_time_ms_(0),
      rvm_processor_(nullptr) {
    // Initialize with default configuration
    config_.running_mode = MediaPipeConfig::LIVE_STREAM;
    config_.target_class = MediaPipeConfig::PERSON;  // Default to person detection
    config_.output_category_mask = false;
    config_.output_confidence_masks = true;
    config_.display_names_locale = "en";
    config_.confidence_threshold = 0.3f;
    config_.temporal_smoothing = true;
    config_.temporal_alpha = 0.8f;
    config_.enable_rvm_processing = false;
}

MediaPipeSegmentation::~MediaPipeSegmentation() {
    // OpenCV DNN cleanup is automatic
}

bool MediaPipeSegmentation::Initialize(const std::string& model_path, const MediaPipeConfig& config) {
    config_ = config;
    return Initialize(model_path);
}

bool MediaPipeSegmentation::Initialize(const std::string& model_path) {
    std::string actual_model_path = model_path;
    if (actual_model_path.empty()) {
        actual_model_path = "models/mediapipe_landscape_segmentation.onnx";
    }

    std::cout << "[MediaPipeSegmentation] Loading MediaPipe ONNX model: " << actual_model_path << std::endl;

    try {
        // Load the ONNX model using OpenCV DNN
        dnn_net_ = cv::dnn::readNetFromONNX(actual_model_path);
        
        if (dnn_net_.empty()) {
            std::cerr << "[MediaPipeSegmentation] Failed to load ONNX model: " << actual_model_path << std::endl;
            return false;
        }

        // Set backend and target
        dnn_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        dnn_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        // Auto-detect model type and set appropriate parameters
        std::string filename = actual_model_path.substr(actual_model_path.find_last_of("/\\") + 1);
        
        if (filename.find("landscape") != std::string::npos) {
            // SelfieSegmenter (landscape): 144x256, float16
            model_width_ = 256;
            model_height_ = 144;
            model_channels_ = 3;
            use_float16_ = true;
            model_type_ = "landscape";
            std::cout << "[MediaPipeSegmentation] Detected: SelfieSegmenter (landscape)" << std::endl;
        } else if (filename.find("multiclass") != std::string::npos) {
            // SelfieMulticlass: 256x256, float32
            model_width_ = 256;
            model_height_ = 256;
            model_channels_ = 3;
            use_float16_ = false;
            model_type_ = "multiclass";
            std::cout << "[MediaPipeSegmentation] Detected: SelfieMulticlass" << std::endl;
        } else if (filename.find("deeplabv3") != std::string::npos || filename.find("deeplab") != std::string::npos) {
            // DeepLab-V3: 257x257, float32
            model_width_ = 257;
            model_height_ = 257;
            model_channels_ = 3;
            use_float16_ = false;
            model_type_ = "deeplabv3";
            std::cout << "[MediaPipeSegmentation] Detected: DeepLab-V3" << std::endl;
        } else if (filename.find("modnet") != std::string::npos || filename.find("MODNet") != std::string::npos) {
            // MODNet: 512x512, float32 (high-quality portrait matting)
            model_width_ = 512;
            model_height_ = 512;
            model_channels_ = 3;
            use_float16_ = false;
            model_type_ = "modnet";
            std::cout << "[MediaPipeSegmentation] Detected: MODNet (Portrait Matting)" << std::endl;
        } else if (filename.find("mobilenet") != std::string::npos || filename.find("MobileNet") != std::string::npos) {
            // MobileNet-v2: 224x224, float32 (classification model - can be adapted for segmentation)
            model_width_ = 224;
            model_height_ = 224;
            model_channels_ = 3;
            use_float16_ = false;
            model_type_ = "mobilenet";
            std::cout << "[MediaPipeSegmentation] Detected: MobileNet-v2 (Classification/Feature Extraction)" << std::endl;
        } else {
            // SelfieSegmenter (square): 256x256, float16 (default)
            model_width_ = 256;
            model_height_ = 256;
            model_channels_ = 3;
            use_float16_ = true;
            model_type_ = "selfie";
            std::cout << "[MediaPipeSegmentation] Detected: SelfieSegmenter (square)" << std::endl;
        }
        
        std::cout << "[MediaPipeSegmentation] MediaPipe ONNX model loaded successfully" << std::endl;
        std::cout << "[MediaPipeSegmentation] Model input dimensions: " 
                  << model_width_ << "x" << model_height_ << "x" << model_channels_ << std::endl;
        
        // Initialize RVM processor if enabled
        if (config_.enable_rvm_processing) {
            rvm_processor_ = std::make_unique<RVMProcessor>();
            if (!rvm_processor_->Initialize(config_.rvm_config)) {
                std::cerr << "[MediaPipeSegmentation] Failed to initialize RVM processor" << std::endl;
                rvm_processor_.reset();
                config_.enable_rvm_processing = false;
            } else {
                std::cout << "[MediaPipeSegmentation] RVM processor initialized" << std::endl;
            }
        }
        
    } catch (const cv::Exception& e) {
        std::cerr << "[MediaPipeSegmentation] OpenCV DNN error loading model: " << e.what() << std::endl;
        return false;
    }

    // Initialize temporal smoothing storage
    if (config_.temporal_smoothing) {
        previous_mask_ = cv::Mat::zeros(model_height_, model_width_, CV_8UC1);
        previous_confidence_ = cv::Mat::zeros(model_height_, model_width_, CV_32F);
    }

    is_initialized_ = true;
    frame_count_ = 0;

    std::cout << "[MediaPipeSegmentation] MediaPipe ONNX segmentation initialized successfully using OpenCV DNN" << std::endl;
    std::cout << "[MediaPipeSegmentation] Model dimensions: " << model_width_ << "x" << model_height_ 
              << "x" << model_channels_ << std::endl;
    std::cout << "[MediaPipeSegmentation] Using MediaPipe ONNX model for selfie segmentation" << std::endl;

    return true;
}

bool MediaPipeSegmentation::ProcessFrame(const cv::Mat& input_frame, cv::Mat& output_mask) {
    if (!is_initialized_ || dnn_net_.empty()) {
        std::cerr << "[MediaPipeSegmentation] Not initialized or DNN network missing" << std::endl;
        return false;
    }

    if (input_frame.empty()) {
        std::cerr << "[MediaPipeSegmentation] Input frame is empty" << std::endl;
        return false;
    }

    frame_count_++;

    try {
        // Preprocess input to model size (256x256x3)
        cv::Mat resized_frame;
        cv::resize(input_frame, resized_frame, cv::Size(model_width_, model_height_));
        
        // Convert BGR to RGB (MediaPipe expects RGB input)
        cv::Mat rgb_frame;
        cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);
        
        // Create blob from image with model-specific data type
        cv::Mat blob;
        if (use_float16_) {
            // MediaPipe models (selfie, landscape) expect float16 but OpenCV DNN uses float32
            // The normalization and processing remains the same
            cv::dnn::blobFromImage(rgb_frame, blob, 1.0/255.0, cv::Size(model_width_, model_height_), cv::Scalar(0, 0, 0), true, false, CV_32F);
        } else {
            // Multi-class and DeepLab models expect float32
            cv::dnn::blobFromImage(rgb_frame, blob, 1.0/255.0, cv::Size(model_width_, model_height_), cv::Scalar(0, 0, 0), true, false, CV_32F);
        }
        
        // Set input to the network
        dnn_net_.setInput(blob);
        
        // Run forward pass
        cv::Mat segmentation_output = dnn_net_.forward();
        
        // Debug: Print output shape to understand the model output format
        std::vector<int> output_shape;
        for (int i = 0; i < segmentation_output.dims; i++) {
            output_shape.push_back(segmentation_output.size[i]);
        }
        if (frame_count_ <= 3) {  // Only print for first few frames
            std::cout << "[MediaPipeSegmentation] Model output shape: [";
            for (size_t i = 0; i < output_shape.size(); i++) {
                std::cout << output_shape[i];
                if (i < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "], total elements: " << segmentation_output.total() << std::endl;
        }
        
        // Process output based on model type
        cv::Mat segmentation_mask;
        if (model_type_ == "multiclass") {
            // Multi-class output: background (class 0) + person classes
            // For confidence mask with background selection, we need to invert background
            if (output_shape.size() == 4 && output_shape[1] > 1) {
                // Extract background class (class 0) and invert it to get person mask
                cv::Mat reshaped_output = segmentation_output.reshape(0, {output_shape[1], output_shape[2], output_shape[3]});
                cv::Mat background_class;
                cv::extractChannel(reshaped_output, background_class, 0);  // Background is class 0
                
                // Invert background to get person mask: person = 1 - background
                segmentation_mask = cv::Mat(model_height_, model_width_, CV_32F, background_class.ptr<float>());
                cv::subtract(cv::Scalar::all(1.0), segmentation_mask, segmentation_mask);
                
                if (frame_count_ <= 3) {
                    std::cout << "[MediaPipeSegmentation] Multi-class: Using inverted background (class 0) for person mask" << std::endl;
                }
            } else {
                std::cerr << "[MediaPipeSegmentation] Unexpected multi-class output format" << std::endl;
                return false;
            }
        } else if (model_type_ == "deeplabv3") {
            // DeepLab v3 output: Can extract different classes based on configuration
            if (output_shape.size() == 4 && output_shape[1] > 15) {
                cv::Mat reshaped_output = segmentation_output.reshape(0, {output_shape[1], output_shape[2], output_shape[3]});
                cv::Mat class_output;
                
                if (config_.target_class == MediaPipeConfig::BACKGROUND) {
                    // Extract background class (class 0)
                    cv::extractChannel(reshaped_output, class_output, 0);
                    segmentation_mask = cv::Mat(model_height_, model_width_, CV_32F, class_output.ptr<float>());
                    
                    if (frame_count_ <= 3) {
                        std::cout << "[MediaPipeSegmentation] DeepLab v3: Using background class (0)" << std::endl;
                    }
                } else {
                    // Extract person class (class 15 in PASCAL VOC)
                    cv::extractChannel(reshaped_output, class_output, 15);
                    segmentation_mask = cv::Mat(model_height_, model_width_, CV_32F, class_output.ptr<float>());
                    
                    if (frame_count_ <= 3) {
                        std::cout << "[MediaPipeSegmentation] DeepLab v3: Using person class (15)" << std::endl;
                    }
                }
            } else {
                std::cerr << "[MediaPipeSegmentation] Unexpected DeepLab v3 output format" << std::endl;
                return false;
            }
        } else if (model_type_ == "modnet") {
            // MODNet output: Alpha matte [0,1] directly representing person probability
            if (output_shape.size() == 4) {
                // MODNet outputs alpha matte in format [batch, height, width, 1] or [batch, 1, height, width]
                if (output_shape[1] == 1) {
                    // Format: [1, 1, 512, 512]
                    segmentation_mask = cv::Mat(model_height_, model_width_, CV_32F, segmentation_output.ptr<float>());
                } else {
                    // Format: [1, 512, 512, 1] - reshape needed
                    cv::Mat reshaped_output = segmentation_output.reshape(0, {output_shape[1], output_shape[2]});
                    segmentation_mask = reshaped_output.clone();
                }
                
                // MODNet outputs alpha matte [0,1] where 1=person, 0=background
                // Apply target class configuration
                if (config_.target_class == MediaPipeConfig::BACKGROUND) {
                    // Invert for background focus: background = 1 - person
                    cv::subtract(cv::Scalar::all(1.0), segmentation_mask, segmentation_mask);
                    if (frame_count_ <= 3) {
                        std::cout << "[MediaPipeSegmentation] MODNet: Using inverted alpha matte for background" << std::endl;
                    }
                } else {
                    if (frame_count_ <= 3) {
                        std::cout << "[MediaPipeSegmentation] MODNet: Using alpha matte for person detection" << std::endl;
                    }
                }
            } else {
                std::cerr << "[MediaPipeSegmentation] Unexpected MODNet output format" << std::endl;
                return false;
            }
        } else if (model_type_ == "mobilenet") {
            // MobileNet-v2 is a classification model - output is feature vector or class probabilities
            // We'll create a simple person detection based on classification confidence
            if (output_shape.size() >= 2) {
                // MobileNet typically outputs [batch_size, num_classes] where num_classes = 1000 for ImageNet
                if (output_shape.back() == 1000) {
                    // ImageNet classification - look for person-related classes
                    // Person class in ImageNet is typically around index 15 or similar
                    float* output_data = segmentation_output.ptr<float>();
                    
                    // Check confidence for person-related classes (simplified approach)
                    float person_confidence = 0.0f;
                    // In ImageNet, person-related classes are scattered, so we'll use a heuristic
                    for (int i = 0; i < std::min(100, (int)output_shape.back()); i++) {
                        if (output_data[i] > person_confidence) {
                            person_confidence = output_data[i];
                        }
                    }
                    
                    // Create a uniform mask based on classification confidence
                    segmentation_mask = cv::Mat::ones(model_height_, model_width_, CV_32F) * person_confidence;
                    
                    if (frame_count_ <= 3) {
                        std::cout << "[MediaPipeSegmentation] MobileNet-v2: Using classification confidence as uniform mask: " 
                                  << person_confidence << std::endl;
                    }
                } else {
                    // Unknown output format - create default mask
                    segmentation_mask = cv::Mat::zeros(model_height_, model_width_, CV_32F);
                    if (frame_count_ <= 3) {
                        std::cout << "[MediaPipeSegmentation] MobileNet-v2: Unknown output format, using default mask" << std::endl;
                    }
                }
            } else {
                std::cerr << "[MediaPipeSegmentation] Unexpected MobileNet-v2 output format" << std::endl;
                return false;
            }
        } else {
            // Standard MediaPipe selfie/landscape models - single class output
            if (output_shape.size() == 4 && output_shape[1] > 1) {
                // Multi-class format but using as single class (extract class 1)
                cv::Mat reshaped_output = segmentation_output.reshape(0, {output_shape[1], output_shape[2], output_shape[3]});
                cv::Mat class_output;
                cv::extractChannel(reshaped_output, class_output, 1);
                segmentation_mask = cv::Mat(model_height_, model_width_, CV_32F, class_output.ptr<float>());
                
                if (frame_count_ <= 3) {
                    std::cout << "[MediaPipeSegmentation] MediaPipe: Using class 1 (person/selfie)" << std::endl;
                }
            } else {
                // Direct single-class output
                segmentation_mask = cv::Mat(model_height_, model_width_, CV_32F, segmentation_output.ptr<float>());
                
                if (frame_count_ <= 3) {
                    std::cout << "[MediaPipeSegmentation] MediaPipe: Using direct single-class output" << std::endl;
                }
            }
        }
        
        // Apply sigmoid activation if needed (some MediaPipe models output logits)
        cv::Mat confidence_mask;
        segmentation_mask.copyTo(confidence_mask);
        
        // Ensure values are in [0, 1] range (sigmoid if needed)
        double minVal, maxVal;
        cv::minMaxLoc(confidence_mask, &minVal, &maxVal);
        if (maxVal > 1.0 || minVal < 0.0) {
            // Apply sigmoid to convert logits to probabilities
            cv::exp(-confidence_mask, confidence_mask);
            confidence_mask = 1.0 / (1.0 + confidence_mask);
        }
        
        // Use confidence-based thresholding instead of hard threshold
        cv::Mat soft_mask;
        confidence_mask.copyTo(soft_mask);
        
        // Apply smooth confidence threshold
        cv::Mat binary_mask;
        cv::threshold(soft_mask, binary_mask, config_.confidence_threshold, 1.0, cv::THRESH_BINARY);
        
        // Debug: Print confidence range to understand detection quality
        if (frame_count_ <= 5) {
            double minVal, maxVal;
            cv::minMaxLoc(soft_mask, &minVal, &maxVal);
            std::cout << "[MediaPipeSegmentation] Frame " << frame_count_ 
                      << " confidence range: [" << minVal << ", " << maxVal << "]" 
                      << ", threshold: " << config_.confidence_threshold << std::endl;
        }
        
        // Convert to 8-bit with smooth edges (preserve confidence for better edges)
        cv::Mat smooth_mask;
        soft_mask.convertTo(smooth_mask, CV_8U, 255.0);
        
        // Use smooth mask for better edge quality
        cv::Mat final_mask;
        cv::resize(smooth_mask, final_mask, cv::Size(input_frame.cols, input_frame.rows), 0, 0, cv::INTER_LINEAR);
        
        // Apply temporal smoothing if enabled
        if (config_.temporal_smoothing) {
            // Create confidence map at the same size as final_mask for temporal smoothing
            cv::Mat confidence_resized;
            cv::resize(soft_mask, confidence_resized, cv::Size(input_frame.cols, input_frame.rows), 0, 0, cv::INTER_LINEAR);
            ApplyTemporalSmoothing(final_mask, confidence_resized);
        }
        
        output_mask = final_mask;
        
        if (frame_count_ % 30 == 0) {  // Log every 30 frames
            double mask_percentage = (cv::sum(output_mask)[0] / (255.0 * output_mask.total()) * 100.0);
            std::cout << "[MediaPipeSegmentation] MediaPipe ONNX inference frame " << frame_count_ 
                      << ", input: " << input_frame.cols << "x" << input_frame.rows
                      << ", person detected: " << mask_percentage << "%" << std::endl;
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "[MediaPipeSegmentation] OpenCV error processing frame: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[MediaPipeSegmentation] Error processing frame: " << e.what() << std::endl;
        return false;
    }
}

void MediaPipeSegmentation::SetTemporalSmoothing(bool enable, float alpha) {
    config_.temporal_smoothing = enable;
    config_.temporal_alpha = std::max(0.0f, std::min(1.0f, alpha));
    
    if (enable && previous_mask_.empty() && is_initialized_) {
        previous_mask_ = cv::Mat::zeros(model_height_, model_width_, CV_8UC1);
    }
    
    std::cout << "[MediaPipeSegmentation] Temporal smoothing: " 
              << (enable ? "enabled" : "disabled") 
              << " (alpha=" << config_.temporal_alpha << ")" << std::endl;
}

SegmentationResult MediaPipeSegmentation::ProcessFrameAdvanced(const cv::Mat& input_frame) {
    SegmentationResult result;
    result.success = false;
    result.timestamp_ms = cv::getTickCount() * 1000 / cv::getTickFrequency();
    
    if (!is_initialized_ || dnn_net_.empty()) {
        std::cerr << "[MediaPipeSegmentation] Not initialized" << std::endl;
        return result;
    }

    if (input_frame.empty()) {
        std::cerr << "[MediaPipeSegmentation] Input frame is empty" << std::endl;
        return result;
    }

    try {
        // Preprocess and run inference (reuse existing code)
        cv::Mat temp_mask;
        if (!ProcessFrame(input_frame, temp_mask)) {
            return result;
        }
        
        // Convert to confidence map [0.0-1.0]
        cv::Mat confidence_float;
        temp_mask.convertTo(confidence_float, CV_32F, 1.0/255.0);
        
        // Apply RVM processing if enabled
        if (config_.enable_rvm_processing && rvm_processor_ && rvm_processor_->IsInitialized()) {
            confidence_float = rvm_processor_->ProcessFrame(input_frame, confidence_float);
        }
        
        // Create result based on configuration
        result = CreateResult(confidence_float, true);
        
        // Execute callback if configured for LIVE_STREAM mode
        if (config_.running_mode == MediaPipeConfig::LIVE_STREAM && config_.result_callback) {
            config_.result_callback(result.category_mask, result.person_percentage);
        }
        
        return result;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[MediaPipeSegmentation] OpenCV error in advanced processing: " << e.what() << std::endl;
        return result;
    }
}

void MediaPipeSegmentation::UpdateConfig(const MediaPipeConfig& config) {
    config_ = config;
    std::cout << "[MediaPipeSegmentation] Configuration updated:" << std::endl;
    std::cout << "  Running mode: " << config_.running_mode << std::endl;
    std::cout << "  Confidence threshold: " << config_.confidence_threshold << std::endl;
    std::cout << "  Output category mask: " << config_.output_category_mask << std::endl;
    std::cout << "  Output confidence masks: " << config_.output_confidence_masks << std::endl;
    std::cout << "  Temporal smoothing: " << config_.temporal_smoothing << std::endl;
}

SegmentationResult MediaPipeSegmentation::CreateResult(const cv::Mat& confidence_map, bool success) {
    SegmentationResult result;
    result.success = success;
    result.timestamp_ms = cv::getTickCount() * 1000 / cv::getTickFrequency();
    
    if (!success) {
        return result;
    }
    
    // Always create confidence mask if requested
    if (config_.output_confidence_masks) {
        confidence_map.copyTo(result.confidence_mask);
    }
    
    // Create category mask if requested
    if (config_.output_category_mask) {
        cv::Mat binary_mask;
        cv::threshold(confidence_map, binary_mask, config_.confidence_threshold, 1.0, cv::THRESH_BINARY);
        binary_mask.convertTo(result.category_mask, CV_8U, 255.0);
    }
    
    // Calculate person percentage
    cv::Mat binary_temp;
    cv::threshold(confidence_map, binary_temp, config_.confidence_threshold, 1.0, cv::THRESH_BINARY);
    result.person_percentage = (cv::sum(binary_temp)[0] / binary_temp.total()) * 100.0;
    
    return result;
}

void MediaPipeSegmentation::ApplyTemporalSmoothing(cv::Mat& current_mask, cv::Mat& confidence_map) {
    if (!config_.temporal_smoothing) {
        return;
    }
    
    if (previous_mask_.empty() || previous_confidence_.empty() || 
        previous_mask_.size() != current_mask.size() || 
        previous_confidence_.size() != confidence_map.size()) {
        previous_mask_ = current_mask.clone();
        previous_confidence_ = confidence_map.clone();
        return;
    }
    
    // Ensure all matrices have the same type and size before operations
    cv::Mat temp_confidence, temp_mask;
    cv::Mat prev_confidence_resized, prev_mask_resized;
    
    // Resize previous matrices to match current if needed
    if (previous_confidence_.size() != confidence_map.size()) {
        cv::resize(previous_confidence_, prev_confidence_resized, confidence_map.size());
    } else {
        prev_confidence_resized = previous_confidence_;
    }
    
    if (previous_mask_.size() != current_mask.size()) {
        cv::resize(previous_mask_, prev_mask_resized, current_mask.size());
    } else {
        prev_mask_resized = previous_mask_;
    }
    
    // Ensure same data types
    prev_confidence_resized.convertTo(prev_confidence_resized, confidence_map.type());
    prev_mask_resized.convertTo(prev_mask_resized, current_mask.type());
    
    try {
        // Smooth confidence map
        cv::addWeighted(confidence_map, 1.0 - config_.temporal_alpha, 
                        prev_confidence_resized, config_.temporal_alpha, 0, temp_confidence);
        
        // Smooth mask
        cv::addWeighted(current_mask, 1.0 - config_.temporal_alpha,
                        prev_mask_resized, config_.temporal_alpha, 0, temp_mask);
        
        // Update outputs
        confidence_map = temp_confidence;
        current_mask = temp_mask;
        
        // Update previous states
        previous_mask_ = current_mask.clone();
        previous_confidence_ = confidence_map.clone();
        
    } catch (const cv::Exception& e) {
        std::cerr << "[MediaPipeSegmentation] Temporal smoothing error: " << e.what() << std::endl;
        // Fallback: just update previous states without smoothing
        previous_mask_ = current_mask.clone();
        previous_confidence_ = confidence_map.clone();
    }
}

void MediaPipeSegmentation::SetRVMProcessing(bool enable, const RVMProcessor::RVMConfig& rvm_config) {
    config_.enable_rvm_processing = enable;
    config_.rvm_config = rvm_config;
    
    if (enable) {
        if (!rvm_processor_) {
            rvm_processor_ = std::make_unique<RVMProcessor>();
        }
        
        if (!rvm_processor_->Initialize(rvm_config)) {
            std::cerr << "[MediaPipeSegmentation] Failed to initialize RVM processor" << std::endl;
            rvm_processor_.reset();
            config_.enable_rvm_processing = false;
        } else {
            std::cout << "[MediaPipeSegmentation] RVM processing enabled with temporal buffer size: " 
                      << rvm_config.temporal_buffer_size << std::endl;
        }
    } else {
        rvm_processor_.reset();
        std::cout << "[MediaPipeSegmentation] RVM processing disabled" << std::endl;
    }
}

} // namespace gpupixel
