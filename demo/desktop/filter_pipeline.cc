#include "filter_pipeline.h"
#include <iostream>

namespace gpupixel {

FilterPipeline::FilterPipeline() : initialized_(false) {
}

FilterPipeline::~FilterPipeline() {
    Shutdown();
}

bool FilterPipeline::Initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "Initializing filter pipeline..." << std::endl;
    
    CreateFilters();
    SetupPipeline();
    
    initialized_ = true;
    std::cout << "Filter pipeline initialized successfully" << std::endl;
    return true;
}

void FilterPipeline::Shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "Shutting down filter pipeline..." << std::endl;
    
    // Reset all shared pointers
    source_camera_.reset();
    sink_raw_data_.reset();
    beauty_filter_.reset();
    reshape_filter_.reset();
    lipstick_filter_.reset();
    blusher_filter_.reset();
    saturation_filter_.reset();
    
    initialized_ = false;
    std::cout << "Filter pipeline shut down" << std::endl;
}

bool FilterPipeline::RecreateCamera(int camera_id, int width, int height) {
    std::cout << "Recreating camera: " << camera_id << " (" << width << "x" << height << ")" << std::endl;
    
    // Stop current camera if running
    if (source_camera_) {
        source_camera_->Stop();
    }
    
    // Create new camera source with parameters
    source_camera_ = SourceCamera::Create(camera_id, width, height);
    if (!source_camera_) {
        std::cerr << "Failed to create source camera" << std::endl;
        return false;
    }
    
    // Rebuild pipeline
    SetupPipeline();
    
    // Start camera
    source_camera_->Start();
    
    std::cout << "Camera recreated successfully" << std::endl;
    return true;
}

void FilterPipeline::UpdateFilterParameters(const FilterParameters& params) {
    if (!initialized_) {
        return;
    }
    
    // Update beauty filter using correct API
    if (beauty_filter_) {
        beauty_filter_->SetSharpen(params.beauty_strength / 100.0f);
        beauty_filter_->SetWhite(params.whitening_strength / 100.0f);
    }
    
    // Update reshape filter
    if (reshape_filter_) {
        reshape_filter_->SetFaceSlimLevel(params.face_slim_strength / 200.0f);
        reshape_filter_->SetEyeZoomLevel(params.eye_enlarge_strength / 100.0f);
    }
    
    // Update lipstick filter  
    if (lipstick_filter_) {
        lipstick_filter_->SetBlendLevel(params.color_tint_strength * 0.5f);
    }
    
    // Update saturation filter using correct API
    if (saturation_filter_) {
        saturation_filter_->setSaturation(1.0f + params.warmth_strength / 100.0f);
    }
}

void FilterPipeline::UpdateFaceLandmarks(const std::vector<float>& landmarks) {
    if (!initialized_ || landmarks.empty()) {
        return;
    }
    
    // Update filters that use face landmarks
    if (reshape_filter_) {
        reshape_filter_->SetFaceLandmarks(landmarks);
    }
    
    if (lipstick_filter_) {
        lipstick_filter_->SetFaceLandmarks(landmarks);
    }
    
    if (blusher_filter_) {
        blusher_filter_->SetFaceLandmarks(landmarks);
    }
}

void FilterPipeline::CreateFilters() {
    std::cout << "Creating filters..." << std::endl;
    
    // Create filters - Only use known working ones
    beauty_filter_ = BeautyFaceFilter::Create();
    if (!beauty_filter_) {
        std::cerr << "Failed to create beauty filter" << std::endl;
    }
    
    // Keep others created but don't add to pipeline initially
    reshape_filter_ = FaceReshapeFilter::Create();
    if (!reshape_filter_) {
        std::cerr << "Failed to create reshape filter" << std::endl;
    }
    
    lipstick_filter_ = LipstickFilter::Create();
    if (!lipstick_filter_) {
        std::cerr << "Failed to create lipstick filter" << std::endl;
    }
    
    blusher_filter_ = BlusherFilter::Create();
    if (!blusher_filter_) {
        std::cerr << "Failed to create blusher filter" << std::endl;
    }
    
    saturation_filter_ = SaturationFilter::Create();
    if (!saturation_filter_) {
        std::cerr << "Failed to create saturation filter" << std::endl;
    }
    
    // Create sink
    sink_raw_data_ = SinkRawData::Create();
    if (!sink_raw_data_) {
        std::cerr << "Failed to create sink raw data" << std::endl;
    }
    
    std::cout << "Filters created successfully" << std::endl;
}

void FilterPipeline::SetupPipeline() {
    if (!source_camera_ || !sink_raw_data_ || !beauty_filter_) {
        std::cerr << "Cannot setup pipeline: missing required components" << std::endl;
        return;
    }
    
    std::cout << "Setting up filter pipeline..." << std::endl;
    
    // Build the pipeline: Camera -> Beauty Filter -> Sink
    // Only use beauty filter for now, others can be enabled later
    source_camera_->AddSink(beauty_filter_)
                  ->AddSink(sink_raw_data_);
    
    std::cout << "Filter pipeline setup complete" << std::endl;
}

} // namespace gpupixel
