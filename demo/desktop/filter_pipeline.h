#pragma once

#include "gpupixel/gpupixel.h"
#include "gpupixel/source/source_camera.h"
#include "gpupixel/sink/sink_raw_data.h"
#include "gpupixel/filter/beauty_face_filter.h"
#include "gpupixel/filter/face_reshape_filter.h"
#include "gpupixel/filter/lipstick_filter.h"
#include "gpupixel/filter/blusher_filter.h"
#include "gpupixel/filter/saturation_filter.h"
#include "camera_manager.h"
#include <memory>

namespace gpupixel {

struct FilterParameters {
    float beauty_strength = 0.0f;
    float whitening_strength = 0.0f;
    float face_slim_strength = 0.0f;
    float eye_enlarge_strength = 0.0f;
    float color_tint_strength = 0.0f;
    float warmth_strength = 0.0f;
};

class FilterPipeline {
public:
    FilterPipeline();
    ~FilterPipeline();
    
    // Pipeline management
    bool Initialize();
    void Shutdown();
    bool RecreateCamera(int camera_id, int width, int height);
    
    // Filter updates
    void UpdateFilterParameters(const FilterParameters& params);
    void UpdateFaceLandmarks(const std::vector<float>& landmarks);
    
    // Getters
    std::shared_ptr<SourceCamera> GetSourceCamera() const { return source_camera_; }
    std::shared_ptr<SinkRawData> GetSinkRawData() const { return sink_raw_data_; }
    
    // Status
    bool IsInitialized() const { return initialized_; }
    
private:
    bool initialized_;
    
    // Pipeline components
    std::shared_ptr<SourceCamera> source_camera_;
    std::shared_ptr<SinkRawData> sink_raw_data_;
    
    // Filters
    std::shared_ptr<BeautyFaceFilter> beauty_filter_;
    std::shared_ptr<FaceReshapeFilter> reshape_filter_;
    std::shared_ptr<LipstickFilter> lipstick_filter_;
    std::shared_ptr<BlusherFilter> blusher_filter_;
    std::shared_ptr<SaturationFilter> saturation_filter_;
    
    // Helper methods
    void CreateFilters();
    void SetupPipeline();
};

} // namespace gpupixel
