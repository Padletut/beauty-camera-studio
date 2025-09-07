#pragma once

#include "gpupixel/filter/filter.h"
#include "gpupixel/gpupixel_define.h"
#include "../background/background_processor.h"
#include <opencv2/opencv.hpp>
#include <memory>

namespace gpupixel {

class GPUPIXEL_API BackgroundEffectFilter : public Filter {
public:
    static std::shared_ptr<BackgroundEffectFilter> Create();
    
    BackgroundEffectFilter();
    virtual ~BackgroundEffectFilter();
    
    virtual void SetInputFramebuffer(
        std::shared_ptr<GPUPixelFramebuffer> framebuffer,
        RotationMode rotation_mode = NoRotation,
        int tex_idx = 0) override;
    
    // Background mode control
    void SetBackgroundMode(BackgroundMode mode);
    BackgroundMode GetBackgroundMode() const;
    
    // Blur settings
    void SetBlurStrength(float strength);
    float GetBlurStrength() const;
    
    // Custom background
    bool LoadCustomBackground(const std::string& image_path);
    void ClearCustomBackground();
    bool HasCustomBackground() const;
    
private:
    std::shared_ptr<BackgroundProcessor> background_processor_;
    
    // Conversion helpers
    cv::Mat FramebufferToMat(std::shared_ptr<GPUPixelFramebuffer> framebuffer);
    void MatToFramebuffer(const cv::Mat& mat, std::shared_ptr<GPUPixelFramebuffer> framebuffer);
};

}  // namespace gpupixel
