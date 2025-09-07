#include "background_effect_filter.h"
#include "core/gpupixel_context.h"
#include <iostream>

namespace gpupixel {

std::shared_ptr<BackgroundEffectFilter> BackgroundEffectFilter::Create() {
    auto filter = std::shared_ptr<BackgroundEffectFilter>(new BackgroundEffectFilter());
    if (filter->background_processor_ && !filter->background_processor_->Initialize()) {
        std::cerr << "Failed to initialize background processor in filter" << std::endl;
    }
    return filter;
}

BackgroundEffectFilter::BackgroundEffectFilter() {
    background_processor_ = std::make_shared<BackgroundProcessor>();
}

BackgroundEffectFilter::~BackgroundEffectFilter() {
}

void BackgroundEffectFilter::SetInputFramebuffer(
    std::shared_ptr<GPUPixelFramebuffer> framebuffer,
    RotationMode rotation_mode,
    int tex_idx) {
    
    // For now, just pass through without any processing
    Filter::SetInputFramebuffer(framebuffer, rotation_mode, tex_idx);
}

cv::Mat BackgroundEffectFilter::FramebufferToMat(std::shared_ptr<GPUPixelFramebuffer> framebuffer) {
    if (!framebuffer) {
        return cv::Mat();
    }
    
    int width = framebuffer->GetWidth();
    int height = framebuffer->GetHeight();
    
    if (width <= 0 || height <= 0) {
        return cv::Mat();
    }
    
    // Read pixel data from framebuffer
    std::vector<uint8_t> pixel_data(width * height * 4); // RGBA
    
    // Bind framebuffer and read pixels
    framebuffer->Activate();
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_data.data());
    framebuffer->Deactivate();
    
    // Create OpenCV Mat from pixel data (RGBA from framebuffer)
    cv::Mat mat(height, width, CV_8UC4, pixel_data.data());
    cv::Mat bgr_mat;
    cv::cvtColor(mat, bgr_mat, cv::COLOR_RGBA2BGR);
    
    // Flip vertically (OpenGL coordinates are flipped)
    cv::flip(bgr_mat, bgr_mat, 0);
    
    return bgr_mat.clone(); // Clone to ensure data persistence
}

void BackgroundEffectFilter::MatToFramebuffer(const cv::Mat& mat, std::shared_ptr<GPUPixelFramebuffer> framebuffer) {
    if (mat.empty() || !framebuffer) {
        return;
    }
    
    // Convert BGR to RGBA for framebuffer upload
    cv::Mat rgba_mat;
    cv::cvtColor(mat, rgba_mat, cv::COLOR_BGR2RGBA);
    
    // Flip vertically (OpenGL coordinates are flipped)
    cv::flip(rgba_mat, rgba_mat, 0);
    
    // Upload to framebuffer texture
    framebuffer->Activate();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgba_mat.cols, rgba_mat.rows, 
                    GL_RGBA, GL_UNSIGNED_BYTE, rgba_mat.data);
    framebuffer->Deactivate();
}

void BackgroundEffectFilter::SetBackgroundMode(BackgroundMode mode) {
    if (background_processor_) {
        background_processor_->SetBackgroundMode(mode);
    }
}

BackgroundMode BackgroundEffectFilter::GetBackgroundMode() const {
    if (background_processor_) {
        return background_processor_->GetBackgroundMode();
    }
    return BackgroundMode::NONE;
}

void BackgroundEffectFilter::SetBlurStrength(float strength) {
    if (background_processor_) {
        background_processor_->SetBlurStrength(strength);
    }
}

float BackgroundEffectFilter::GetBlurStrength() const {
    if (background_processor_) {
        return background_processor_->GetBlurStrength();
    }
    return 15.0f;
}

bool BackgroundEffectFilter::LoadCustomBackground(const std::string& image_path) {
    if (background_processor_) {
        return background_processor_->LoadCustomBackground(image_path);
    }
    return false;
}

void BackgroundEffectFilter::ClearCustomBackground() {
    if (background_processor_) {
        background_processor_->ClearCustomBackground();
    }
}

bool BackgroundEffectFilter::HasCustomBackground() const {
    if (background_processor_) {
        return background_processor_->HasCustomBackground();
    }
    return false;
}

}  // namespace gpupixel
