#pragma once

#include <string>
#include <map>
#include <opencv2/opencv.hpp>

namespace gpupixel {

struct BeautyProfile {
    std::string name;
    
    // Beauty filter parameters
    float beauty = 15.0f;
    float whitening = 0.0f;
    float face_slim = 0.0f;
    float eye_enlarge = 0.0f;
    float color_tint = 0.0f;
    float warmth = 0.0f;
    
    // Camera settings
    float camera_brightness = 0.0f;
    float camera_contrast = 50.0f;
    float camera_saturation = 50.0f;
    float camera_gain = 50.0f;
    float camera_sharpness = 50.0f;
    float camera_zoom = 100.0f;
    bool camera_auto_focus = true;
    bool camera_auto_gain = true;
    cv::Size resolution{1920, 1080};
    
    // AI settings
    float ai_confidence_threshold = 0.1f;
    float ai_temporal_smoothing = 0.7f;
};

class BeautyProfileManager {
public:
    BeautyProfileManager();
    ~BeautyProfileManager();
    
    // Profile management
    void SaveProfile(const std::string& name, const BeautyProfile& profile);
    bool LoadProfile(const std::string& name, BeautyProfile& profile);
    void DeleteProfile(const std::string& name);
    
    // File I/O
    bool SaveProfilesToFile();
    bool LoadProfilesFromFile();
    
    // Default profile management
    void SetDefaultProfile(const std::string& name);
    const std::string& GetDefaultProfileName() const { return default_profile_name_; }
    bool GetDefaultProfile(BeautyProfile& profile) const;
    
    // Getters
    const std::map<std::string, BeautyProfile>& GetAllProfiles() const { return profiles_; }
    std::vector<std::string> GetProfileNames() const;
    bool HasProfile(const std::string& name) const;
    
    // Create built-in profiles
    void CreateBuiltInProfiles();
    
private:
    std::map<std::string, BeautyProfile> profiles_;
    std::string default_profile_name_;
    
    std::string GetProfilesDirectory();
    std::string GetProfilesFilePath();
};

} // namespace gpupixel
