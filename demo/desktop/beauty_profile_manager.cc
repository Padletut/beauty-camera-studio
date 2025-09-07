#include "beauty_profile_manager.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <unistd.h>

namespace gpupixel {

BeautyProfileManager::BeautyProfileManager() 
    : default_profile_name_("Default") {
    CreateBuiltInProfiles();
    LoadProfilesFromFile();
}

BeautyProfileManager::~BeautyProfileManager() {
    SaveProfilesToFile();
}

void BeautyProfileManager::SaveProfile(const std::string& name, const BeautyProfile& profile) {
    BeautyProfile new_profile = profile;
    new_profile.name = name;
    profiles_[name] = new_profile;
    std::cout << "Beauty profile '" << name << "' saved" << std::endl;
}

bool BeautyProfileManager::LoadProfile(const std::string& name, BeautyProfile& profile) {
    auto it = profiles_.find(name);
    if (it != profiles_.end()) {
        profile = it->second;
        std::cout << "Beauty profile '" << name << "' loaded" << std::endl;
        return true;
    }
    std::cerr << "Beauty profile '" << name << "' not found" << std::endl;
    return false;
}

void BeautyProfileManager::DeleteProfile(const std::string& name) {
    auto it = profiles_.find(name);
    if (it != profiles_.end()) {
        profiles_.erase(it);
        std::cout << "Beauty profile '" << name << "' deleted" << std::endl;
        
        // If we deleted the default profile, reset to "Default"
        if (default_profile_name_ == name) {
            default_profile_name_ = "Default";
        }
    }
}

bool BeautyProfileManager::SaveProfilesToFile() {
    std::string profiles_dir = GetProfilesDirectory();
    
    // Create directory if it doesn't exist
    struct stat st = {0};
    if (stat(profiles_dir.c_str(), &st) == -1) {
        if (mkdir(profiles_dir.c_str(), 0755) != 0) {
            std::cerr << "Failed to create profiles directory: " << profiles_dir << std::endl;
            return false;
        }
    }
    
    std::string filepath = GetProfilesFilePath();
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to save profiles to: " << filepath << std::endl;
        return false;
    }
    
    // Write default profile name
    file << "default_profile=" << default_profile_name_ << std::endl;
    
    // Write each profile
    for (const auto& pair : profiles_) {
        const BeautyProfile& profile = pair.second;
        file << "[" << pair.first << "]" << std::endl;
        file << "beauty=" << profile.beauty << std::endl;
        file << "whitening=" << profile.whitening << std::endl;
        file << "face_slim=" << profile.face_slim << std::endl;
        file << "eye_enlarge=" << profile.eye_enlarge << std::endl;
        file << "color_tint=" << profile.color_tint << std::endl;
        file << "warmth=" << profile.warmth << std::endl;
        file << "camera_brightness=" << profile.camera_brightness << std::endl;
        file << "camera_contrast=" << profile.camera_contrast << std::endl;
        file << "camera_saturation=" << profile.camera_saturation << std::endl;
        file << "camera_gain=" << profile.camera_gain << std::endl;
        file << "camera_sharpness=" << profile.camera_sharpness << std::endl;
        file << "camera_zoom=" << profile.camera_zoom << std::endl;
        file << "camera_auto_focus=" << (profile.camera_auto_focus ? 1 : 0) << std::endl;
        file << "camera_auto_gain=" << (profile.camera_auto_gain ? 1 : 0) << std::endl;
        file << "resolution_width=" << profile.resolution.width << std::endl;
        file << "resolution_height=" << profile.resolution.height << std::endl;
        file << "ai_confidence_threshold=" << profile.ai_confidence_threshold << std::endl;
        file << "ai_temporal_smoothing=" << profile.ai_temporal_smoothing << std::endl;
        file << std::endl;
    }
    
    file.close();
    std::cout << "Beauty profiles saved to: " << filepath << std::endl;
    return true;
}

bool BeautyProfileManager::LoadProfilesFromFile() {
    std::string filepath = GetProfilesFilePath();
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cout << "No existing profiles file found, using defaults" << std::endl;
        return false;
    }
    
    std::cout << "Loading beauty profiles from: " << filepath << std::endl;
    
    std::string line;
    std::string current_profile_name;
    BeautyProfile current_profile;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Check for default profile setting
        if (line.find("default_profile=") == 0) {
            default_profile_name_ = line.substr(15);
            continue;
        }
        
        // Check for profile section start
        if (line[0] == '[' && line.back() == ']') {
            // Save previous profile if we have one
            if (!current_profile_name.empty()) {
                current_profile.name = current_profile_name;
                profiles_[current_profile_name] = current_profile;
                std::cout << "  Loaded profile: " << current_profile_name << std::endl;
            }
            
            // Start new profile
            current_profile_name = line.substr(1, line.length() - 2);
            current_profile = BeautyProfile(); // Reset to defaults
            continue;
        }
        
        // Parse key=value pairs
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);
            
            // Parse profile values
            if (key == "beauty") current_profile.beauty = std::stof(value);
            else if (key == "whitening") current_profile.whitening = std::stof(value);
            else if (key == "face_slim") current_profile.face_slim = std::stof(value);
            else if (key == "eye_enlarge") current_profile.eye_enlarge = std::stof(value);
            else if (key == "color_tint") current_profile.color_tint = std::stof(value);
            else if (key == "warmth") current_profile.warmth = std::stof(value);
            else if (key == "camera_brightness") current_profile.camera_brightness = std::stof(value);
            else if (key == "camera_contrast") current_profile.camera_contrast = std::stof(value);
            else if (key == "camera_saturation") current_profile.camera_saturation = std::stof(value);
            else if (key == "camera_gain") current_profile.camera_gain = std::stof(value);
            else if (key == "camera_sharpness") current_profile.camera_sharpness = std::stof(value);
            else if (key == "camera_zoom") current_profile.camera_zoom = std::stof(value);
            else if (key == "camera_auto_focus") current_profile.camera_auto_focus = (std::stoi(value) != 0);
            else if (key == "camera_auto_gain") current_profile.camera_auto_gain = (std::stoi(value) != 0);
            else if (key == "resolution_width") current_profile.resolution.width = std::stoi(value);
            else if (key == "resolution_height") current_profile.resolution.height = std::stoi(value);
            else if (key == "ai_confidence_threshold") current_profile.ai_confidence_threshold = std::stof(value);
            else if (key == "ai_temporal_smoothing") current_profile.ai_temporal_smoothing = std::stof(value);
        }
    }
    
    // Save the last profile
    if (!current_profile_name.empty()) {
        current_profile.name = current_profile_name;
        profiles_[current_profile_name] = current_profile;
        std::cout << "  Loaded profile: " << current_profile_name << std::endl;
    }
    
    file.close();
    std::cout << "Loaded " << profiles_.size() << " beauty profiles" << std::endl;
    return true;
}

void BeautyProfileManager::SetDefaultProfile(const std::string& name) {
    if (profiles_.find(name) != profiles_.end()) {
        default_profile_name_ = name;
        std::cout << "Default profile set to: " << name << std::endl;
    } else {
        std::cerr << "Cannot set default profile: '" << name << "' does not exist" << std::endl;
    }
}

bool BeautyProfileManager::GetDefaultProfile(BeautyProfile& profile) const {
    auto it = profiles_.find(default_profile_name_);
    if (it != profiles_.end()) {
        profile = it->second;
        return true;
    }
    return false;
}

std::vector<std::string> BeautyProfileManager::GetProfileNames() const {
    std::vector<std::string> names;
    for (const auto& pair : profiles_) {
        names.push_back(pair.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

bool BeautyProfileManager::HasProfile(const std::string& name) const {
    return profiles_.find(name) != profiles_.end();
}

void BeautyProfileManager::CreateBuiltInProfiles() {
    // Default profile
    BeautyProfile default_profile;
    default_profile.name = "Default";
    default_profile.beauty = 15.0f;
    default_profile.whitening = 0.0f;
    default_profile.face_slim = 0.0f;
    default_profile.eye_enlarge = 0.0f;
    default_profile.color_tint = 0.0f;
    default_profile.warmth = 0.0f;
    default_profile.camera_brightness = 0.0f;
    default_profile.camera_contrast = 50.0f;
    default_profile.camera_saturation = 50.0f;
    default_profile.camera_gain = 50.0f;
    default_profile.camera_sharpness = 50.0f;
    default_profile.camera_zoom = 100.0f;
    default_profile.camera_auto_focus = true;
    default_profile.camera_auto_gain = true;
    default_profile.resolution = cv::Size(1920, 1080);
    default_profile.ai_confidence_threshold = 0.1f;
    default_profile.ai_temporal_smoothing = 0.7f;
    profiles_["Default"] = default_profile;
    
    // Discord profile (optimized for video calls)
    BeautyProfile discord_profile = default_profile;
    discord_profile.name = "Discord";
    discord_profile.beauty = 15.312f;
    discord_profile.camera_brightness = -31.25f;
    discord_profile.camera_saturation = 50.52f;
    discord_profile.camera_sharpness = 31.771f;
    profiles_["Discord"] = discord_profile;
    
    std::cout << "Created built-in beauty profiles" << std::endl;
}

std::string BeautyProfileManager::GetProfilesDirectory() {
    char buffer[1024];
    if (readlink("/proc/self/exe", buffer, sizeof(buffer) - 1) != -1) {
        std::string exe_path(buffer);
        size_t last_slash = exe_path.find_last_of('/');
        if (last_slash != std::string::npos) {
            return exe_path.substr(0, last_slash) + "/profiles";
        }
    }
    return "./profiles";
}

std::string BeautyProfileManager::GetProfilesFilePath() {
    return GetProfilesDirectory() + "/beauty_profiles.ini";
}

} // namespace gpupixel
