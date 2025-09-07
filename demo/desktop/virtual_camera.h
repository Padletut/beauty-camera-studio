#pragma once

#include <string>

namespace gpupixel {

class VirtualCamera {
public:
    VirtualCamera();
    ~VirtualCamera();
    
    // Virtual camera management
    bool Initialize();
    void Shutdown();
    bool IsEnabled() const { return enabled_; }
    
    // Frame writing
    bool WriteFrame(const unsigned char* frame_data, int width, int height, int channels);
    
    // Control
    void Toggle();
    void Enable();
    void Disable();
    
    // Status
    const std::string& GetDevicePath() const { return device_path_; }
    bool IsInitialized() const { return fd_ >= 0; }
    
private:
    bool enabled_;
    int fd_;
    std::string device_path_;
    
    // Helper methods
    bool OpenDevice();
    void CloseDevice();
    bool SetupV4L2Format(int width, int height);
    std::string FindAvailableLoopbackDevice();
};

} // namespace gpupixel
