#include "virtual_camera.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <cstring>
#include <opencv2/opencv.hpp>

namespace gpupixel {

VirtualCamera::VirtualCamera() 
    : enabled_(false), fd_(-1), device_path_("/dev/video20") {
}

VirtualCamera::~VirtualCamera() {
    Shutdown();
}

bool VirtualCamera::Initialize() {
    if (IsInitialized()) {
        return true;
    }
    
    std::cout << "Initializing virtual camera..." << std::endl;
    
    // Find available loopback device
    device_path_ = FindAvailableLoopbackDevice();
    if (device_path_.empty()) {
        std::cerr << "No available v4l2loopback device found" << std::endl;
        std::cerr << "Please install v4l2loopback-utils and create a loopback device:" << std::endl;
        std::cerr << "  sudo modprobe v4l2loopback devices=1 video_nr=20 card_label=\"Virtual Camera\"" << std::endl;
        return false;
    }
    
    if (!OpenDevice()) {
        std::cerr << "Failed to open virtual camera device: " << device_path_ << std::endl;
        return false;
    }
    
    std::cout << "Virtual camera initialized: " << device_path_ << std::endl;
    return true;
}

void VirtualCamera::Shutdown() {
    CloseDevice();
}

bool VirtualCamera::WriteFrame(const unsigned char* frame_data, int width, int height, int channels) {
    if (!IsInitialized() || !enabled_) {
        return false;
    }
    
    // Convert frame format if needed (RGBA -> RGB)
    cv::Mat frame;
    if (channels == 4) {
        cv::Mat rgba_frame(height, width, CV_8UC4, const_cast<unsigned char*>(frame_data));
        cv::cvtColor(rgba_frame, frame, cv::COLOR_RGBA2RGB);
    } else if (channels == 3) {
        frame = cv::Mat(height, width, CV_8UC3, const_cast<unsigned char*>(frame_data));
    } else {
        std::cerr << "Unsupported frame format: " << channels << " channels" << std::endl;
        return false;
    }
    
    // Setup format if not already done
    static bool format_set = false;
    if (!format_set) {
        if (!SetupV4L2Format(width, height)) {
            return false;
        }
        format_set = true;
    }
    
    // Write frame to device
    size_t frame_size = frame.total() * frame.elemSize();
    ssize_t bytes_written = write(fd_, frame.data, frame_size);
    
    if (bytes_written != static_cast<ssize_t>(frame_size)) {
        std::cerr << "Failed to write complete frame to virtual camera" << std::endl;
        return false;
    }
    
    return true;
}

void VirtualCamera::Toggle() {
    if (enabled_) {
        Disable();
    } else {
        Enable();
    }
}

void VirtualCamera::Enable() {
    if (!IsInitialized()) {
        if (!Initialize()) {
            return;
        }
    }
    
    enabled_ = true;
    std::cout << "Virtual camera enabled" << std::endl;
}

void VirtualCamera::Disable() {
    enabled_ = false;
    std::cout << "Virtual camera disabled" << std::endl;
}

bool VirtualCamera::OpenDevice() {
    fd_ = open(device_path_.c_str(), O_WRONLY);
    return fd_ >= 0;
}

void VirtualCamera::CloseDevice() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
        std::cout << "Virtual camera device closed" << std::endl;
    }
}

bool VirtualCamera::SetupV4L2Format(int width, int height) {
    struct v4l2_format format = {0};
    format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
    format.fmt.pix.width = width;
    format.fmt.pix.height = height;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    format.fmt.pix.bytesperline = width * 3;
    format.fmt.pix.sizeimage = width * height * 3;
    
    if (ioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
        std::cerr << "Failed to set V4L2 format for virtual camera" << std::endl;
        return false;
    }
    
    std::cout << "Virtual camera format set: " << width << "x" << height << " RGB24" << std::endl;
    return true;
}

std::string VirtualCamera::FindAvailableLoopbackDevice() {
    // Try common loopback device paths
    std::vector<std::string> candidate_devices = {
        "/dev/video20", "/dev/video21", "/dev/video22", "/dev/video23", "/dev/video24",
        "/dev/video10", "/dev/video11", "/dev/video12", "/dev/video13", "/dev/video14"
    };
    
    for (const std::string& device : candidate_devices) {
        int test_fd = open(device.c_str(), O_WRONLY);
        if (test_fd >= 0) {
            // Check if it's a loopback device by trying to set format
            struct v4l2_capability cap;
            if (ioctl(test_fd, VIDIOC_QUERYCAP, &cap) >= 0) {
                // Check if it's a video output device
                if (cap.capabilities & V4L2_CAP_VIDEO_OUTPUT) {
                    close(test_fd);
                    std::cout << "Found available loopback device: " << device << std::endl;
                    return device;
                }
            }
            close(test_fd);
        }
    }
    
    return "";
}

} // namespace gpupixel
