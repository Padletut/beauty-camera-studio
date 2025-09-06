<h1 align="center">
  <a href="https://github.com/Padletut/beauty-camera-studio"><img src="./docs/image/cover.png"></a>
</h1>

<p align="center">
  <strong>Beauty Camera Studio</strong><br>
  <em>AI-powered beauty camera desktop application with virtual webcam support</em>
</p>

<p align="center">
  Built on <a href="https://github.com/pixpark/gpupixel">GPUPixel</a> - A high-performance image and video filter library
</p>

<p align="center">
   <a href="https://github.com/Padletut/beauty-camera-studio/stargazers"><img alt="Beauty Camera Studio Stars" src="https://img.shields.io/github/stars/Padletut/beauty-camera-studio?style=social"/></a>
    <a href="https://github.com/Padletut/beauty-camera-studio/releases/latest"><img alt="Beauty Camera Studio Release" src="https://img.shields.io/github/v/release/Padletut/beauty-camera-studio"/></a>
    <a href="#"><img alt="Platform Support" src="https://img.shields.io/badge/Platform-Linux-blue"/></a>
    <a href="https://github.com/Padletut/beauty-camera-studio/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Padletut/beauty-camera-studio"/></a>
</p>

---

> 🌟 Join us in making GPUPixel better through [discussions](https://github.com/pixpark/gpupixel/discussions), [issues](https://github.com/pixpark/gpupixel/issues/new/choose), and [PRs](https://github.com/pixpark/gpupixel/pulls).

> 📢 Note: VNN face detection library has been replaced with Mars-Face from v1.3.0-beta

## Introduction

🎥 **Beauty Camera Studio** is a professional AI-powered desktop beauty camera application for Linux with virtual webcam support.

🚀 Built on the high-performance **GPUPixel** library, featuring real-time beauty filters and AI face detection.

💻 Perfect for video calls, streaming, content creation, and any application requiring enhanced camera input.

🌐 **Virtual webcam integration** - works seamlessly with Discord, OBS, Zoom, Teams, and any application that supports video input.

---

## 🙏 **Built on GPUPixel**

This application is built on the excellent [GPUPixel](https://github.com/pixpark/gpupixel) library by pixpark. GPUPixel provides the high-performance, cross-platform image and video filter foundation that makes Beauty Camera Studio possible.

**Original GPUPixel features:**
- ⚡ High-performance C++11 and OpenGL/ES implementation
- 🌍 Cross-platform support (iOS, Android, Mac, Windows, Linux)
- 🎨 Comprehensive filter and effect system
- 📚 [Complete documentation](https://gpupixel.pixpark.net/)

**Beauty Camera Studio additions:**
- 🖥️ Complete desktop application interface
- 📷 Virtual webcam output support
- 🤖 AI face detection integration
- 💾 Profile management system
- 📦 AppImage distribution

## Effects Preview

https://github.com/user-attachments/assets/6b760fa6-e28f-4428-bfca-dec54a4e82d8

## 🎥 Beauty Camera Studio Application

Building on the GPUPixel library, this repository now includes **Beauty Camera Studio** - a complete desktop beauty camera application for Linux with virtual webcam support.

### ✨ Features
- **🎭 Real-time Beauty Effects**: Face smoothing, whitening, slimming, and eye enlargement
- **🤖 AI Face Detection**: Real-time face tracking using OpenCV and Mars-Face models
- **📷 Virtual Webcam**: Creates `/dev/video10` for use in Discord, OBS, Zoom, Teams, etc.
- **🎨 Color Grading**: Multiple LUT-based color filters (Gray, Skin, Light, Custom)
- **⚙️ Camera Controls**: Resolution, brightness, contrast, saturation, zoom, focus
- **💾 Profile System**: Save and load beauty presets
- **📦 Portable Distribution**: Ready-to-use AppImage for any Linux distro

### 🚀 Quick Start (Linux)
```bash
# Clone and build
git clone https://github.com/Padletut/beauty-camera-studio.git
cd beauty-camera-studio
./script/build_linux.sh

# Create AppImage (compact)
cd appimage
./build-appimage.sh

# OR create AppImage with LinuxDeploy (better compatibility)
./build-appimage-linuxdeploy.sh

# Run Beauty Camera Studio
./output/bin/app  # or use the AppImage
```

### 📋 Virtual Camera Setup
```bash
# Install v4l2loopback for virtual camera support
sudo apt install v4l2loopback-dkms  # Ubuntu/Debian
sudo dnf install v4l2loopback        # Fedora
sudo modprobe v4l2loopback video_nr=10 card_label="Virtual Camera 10"

# Now select "Virtual Camera 10" in Discord, OBS, etc.
```

### 🎯 System Requirements

#### 📦 **AppImage Distribution (Recommended)**
- **OS**: Linux (x86_64) with GLIBC 2.38+ 
  - ✅ **Ubuntu 24.04+**, Fedora 40+, Arch (recent), OpenSUSE Tumbleweed
  - ❌ **Ubuntu 20.04/22.04** (GLIBC 2.31/2.35 - use manual build instead)
- **Graphics**: OpenGL support (any modern desktop)
- **Camera**: USB webcam or built-in camera
- **Virtual Camera**: `v4l2loopback-dkms` package

#### 🔧 **Manual Build (Maximum Compatibility)**
- **OS**: Linux (x86_64) - Ubuntu 18.04+, Fedora 30+, Arch, etc.
- **Build Tools**: CMake, GCC/Clang, pkg-config
- **Dependencies**: OpenCV, v4l2loopback (auto-installed during build)
 
## Before You Start
⭐ Star us on GitHub for notifications about new releases!

![](./docs/image/give-star.gif)

 
## Getting Started

### 🏗️ **For Beauty Camera Studio Users**
Ready-to-use desktop beauty camera application:
1. **Build**: `./script/build_linux.sh`
2. **Run**: `./output/bin/app`
3. **AppImage (Compact)**: `./appimage/build-appimage.sh` for 3.6MB portable package
4. **AppImage (Compatible)**: `./appimage/build-appimage-linuxdeploy.sh` for 111MB with all dependencies
5. **Virtual Camera**: Install `v4l2loopback` and select "Virtual Camera 10" in your apps

### 🛠️ **For GPUPixel Library Development**
To use the underlying GPUPixel library in your own projects:
- 📖 See the original [GPUPixel documentation](https://gpupixel.pixpark.net/)
- 🔧 [Build Guide](https://gpupixel.pixpark.net/guide/build)
- 🎮 [Demo Examples](https://gpupixel.pixpark.net/guide/demo) 
- 🔗 [Integration Guide](https://gpupixel.pixpark.net/guide/integrated)




## Contributing

🤝 Improve GPUPixel by joining [discussions](https://github.com/pixpark/gpupixel/discussions), opening [issues](https://github.com/pixpark/gpupixel/issues/new/choose), or submitting [PRs](https://github.com/pixpark/gpupixel/pulls). See our [Contributing Guide](docs/docs/en/guide/contributing.md) to get started.

Consider sharing GPUPixel on social media and at events.

## Contributors
 [![](https://opencollective.com/gpupixel/contributors.svg?width=890&button=false)](https://github.com/pixpark/gpupixel/graphs/contributors)

## Sponsorship
💖 Support this project through:

| [☕ Support me on Ko-fi](docs/docs/en/sponsor.md#ko-fi) | [💝 Support on Open Collective](docs/docs/en/sponsor.md#open-collective) | [💰 WeChat Sponsor](docs/docs/en/sponsor.md#wechat) |
|:---:|:---:|:---:|

## Sponsors

🙏 Thanks to these contributors for their generous support:

<a href="https://github.com/leavenotrace">
  <picture>
    <img src="https://github.com/leavenotrace.png" width="50" height="50" style="border-radius: 50%;" alt="@leavenotrace">
  </picture>
</a>
<a href="https://github.com/weiyu666">
  <picture>
    <img src="https://github.com/weiyu666.png" width="50" height="50" style="border-radius: 50%;" alt="@weiyu666">
  </picture>
</a>
<a href="https://github.com/lambiengcode">
  <picture>
    <img src="https://github.com/lambiengcode.png" width="50" height="50" style="border-radius: 50%;" alt="@lambiengcode">
  </picture>
</a>

## Contact & Support
- 📚 [Docs](https://gpupixel.pixpark.net/): Documentation
- 🐛 [Issues](https://github.com/pixpark/gpupixel/issues/new/choose): Bug reports and feature requests
- 📧 [Email](mailto:jaaronkot@gmail.com?subject=[GitHub]Questions%20About%20GPUPixel): Contact us
- 📞 [Contact](docs/docs/en/about/contact.md): More ways to connect

## Acknowledgements
### 🔗 Reference Projects
1. [GPUImage](https://github.com/BradLarson/GPUImage) 
2. [GPUImage-x](https://github.com/wangyijin/GPUImage-x)
3. [CainCamera](https://github.com/CainKernel/CainCamera)
4. [VNN](https://github.com/joyycom/VNN)

## License
This repository is available under the [Apache-2.0 License](https://github.com/pixpark/gpupixel?tab=Apache-2.0-1-ov-file).

