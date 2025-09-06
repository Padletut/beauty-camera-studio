<h1 align="center">
  <a href="https://github.com/pixpark/gpupixel"><img src="./docs/image/cover.png"></a>
</h1>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://gpupixel.pixpark.net/" target="_blank">Doc</a>
  <span> · </span>
  <a href="https://gpupixel.pixpark.net/zh" target="_blank">文档</a>
</p>

<p align="center">
   <a href="https://github.com/pixpark/gpupixel/stargazers"><img alt="GPUPixel Stars" src="https://img.shields.io/github/stars/pixpark/gpupixel?style=social"/></a>
    <a href="https://github.com/pixpark/gpupixel/releases/latest"><img alt="GPUPixel Release" src="https://img.shields.io/github/v/release/pixpark/gpupixel"/></a>
    <a href="#"><img alt="GPUPixel Stars" src="https://img.shields.io/badge/Platform-iOS_%7C_Android_%7C_Mac_%7C_Win_%7C_Linux-red"/></a>
     <a href="https://github.com/pixpark/gpupixel/actions/workflows/build.yml"><img src="https://github.com/pixpark/gpupixel/actions/workflows/build.yml/badge.svg"></a>
    <a href="https://github.com/pixpark/gpupixel/blob/main/LICENSE"><img alt="GPUPixel Stars" src="https://img.shields.io/github/license/pixpark/gpupixel"/></a>
</p>

<p align="center">
<a href="https://discord.gg/q2MjmqK4" target="_blank"><img alt="GPUPixel Discord" src="https://img.shields.io/badge/Chat-Discord-blue?logo=discord&logoColor=white&labelColor=grey&color=blue"/></a>
<a href="https://gpupixel.pixpark.net/about/contact#qq-group" target="_blank"><img alt="QQ Group" src="https://img.shields.io/badge/-QQ群-gray?logo=qq&logoColor=white&labelColor=gray&color=blue&style=flat"/></a>
<a href="https://gpupixel.pixpark.net/about/contact#wechat-official-account" target="_blank"><img alt="GPUPixel Wechat" src="https://img.shields.io/badge/-公众号-gray?logo=wechat&logoColor=white&labelColor=gray&color=07C160&style=flat"/></a>
<a href="https://github.com/pixpark/gpupixel#Sponsorship" target="_blank"><img alt="Sponsor" src="https://img.shields.io/badge/-Sponsor-gray?logo=githubsponsors&logoColor=white&labelColor=grey&color=FE6AB2&style=flat"/></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/7103" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7103" alt="pixpark%2Fgpupixel | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

---

> 🌟 Join us in making GPUPixel better through [discussions](https://github.com/pixpark/gpupixel/discussions), [issues](https://github.com/pixpark/gpupixel/issues/new/choose), and [PRs](https://github.com/pixpark/gpupixel/pulls).

> 📢 Note: VNN face detection library has been replaced with Mars-Face from v1.3.0-beta

## Introduction

🚀 A high-performance, cross-platform image and video filter library with a small footprint.

💻 Built with C++11 and OpenGL/ES, featuring beauty filters.

🌐 Supports iOS, Android, Mac, Windows, and Linux—compatible with any OpenGL/ES platform.

🎥 **NEW**: Complete Beauty Camera Studio desktop application with virtual webcam support for Discord, OBS, and more!

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
git clone https://github.com/pixpark/gpupixel.git
cd gpupixel
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
- **OS**: Linux (x86_64) - Ubuntu 18.04+, Fedora 30+, Arch, etc.
- **Graphics**: OpenGL support (any modern desktop)
- **Camera**: USB webcam or built-in camera
- **Dependencies**: OpenCV, v4l2loopback (for virtual camera)
 
## Before You Start
⭐ Star us on GitHub for notifications about new releases!

![](./docs/image/give-star.gif)

 
## Getting Started

### 🏗️ Library Development
🔍 See the docs: [Introduction](https://gpupixel.pixpark.net/guide/build) | [Build](https://gpupixel.pixpark.net/guide/build) | [Demo](https://gpupixel.pixpark.net/guide/demo) | [Integration](https://gpupixel.pixpark.net/guide/integrated)

### 💻 Beauty Camera Studio Application
For the complete desktop beauty camera app:
1. **Build**: `./script/build_linux.sh`
2. **Run**: `./output/bin/app`
3. **AppImage (Compact)**: `./appimage/build-appimage.sh` for 3.6MB portable package
4. **AppImage (Compatible)**: `./appimage/build-appimage-linuxdeploy.sh` for 111MB with all dependencies
5. **Virtual Camera**: Install `v4l2loopback` and select "Virtual Camera 10" in your apps




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

