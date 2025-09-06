/*
 * GPUPixel
 *
 * Created by PixPark on 2021/6/24.
 * Copyright © 2021 PixPark. All rights reserved.
 */

#include "core/gpupixel_context.h"
#include "utils/dispatch_queue.h"
#include "utils/logging.h"
#include "utils/util.h"
#include <thread>
#if defined(GPUPIXEL_WASM)
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

namespace gpupixel {

GPUPixelContext* GPUPixelContext::instance_ = 0;
std::mutex GPUPixelContext::mutex_;

GPUPixelContext::GPUPixelContext() : current_shader_program_(0) {
  LOG_DEBUG("Creating GPUPixelContext");
  main_thread_id_ = std::this_thread::get_id();
#if !defined(GPUPIXEL_WASM)
  task_queue_ = std::make_shared<DispatchQueue>();
#endif
  framebuffer_factory_ = new FramebufferFactory();
  Init();
}

GPUPixelContext::~GPUPixelContext() {
  LOG_DEBUG("Destroying GPUPixelContext");
  ReleaseContext();
  delete framebuffer_factory_;
  task_queue_->stop();
}

GPUPixelContext* GPUPixelContext::GetInstance() {
  if (!instance_) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!instance_) {
      instance_ = new (std::nothrow) GPUPixelContext;
    }
  }
  return instance_;
};

void GPUPixelContext::Destroy() {
  if (instance_) {
    delete instance_;
    instance_ = 0;
  }
}

void GPUPixelContext::Init() {
  LOG_INFO("Initializing GPUPixelContext");
  // Call CreateContext directly since SyncRunWithContext needs gl_context_ to be set first
  this->CreateContext();
}

FramebufferFactory* GPUPixelContext::GetFramebufferFactory() const {
  return framebuffer_factory_;
}

void GPUPixelContext::SetActiveGlProgram(GPUPixelGLProgram* shaderProgram) {
  if (current_shader_program_ != shaderProgram) {
    current_shader_program_ = shaderProgram;
    shaderProgram->UseProgram();
  }
}

void GPUPixelContext::Clean() {
  LOG_DEBUG("Cleaning GPUPixelContext resources");
  framebuffer_factory_->Clean();
}

void GPUPixelContext::CreateContext() {
#if defined(GPUPIXEL_IOS)
  LOG_DEBUG("Creating iOS OpenGL ES 2.0 context");
  egl_context_ = [[EAGLContext alloc] initWithAPI:kEAGLRenderingAPIOpenGLES2];
  if (!egl_context_) {
    LOG_ERROR("Failed to create iOS OpenGL ES 2.0 context");
    return;
  }
  [EAGLContext setCurrentContext:egl_context_];
  LOG_INFO("iOS OpenGL ES 2.0 context created successfully");
#elif defined(GPUPIXEL_MAC)
  LOG_DEBUG("Creating macOS OpenGL context");
  NSOpenGLPixelFormatAttribute pixelFormatAttributes[] = {
      NSOpenGLPFADoubleBuffer,
      NSOpenGLPFAOpenGLProfile,
      NSOpenGLProfileVersionLegacy,
      NSOpenGLPFAAccelerated,
      0,
      NSOpenGLPFAColorSize,
      24,
      NSOpenGLPFAAlphaSize,
      8,
      NSOpenGLPFADepthSize,
      24,
      0};

  pixel_format_ =
      [[NSOpenGLPixelFormat alloc] initWithAttributes:pixelFormatAttributes];
  if (!pixel_format_) {
    LOG_ERROR("Failed to create NSOpenGLPixelFormat");
    return;
  }

  image_processing_context_ =
      [[NSOpenGLContext alloc] initWithFormat:pixel_format_ shareContext:nil];
  if (!image_processing_context_) {
    LOG_ERROR("Failed to create NSOpenGLContext");
    return;
  }

  GLint interval = 0;
  [image_processing_context_ makeCurrentContext];
  [image_processing_context_ setValues:&interval
                          forParameter:NSOpenGLContextParameterSwapInterval];
  LOG_INFO("macOS OpenGL context created successfully");
#elif defined(GPUPIXEL_ANDROID)
  LOG_DEBUG("Creating Android EGL context");
  // Initialize EGL
  egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (egl_display_ == EGL_NO_DISPLAY) {
    LOG_ERROR("Failed to get EGL display");
    return;
  }

  EGLint major, minor;
  if (!eglInitialize(egl_display_, &major, &minor)) {
    LOG_ERROR("Failed to initialize EGL");
    return;
  }
  LOG_DEBUG("EGL initialized: version major:{} minor:{}", major, minor);

  // Configure EGL
  const EGLint configAttribs[] = {EGL_RED_SIZE,
                                  8,
                                  EGL_GREEN_SIZE,
                                  8,
                                  EGL_BLUE_SIZE,
                                  8,
                                  EGL_ALPHA_SIZE,
                                  8,
                                  EGL_DEPTH_SIZE,
                                  16,
                                  EGL_STENCIL_SIZE,
                                  0,
                                  EGL_SURFACE_TYPE,
                                  EGL_PBUFFER_BIT,
                                  EGL_RENDERABLE_TYPE,
                                  EGL_OPENGL_ES2_BIT,
                                  EGL_NONE};

  EGLint numConfigs;
  if (!eglChooseConfig(egl_display_, configAttribs, &egl_config_, 1,
                       &numConfigs)) {
    LOG_ERROR("Failed to choose EGL config");
    return;
  }

  // Create EGL context
  const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};

  egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT,
                                  contextAttribs);
  if (egl_context_ == EGL_NO_CONTEXT) {
    LOG_ERROR("Failed to create EGL context");
    return;
  }

  // Create offscreen rendering surface
  const EGLint pbufferAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};

  egl_surface_ =
      eglCreatePbufferSurface(egl_display_, egl_config_, pbufferAttribs);
  if (egl_surface_ == EGL_NO_SURFACE) {
    LOG_ERROR("Failed to create EGL surface");
    return;
  }

  // Set current context
  if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_)) {
    LOG_ERROR("Failed to make EGL context current");
    return;
  }
  LOG_INFO("Android EGL context created successfully");
#elif defined(GPUPIXEL_WIN) || defined(GPUPIXEL_LINUX)
  LOG_DEBUG("Creating Windows/Linux OpenGL context");
  
  // Always use existing GLFW context - don't create a new one
  GLFWwindow* existing_context = glfwGetCurrentContext();
  if (existing_context) {
    LOG_INFO("Using existing GLFW context");
    gl_context_ = existing_context;
    // Don't call glfwMakeContextCurrent here as the context is already current
  } else {
    LOG_ERROR("No existing GLFW context found. GPUPixel requires an existing OpenGL context.");
    return;
  }

  if (!gladLoadGL()) {
    LOG_ERROR("Failed to initialize GLAD");
    return;
  }
  LOG_INFO("Windows/Linux OpenGL context created successfully");
#elif defined(GPUPIXEL_WASM)
  LOG_DEBUG("Creating WebGL context");
  EmscriptenWebGLContextAttributes attrs;
  emscripten_webgl_init_context_attributes(&attrs);
  attrs.majorVersion = 2;  // Use WebGL 2.0
  attrs.minorVersion = 0;
  wasm_context_ = emscripten_webgl_create_context("#gpupixel_canvas", &attrs);
  if (wasm_context_ <= 0) {
    LOG_ERROR("Failed to create WebGL context: {}", wasm_context_);
    return;
  }
  emscripten_webgl_make_context_current(wasm_context_);
  LOG_INFO("WebGL context created successfully");
#endif
}

void GPUPixelContext::UseAsCurrent() {
#if defined(GPUPIXEL_IOS)
  if ([EAGLContext currentContext] != egl_context_) {
    LOG_TRACE("Setting current EAGLContext");
    [EAGLContext setCurrentContext:egl_context_];
  }
#elif defined(GPUPIXEL_MAC)
  if ([NSOpenGLContext currentContext] != image_processing_context_) {
    LOG_TRACE("Setting current NSOpenGLContext");
    [image_processing_context_ makeCurrentContext];
  }
#elif defined(GPUPIXEL_ANDROID)
  if (eglGetCurrentContext() != egl_context_) {
    LOG_TRACE("Setting current EGL context");
    eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_);
  }
#elif defined(GPUPIXEL_WIN) || defined(GPUPIXEL_LINUX)
  // Since we're using the existing main context, only switch if necessary
  if (glfwGetCurrentContext() != gl_context_) {
    // Check if we're in the main thread by comparing with the thread that created the context
    if (std::this_thread::get_id() == main_thread_id_) {
      LOG_TRACE("Setting current GLFW context in main thread");
      glfwMakeContextCurrent(gl_context_);
    } else {
      LOG_WARN("Skipping glfwMakeContextCurrent call from non-main thread - context ID mismatch");
    }
  } else {
    LOG_TRACE("GLFW context already current, no switch needed");
  }
#elif defined(GPUPIXEL_WASM)
  LOG_TRACE("Setting current WebGL context");
  emscripten_webgl_make_context_current(wasm_context_);
#endif
}

void GPUPixelContext::PresentBufferForDisplay() {
#if defined(GPUPIXEL_IOS)
  [egl_context_ presentRenderbuffer:GL_RENDERBUFFER];
#elif defined(GPUPIXEL_MAC)
  // No implementation needed
#elif defined(GPUPIXEL_ANDROID)
  // For offscreen rendering, no need to swap buffers
  // If display to screen is needed, use eglSwapBuffers(egl_display_,
  // egl_surface_);
#endif
}

void GPUPixelContext::ReleaseContext() {
  LOG_DEBUG("Releasing OpenGL context");
#if defined(GPUPIXEL_ANDROID)
  if (egl_display_ != EGL_NO_DISPLAY) {
    eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                   EGL_NO_CONTEXT);

    if (egl_surface_ != EGL_NO_SURFACE) {
      LOG_TRACE("Destroying EGL surface");
      eglDestroySurface(egl_display_, egl_surface_);
      egl_surface_ = EGL_NO_SURFACE;
    }

    if (egl_context_ != EGL_NO_CONTEXT) {
      LOG_TRACE("Destroying EGL context");
      eglDestroyContext(egl_display_, egl_context_);
      egl_context_ = EGL_NO_CONTEXT;
    }

    LOG_TRACE("Terminating EGL display");
    eglTerminate(egl_display_);
    egl_display_ = EGL_NO_DISPLAY;
  }
#elif defined(GPUPIXEL_WIN) || defined(GPUPIXEL_LINUX)
  // Don't destroy the context since it belongs to the main application
  LOG_TRACE("Not destroying GLFW context - managed by main application");
  gl_context_ = nullptr;
#elif defined(GPUPIXEL_WASM)
  LOG_TRACE("Destroying WebGL context");
  emscripten_webgl_destroy_context(wasm_context_);
#endif
  LOG_INFO("OpenGL context released successfully");
}

void GPUPixelContext::SyncRunWithContext(std::function<void(void)> task) {
#if defined(GPUPIXEL_IOS) || defined(GPUPIXEL_MAC)
  if (!Util::IsAppleAppActive()) {
    return;
  }
#endif

#if defined(GPUPIXEL_WASM)
  LOG_TRACE("Running task synchronously (WebGL)");
  UseAsCurrent();
  task();
#else
  // Check if we're already in the main thread
  if (std::this_thread::get_id() == main_thread_id_) {
    LOG_TRACE("Running task directly in main thread");
    UseAsCurrent();
    task();
  } else {
    LOG_TRACE("Running task on task queue");
    task_queue_->runTask([=]() {
      UseAsCurrent();
      task();
    });
  }
#endif
}
}  // namespace gpupixel
