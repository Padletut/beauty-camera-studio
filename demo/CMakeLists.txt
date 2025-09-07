# ---- Application project definition ----
# Main desktop demo application for GPUPixel library

# Find OpenCV for camera support
find_package(OpenCV REQUIRED)


set(GPL_APP_NAME "app")

# ---- Source files configuration ----
# add desktop application source files
set(APP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/desktop/app.cc")

# add refactored component files
list(APPEND APP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/desktop/camera_manager.cc")
list(APPEND APP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/desktop/beauty_profile_manager.cc") 
list(APPEND APP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/desktop/filter_pipeline.cc")
list(APPEND APP_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/desktop/virtual_camera.cc")

# add OpenCV face detector
list(APPEND APP_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/face_detector/opencv_face_detector.cc")

# add background effects
list(APPEND APP_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/background/person_segmentator.cc")
list(APPEND APP_SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/background/background_processor.cc")

# MediaPipe segmentation is now included in the gpupixel library

# add ImGui source files ImGui provides the GUI components for the desktop demo
file(GLOB IMGUI_SOURCE_FILES "${PROJECT_SOURCE_DIR}/third_party/imgui/*.cpp"
     "${PROJECT_SOURCE_DIR}/third_party/imgui/backends/imgui_impl_glfw.cpp"
     "${PROJECT_SOURCE_DIR}/third_party/imgui/backends/imgui_impl_opengl3.cpp")
list(APPEND APP_SOURCE_FILES ${IMGUI_SOURCE_FILES})

# ---- Build configuration ----
# create executable from source files
add_executable(${GPL_APP_NAME} ${APP_SOURCE_FILES})

# set common include directories
target_include_directories(${GPL_APP_NAME}
                           PRIVATE ${PROJECT_SOURCE_DIR}/third_party/imgui
                                   ${PROJECT_SOURCE_DIR}/src
                                   ${OpenCV_INCLUDE_DIRS})

# Link TensorFlow Lite for actual model inference
if(TARGET tensorflow-lite)
  target_link_libraries(${GPL_APP_NAME} PRIVATE tensorflow-lite)
  target_compile_definitions(${GPL_APP_NAME} PRIVATE TFLITE_AVAILABLE)
endif()

# Link FlatBuffers for TensorFlow Lite model reading
if(TARGET flatbuffers)
  target_include_directories(${GPL_APP_NAME} PRIVATE ${flatbuffers_SOURCE_DIR}/include)
  target_link_libraries(${GPL_APP_NAME} PRIVATE flatbuffers)
endif()

# ---- Platform-specific configuration ----
# Each platform requires different compiler flags and linked libraries
if(APPLE)
  # macOS specific configuration Objective-C++ compilation and proper runtime
  # library paths
  set_target_properties(
    ${GPL_APP_NAME}
    PROPERTIES COMPILE_FLAGS "-x objective-c++"
               MACOSX_BUNDLE FALSE
               INSTALL_RPATH "@executable_path/../lib"
               BUILD_WITH_INSTALL_RPATH TRUE)

  target_compile_options(${GPL_APP_NAME} PRIVATE "-fno-objc-arc")
  target_link_libraries(
    ${GPL_APP_NAME}
    PRIVATE "-framework OpenGL"
            dl
            glad::glad
            glfw::glfw
            marsface::marsface
            gpupixel::gpupixel
            ghc::filesystem)
elseif(WIN32)
  # Windows specific configuration Support delayed loading of DLLs for better
  # startup performance
  target_compile_definitions(${GPL_APP_NAME} PRIVATE WINDOWS_DLL_DELAY_LOAD)

  target_link_libraries(
    ${GPL_APP_NAME} PRIVATE opengl32 delayimp glad::glad glfw::glfw
                            gpupixel::gpupixel ghc::filesystem)

  # set delayed DLL loading
  if(MSVC)
    set_target_properties(
      ${GPL_APP_NAME}
      PROPERTIES LINK_FLAGS
                 "/DELAYLOAD:gpupixel.dll /DELAYLOAD:mars-face-kit.dll")
  endif()

else()
  # Linux specific configuration Set runtime path to find shared libraries in
  # the lib subdirectory
  set_target_properties(
    ${GPL_APP_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib"
                               BUILD_WITH_INSTALL_RPATH TRUE)

  target_link_libraries(
    ${GPL_APP_NAME}
    PRIVATE GL
            dl
            glad::glad
            glfw::glfw
            gpupixel::gpupixel
            ghc::filesystem
            stdc++fs
            ${OpenCV_LIBS})
endif()

# ---- Installation configuration ----
# Define where and how files should be installed
if(GPUPIXEL_INSTALL)
  install(
    TARGETS ${GPL_APP_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

  # Install resource files
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/desktop/demo.png
    DESTINATION bin
    COMPONENT resource)

  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/desktop/demo.png
       DESTINATION ${BUILD_OUTPUT_DIR}/bin)
endif()
