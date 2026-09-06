# Install script for directory: /Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/libggml.0.17.0.dylib"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/libggml.0.dylib"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.0.17.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml.0.dylib"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      execute_process(COMMAND /usr/bin/install_name_tool
        -delete_rpath "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src"
        -delete_rpath "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/ggml-blas"
        -delete_rpath "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/ggml-metal"
        "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" -x "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/libggml.dylib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-cpu.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-alloc.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-backend.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-blas.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-cann.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-cpp.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-cuda.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-opt.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-metal.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-rpc.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-virtgpu.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-sycl.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-vulkan.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-webgpu.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-zendnn.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/ggml-openvino.h"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/libggml-base.0.17.0.dylib"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/libggml-base.0.dylib"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.0.17.0.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libggml-base.0.dylib"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" -x "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/src/libggml-base.dylib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/ggml-config.cmake"
    "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/ggml-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/Users/ernest.lee/Desktop/weftspun/3-interactor/motion-bricks-cpp/build-verify/ggml/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
