cmake_minimum_required(VERSION 3.16)
project(stereo-camera CXX CUDA)

# OpenCV
find_package(OpenCV REQUIRED)
add_library(OpenCV INTERFACE)
target_include_directories(OpenCV INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV INTERFACE ${OpenCV_LIBS})

# Eigen
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE "${Eigen_INCLUDE_DIR}")

add_executable(main src/main.cpp src/kernel.cu)

target_link_libraries(main OpenCV Eigen)

add_custom_target(run VERBATIM COMMAND main WORKING_DIRECTORY ${stereo-camera_SOURCE_DIR})

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.21)
    add_custom_target(copy-dependency ALL COMMAND_EXPAND_LISTS VERBATIM
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:main> $<TARGET_FILE_DIR:main>)
endif()

set_property(DIRECTORY "${PROJECT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT main)

if(MSVC)
    target_compile_options(main PRIVATE
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>: /utf-8 >
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>: -Xcompiler=/utf-8>
    )
endif()