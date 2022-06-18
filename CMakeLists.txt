cmake_minimum_required(VERSION 3.16)
project(stereo-camera CXX)

# OpenCV
find_package(OpenCV REQUIRED)
add_library(OpenCV INTERFACE)
target_include_directories(OpenCV INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV INTERFACE ${OpenCV_LIBS})

# Eigen
add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE "${Eigen_INCLUDE_DIR}")

# OpenMP
find_package(OpenMP REQUIRED)

add_executable(main
    src/main.cpp
    src/util.cpp
    src/compute_cost.cpp
    src/bilateral_filter.cpp
    src/choose_disparity.cpp
    src/refine_disparity.cpp
    src/construct_graph.cpp
    src/stereo_rectification.cpp
)
    src/segment_tree.cpp
    )

target_link_libraries(main OpenCV Eigen OpenMP::OpenMP_CXX)

add_custom_target(run VERBATIM COMMAND main WORKING_DIRECTORY ${stereo-camera_SOURCE_DIR})

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.21)
    add_custom_target(copy-dependency ALL COMMAND_EXPAND_LISTS VERBATIM
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:main> $<TARGET_FILE_DIR:main>)
endif()

set_property(DIRECTORY "${PROJECT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT main)

set_target_properties(main PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    VS_DEBUGGER_COMMAND_ARGUMENTS "data/artroom1")

if(MSVC)
    target_compile_options(main PRIVATE
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>: /utf-8 >
        $<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>: -Xcompiler=/utf-8>
        $<$<BUILD_INTERFACE:$<CONFIG:Release,RelWithDebInfo>>: /Ob3>
    )
endif()

target_compile_features(main PRIVATE cxx_std_17)
