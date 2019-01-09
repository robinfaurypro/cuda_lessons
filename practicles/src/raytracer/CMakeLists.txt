cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(SOURCE
  src/main.cu
  src/vector3.cpp
)

add_executable(cuda_lessons_raytracer ${SOURCE})

target_compile_features(cuda_lessons_raytracer PUBLIC cxx_std_14)

set_target_properties(
  cuda_lessons_raytracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cuda_lessons_raytracer PUBLIC include)

if(APPLE)
	set_property(
    TARGET cuda_lessons_raytracer 
    PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  )
endif()
