cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

set(SOURCE
  src/main.cu
)

add_executable(cuda_lessons_helloworld ${SOURCE})

target_compile_features(cuda_lessons_helloworld PUBLIC cxx_std_14)

set_target_properties(
  cuda_lessons_helloworld PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

if(APPLE)
	set_property(
    TARGET cuda_lessons_helloworld 
    PROPERTY BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
  )
endif()
