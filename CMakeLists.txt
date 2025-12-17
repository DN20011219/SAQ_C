cmake_minimum_required(VERSION 3.12)

project(SAQ_C C)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# SIMD selection: -DSIMD=AVX
set(SIMD "AUTO" CACHE STRING "SIMD to use: AUTO, NONE, AVX, NEON")
set_property(CACHE SIMD PROPERTY STRINGS AUTO NONE AVX NEON)

# Collect common sources and include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Ensure POSIX APIs like clock_gettime are declared
add_compile_definitions(_POSIX_C_SOURCE=200809L)
add_compile_definitions(_XOPEN_SOURCE=700)

# Apply SIMD-specific compile options
set(SAQ_SIMD_COMPILE_OPTIONS "")
if(SIMD STREQUAL "AVX")
  list(APPEND SAQ_SIMD_COMPILE_OPTIONS -mavx -mavx2)
elseif(SIMD STREQUAL "NEON")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    # On aarch64, NEON is baseline; no extra flags typically required
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARMv7|armv7)$")
    list(APPEND SAQ_SIMD_COMPILE_OPTIONS -mfpu=neon)
  endif()
elseif(SIMD STREQUAL "NONE")
  # No SIMD flags
else()
  # AUTO: leave it to the compiler or user env
endif()

# High-performance compiler switches (Release config only)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  list(APPEND SAQ_SIMD_COMPILE_OPTIONS -O3 -funroll-loops)
endif()

# Helper to add executable with common flags and link libs
function(saq_add_exec name)
  add_executable(${name} ${ARGN})
  if(SAQ_SIMD_COMPILE_OPTIONS)
    target_compile_options(${name} PRIVATE ${SAQ_SIMD_COMPILE_OPTIONS})
  endif()
  if(UNIX AND NOT APPLE)
    target_link_libraries(${name} m)
  endif()
endfunction()

# Binaries
saq_add_exec(estimator estimator.c)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/encoder_example.c)
  saq_add_exec(encoder_example encoder_example.c)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quantizer_example.c)
  saq_add_exec(quantizer_example quantizer_example.c)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/rotator_example.c)
  saq_add_exec(rotator_example rotator_example.c)
endif()

message(STATUS "SIMD mode: ${SIMD}")
