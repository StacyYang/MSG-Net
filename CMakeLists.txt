##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## Free to reuse and distribute this software for research or 
## non-profit purpose, subject to the following conditions:
##  1. The code must retain the above copyright notice, this list of
##     conditions.
##  2. Original authors' names are not deleted.
##  3. The authors' names are not used to endorse or promote products
##      derived from this software 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR) 
CMAKE_POLICY(VERSION 2.8)

# Find Torch and CUDA
FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(CUDA 6.5 REQUIRED)

# Detect CUDA architecture and get best NVCC flags
IF(NOT COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
  INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
ENDIF()
CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA $ENV{TORCH_CUDA_ARCH_LIST})
LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

INCLUDE_DIRECTORIES(
	./include
	${CMAKE_CURRENT_SOURCE_DIR} 
	"${Torch_INSTALL_INCLUDE}/THC" 
)
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

# ADD lua source files
FILE(GLOB src *.c *.cu) 
FILE(GLOB luasrc *.lua modules/*.lua) 

# ADD the torch package and link denpendencies
ADD_TORCH_PACKAGE(texture "" "${luasrc}")
TARGET_LINK_LIBRARIES(texture 
)
