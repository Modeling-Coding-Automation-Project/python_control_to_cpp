cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0148 NEW)

project(DiscreteTransferFunctionSIL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(DiscreteTransferFunctionSIL discrete_transfer_function_SIL.cpp)

target_include_directories(DiscreteTransferFunctionSIL PRIVATE
    ${CMAKE_SOURCE_DIR}/../../python_control
    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_numpy_to_cpp/python_numpy
    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_numpy_to_cpp/base_matrix
    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_math_to_cpp/base_math
    ${CMAKE_SOURCE_DIR}/../../external_libraries/python_math_to_cpp/python_math
    ${CMAKE_SOURCE_DIR}/../../external_libraries/base_utility_cpp/base_utility
    ${CMAKE_SOURCE_DIR}/../../test_sil/discrete_transfer_function
)
