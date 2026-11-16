cmake_minimum_required(VERSION 3.20)
project(petsird_fast LANGUAGES CXX)

# the libpetsird headers (via xtensor >= 0.26) require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# yardl-generated C++ code of the PETSIRD model (conda-forge: libpetsird)
find_package(PETSIRD REQUIRED)

nanobind_add_module(_petsird_fast NB_STATIC src/petsird_fast_ext.cpp)
target_link_libraries(_petsird_fast PRIVATE petsird)

install(TARGETS _petsird_fast LIBRARY DESTINATION petsird_fast)
