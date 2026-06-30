cmake_minimum_required(VERSION 3.10)
project(_Navigation_RL)
set(CMAKE_CXX_STANDARD 17)

# RaiSim2 path - update this to your RaiSim2 install location
if(NOT DEFINED RAISIM2_ROOT)
    if(DEFINED ENV{RAISIM2_ROOT})
        set(RAISIM2_ROOT "$ENV{RAISIM2_ROOT}")
    else()
        set(RAISIM2_ROOT "$ENV{HOME}/raisim_ws/raisim2Lib")
    endif()
endif()

# Find RaiSim2
list(APPEND CMAKE_PREFIX_PATH ${RAISIM2_ROOT}/raisim)
find_package(raisim CONFIG REQUIRED)

# Find nanobind from RaiSim2 thirdParty
add_subdirectory(${RAISIM2_ROOT}/thirdParty/nanobind nanobind)

# Ensure Python include dirs are set for nanobind
if(NOT Python_INCLUDE_DIRS AND TARGET Python::Module)
    get_target_property(_python_module_include_dirs Python::Module INTERFACE_INCLUDE_DIRECTORIES)
    if(_python_module_include_dirs)
        set(Python_INCLUDE_DIRS "${_python_module_include_dirs}")
    endif()
endif()
if(NOT Python_INCLUDE_DIRS AND DEFINED _Python_INCLUDE_DIR)
    set(Python_INCLUDE_DIRS "${_Python_INCLUDE_DIR}")
endif()

set(Eigen3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/share/eigen3/cmake)
find_package(Eigen3 3.4.0 PATHS ${Eigen3_DIR})
find_package(OpenMP REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(resource_directory ${CMAKE_CURRENT_SOURCE_DIR}/rsc/)

file(GLOB ENV_SRCS src/env/*.hpp)

# nanobind module
nanobind_add_module(rsg_NavRL src/env/raisim_gym.cpp src/env/Yaml.cpp)
target_link_libraries(rsg_NavRL PRIVATE raisim::raisim yaml-cpp)
target_include_directories(rsg_NavRL
        PUBLIC
        ${EIGEN3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/env)
target_compile_options(rsg_NavRL PRIVATE -mtune=native -fPIC -O3)
target_compile_definitions(rsg_NavRL PRIVATE "-DRAISIMGYM_TORCH_ENV_NAME=rsg_NavRL")
set_target_properties(rsg_NavRL PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Navigation_RL/bin)

# Debug app
add_executable(debug_app src/env/debug_app.cpp ${ENV_SRCS} src/env/Yaml.cpp)
target_link_libraries(debug_app PRIVATE raisim::raisim yaml-cpp)
target_include_directories(debug_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/env ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(debug_app PRIVATE -mtune=native -fPIC -g -O0)
