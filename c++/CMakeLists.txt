project(shape_matching_dd)

cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# make project executable
if( BUILD_PYTHON_BINDINGS )
    message("Building Python Bindings")
    set(PROJECT_NAME shape_match_model_pb)
else()
    message("Building CPP Executable")
    add_executable(${PROJECT_NAME} main.cpp)
endif()

# python bindings
add_subdirectory(external/pybind11)
pybind11_add_module(shape_match_model_pb shapeMatchModel/shapeMatchModelPB.cpp)
find_package(Python3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC external/pybind11/include)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_TRIANGLE "Use Triangle" ON)
target_include_directories(${PROJECT_NAME} PUBLIC external/libigl/include)
target_include_directories(${PROJECT_NAME} PUBLIC external/libigl/external/glad/include)
target_include_directories(${PROJECT_NAME} PUBLIC external/libigl/external/glfw/include)
find_package(LIBIGL REQUIRED QUIET)

# eigen (header only library)
target_include_directories(${PROJECT_NAME} PUBLIC external/eigen)

# tri-mesh-hole-closer (header only library)
add_subdirectory(external/tri-mesh-hole-closer)

# robin-map (header only library)
target_include_directories(${PROJECT_NAME} PUBLIC external/robin-map)

# bdd (dual solver)
target_include_directories(${PROJECT_NAME} PUBLIC external/bdd/include/)
add_subdirectory(external/bdd)


# open mp (for parallel execution)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
target_link_libraries(${PROJECT_NAME} PRIVATE
            LPMP-BDD
            igl::core
            igl::opengl_glfw
            OpenMP::OpenMP_CXX
            tri-mesh-hole-closer
            )
else()
target_link_libraries(${PROJECT_NAME} PRIVATE
			LPMP-BDD
			igl::core
			igl::opengl_glfw
			tri-mesh-hole-closer
			)
endif()


# set project include dir
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR})

# project subdirectories
add_subdirectory(helper)
add_subdirectory(test)
add_subdirectory(shapeMatchModel)


