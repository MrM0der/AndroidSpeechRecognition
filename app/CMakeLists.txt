cmake_minimum_required(VERSION 3.4.1)

set(LIB_DIR ${CMAKE_BINARY_DIR})

set(APP_NAME speech_recognizer)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
        onnxruntime
        PROPERTIES IMPORTED_LOCATION
        ${LIB_DIR}/libonnxruntime.so
)
add_library(custom_op_library SHARED IMPORTED)
set_target_properties(
        custom_op_library
        PROPERTIES IMPORTED_LOCATION
        ${LIB_DIR}/libcustom_op_library.so
)

file(GLOB_RECURSE CPP_SRCS src/main/jni/*.cpp)
add_library(${APP_NAME} SHARED ${CPP_SRCS})

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

add_definitions(-DAPP_NAME=${APP_NAME})

set_property(TARGET ${APP_NAME} PROPERTY CXX_STANDARD 17)
target_link_libraries(${APP_NAME} onnxruntime custom_op_library ${log-lib} ${jnigraphics-lib})