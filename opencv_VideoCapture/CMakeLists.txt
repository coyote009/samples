cmake_minimum_required( VERSION 2.8 )

find_package( OpenCV 3 REQUIRED )

set( TARGET opencv_VideoCapture )

file( GLOB SRCS src/*.cpp )

add_executable( ${TARGET} ${SRCS} )
target_link_libraries( ${TARGET} opencv_core opencv_highgui opencv_videoio )
