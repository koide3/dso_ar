cmake_minimum_required(VERSION 2.8.3)
project(dso_ar)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  pcl_ros
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ccf_feature_extraction
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
 ${catkin_INCLUDE_DIRS}
)

add_executable(plane_estimation_node
  src/plane_estimation_node.cpp
)
target_link_libraries(plane_estimation_node
  ${catkin_LIBRARIES}
)
