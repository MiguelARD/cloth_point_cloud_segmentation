cmake_minimum_required(VERSION 3.0.2)
project(cloth_segmentation)

find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  darknet_ros_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
)


find_package(Boost REQUIRED COMPONENTS system)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)

catkin_package(
  DEPENDS 
  roscpp
  std_msgs
  sensor_msgs
  darknet_ros_msgs
  cv_bridge
)


add_executable(cloth_segmentation src/cloth_segmentation.cpp)
add_dependencies(cloth_segmentation ${catkin_EXPORTED_TARGETS} ${${PROJECT_NAME}_EXPORTED_TARGETS})
target_link_libraries(cloth_segmentation ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
