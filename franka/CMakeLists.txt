cmake_minimum_required(VERSION 3.0.2)
project(hrii_bm2pc_rope)

add_compile_options(-std=c++17 -O2)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  geometry_msgs
  visualization_msgs
  hrii_gri_interface
  hrii_utils
  hrii_ra_interface
  hrii_robot_msgs
  tf
  tf_conversions
  tf2_ros
  moveit_msgs
  controller_interface
  franka_hw
  hardware_interface
  pluginlib
  realtime_tools
  eigen_conversions
)

find_package(matlogger2 REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

find_package(Franka 0.9.0 QUIET)
if(NOT Franka_FOUND)
  find_package(Franka 0.8.0 REQUIRED)
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rope_cartesian_pose_controller
  CATKIN_DEPENDS
    roscpp
    rospy
    roslib
    std_msgs
    geometry_msgs
    visualization_msgs
    hrii_utils
    hrii_gri_interface
    hrii_ra_interface
    hrii_robot_msgs
    tf
    tf_conversions
    tf2_ros
    moveit_msgs
    controller_interface
    franka_hw
    hardware_interface
    pluginlib
    realtime_tools
    eigen_conversions
  DEPENDS Franka
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ZMQ_INCLUDE_DIRS}
  ${Franka_INCLUDE_DIRS}
)

link_directories(
  ${ZMQ_LIBRARY_DIRS}
)

install(DIRECTORY
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(FILES rope_cartesian_pose_controller_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

add_executable(bm2pc_franka_client_node src/bm2pc_franka_client.cpp)

add_dependencies(bm2pc_franka_client_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(bm2pc_franka_client_node
  ${catkin_LIBRARIES}
  ${ZMQ_LIBRARIES}
  matlogger2::matlogger2
)

add_executable(bm2pc_franka_obstacle_client_node src/bm2pc_franka_obstacle_client.cpp)

add_dependencies(bm2pc_franka_obstacle_client_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(bm2pc_franka_obstacle_client_node
  ${catkin_LIBRARIES}
  ${ZMQ_LIBRARIES}
  matlogger2::matlogger2
)

add_executable(bm2pc_franka_pose_client_node src/bm2pc_franka_pose_client.cpp)

add_dependencies(bm2pc_franka_pose_client_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(bm2pc_franka_pose_client_node
  ${catkin_LIBRARIES}
  ${ZMQ_LIBRARIES}
  matlogger2::matlogger2
)

# --- ros_control plugin: rope_cartesian_pose_controller ---
add_library(rope_cartesian_pose_controller
  src/rope_cartesian_pose_controller.cpp
)

add_dependencies(rope_cartesian_pose_controller
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(rope_cartesian_pose_controller
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
  matlogger2::matlogger2
)

install(TARGETS
  bm2pc_franka_client_node
  bm2pc_franka_obstacle_client_node
  bm2pc_franka_pose_client_node
  rope_cartesian_pose_controller
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
