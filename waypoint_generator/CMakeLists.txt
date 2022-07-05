cmake_minimum_required(VERSION 3.0.2)
project(waypoint_generator)

find_package(catkin REQUIRED COMPONENTS
  rospy geometry_msgs sensor_msgs std_msgs move_base_msgs nav_msgs
)
catkin_package(

  # INCLUDE_DIRS include
  # LIBRARIES autorace
  CATKIN_DEPENDS geometry_msgs rospy sensor_msgs std_msgs move_base_msgs nav_msgs

  # DEPENDS system_lib
)
catkin_install_python(PROGRAMS
  scripts/corner_handler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(

  # include
  ${catkin_INCLUDE_DIRS}
)
