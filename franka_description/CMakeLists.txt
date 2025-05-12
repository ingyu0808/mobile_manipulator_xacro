cmake_minimum_required(VERSION 3.5)
project(franka_description)

find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

install(DIRECTORY meshes robots
  DESTINATION share/${PROJECT_NAME}
)

ament_package()