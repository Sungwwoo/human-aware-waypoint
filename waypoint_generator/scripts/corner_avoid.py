import rospy
import os
from waypoint_generator.corner_handler import CornerHandler

if __name__ == "__main__":
    rospy.init_node("human_aware_corner_avoid")
    cornerHandler = CornerHandler()
    cornerHandler.prepare()

    while not rospy.is_shutdown():
        cornerHandler.getRobotLocation()
        rospy.sleep(1)
