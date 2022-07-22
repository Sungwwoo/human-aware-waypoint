import rospy
from waypoint_generator.corner_handler import CornerHandler

if __name__ == "__main__":
    rospy.init_node("human_aware_corner_avoid")
    cornerHandler = CornerHandler()
    if not cornerHandler.extractCorners():
        exit()

    while not rospy.is_shutdown():
        cornerHandler.checkDistance()
        rospy.sleep(0.3)
