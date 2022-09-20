#! /usr/bin/env python3

import time
import cv2 as cv
import rospy
import message_filters
import tf2_ros
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from math import sqrt
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from visualization_msgs.msg import Marker, MarkerArray
from pedsim_msgs.msg import TrackedPersons
from actionlib import SimpleActionClient
import matplotlib.pyplot as plt

corners = []
inRangeCorners = []
g_path = []
g_goal = []
markerArray = MarkerArray()
content = []


def getTF(target_frame, source_frame):

    while True:
        try:
            trans = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
            return trans
        except:
            rospy.sleep(1)
            continue


def cbOdom(odom):
    t = (rospy.Time.now() - start).to_sec()
    vel_file.write(str(t) + "\t" + str(odom.twist.twist.linear.x) + "\t" + str(odom.twist.twist.angular.z) + "\n")


if __name__ == "__main__":
    rospy.init_node("global_path_listner")
    # tf listner
    tfBuffer = tf2_ros.Buffer()
    tfListner = tf2_ros.TransformListener(tfBuffer)
    client = SimpleActionClient("move_base", MoveBaseAction)
    client.wait_for_server()

    # start point: 0, 6
    targetX, targetY = 6, 12
    theta = 0
    base_goal_quat = quaternion_from_euler(0, 0, theta)

    print("Moving to ", targetX, targetY)
    base_goal = MoveBaseGoal()
    base_goal.target_pose.header.frame_id = "map"
    base_goal.target_pose.header.stamp = rospy.Time.now()
    base_goal.target_pose.pose.position.x = targetX
    base_goal.target_pose.pose.position.y = targetY
    base_goal.target_pose.pose.position.z = 0
    base_goal.target_pose.pose.orientation.x = base_goal_quat[0]
    base_goal.target_pose.pose.orientation.y = base_goal_quat[1]
    base_goal.target_pose.pose.orientation.z = base_goal_quat[2]
    base_goal.target_pose.pose.orientation.w = base_goal_quat[3]

    rospy.loginfo("Node initialized")

    prevLoc, prevPLoc = None, None

    lctime = time.localtime(time.time())
    sim_name = "{0}{1}{2}-{3}{4}{5}".format(lctime[0], lctime[1], lctime[2], lctime[3], lctime[4], lctime[5])
    vel_file = open("velocity_logs/" + sim_name + ".txt", "w")
    sub_odom = rospy.Subscriber("/husky_velocity_controller/odom", Odometry, cbOdom)

    start = rospy.Time.now()
    client.send_goal(base_goal)

    try:
        while not rospy.is_shutdown():
            rospy.spin()
    except:
        vel_file.close()
        exit()
