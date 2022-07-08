#! /usr/bin/env python3

from math import sqrt, cos, sin
import sys
from tomlkit import string
import yaml
import heapq
import rospy
import tf2_ros
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def getTF(target_frame: str, source_frame: str):
    # tf listner
    tfBuffer = tf2_ros.Buffer()
    tfListner = tf2_ros.TransformListener(tfBuffer)

    while True:
        try:
            trans = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time())
            return trans
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.sleep(1)
            continue


def calcDistance(a: list, b: list):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def toRad(degree: int):
    return float(degree) * np.pi / 180.0


class CornerHandler:
    def __init__(self):

        # Load params
        self.map_file = rospy.get_param("/corner_detector/map_file", default="")
        self.show_image = rospy.get_param("/corner_detector/show_image", default=False)
        self.USE_APPROXIMATION = rospy.get_param("/corner_detector/use_approximation", default=True)
        self.APPROX_PARAM = rospy.get_param("/corner_detector/approx_param", default=0.005)
        self.DISTANCE_THRESHOLD = rospy.get_param("/corner_detector/distance_threshold", default=5)
        self.CONVEX_HULL_IDX = rospy.get_param("/corner_detector/convex_hull_idx", default=2)
        self.global_plan_name = rospy.get_param("/global_plan_name", default="/move_base/NavfnROS/plan")
        self.global_goal_name = rospy.get_param("/global_goal_name", default="/move_base/current_goal")
        self.CORNER_DIST_THRES = rospy.get_param("corner_avoid/corner_distance_threshold", default=0.7)
        self.CORNER_AVOID_RADIUS = rospy.get_param("corner_avoid/radius", default=1.0)
        self.WAYPOINT_STEPSIZE = rospy.get_param("corner_avoid/step_size", default=15)
        self.waypointMarker = MarkerArray()

        rospy.loginfo("Parameter loaded")

        # Load map image
        if self.map_file == "":
            rospy.logerr("No static map specified. Use same map file with navigation stack.")
            exit()
        if self.map_file[-5:] == ".yaml":
            self.image_dir = self.map_file[:-5] + ".pgm"
        else:
            rospy.logerr("Invalid map_file name. Map file must be end with '.yaml'.")
            exit()

        self.img = cv.imread(self.image_dir, cv.IMREAD_GRAYSCALE)

        # Load map data
        with open(self.map_file) as file:
            mapData = yaml.safe_load(file)

        self.map_resolution = mapData["resolution"]
        self.map_origin = mapData["origin"]

        self.extracted = False
        self.corners = []
        self.g_path = []
        self.inRangeCorners = []
        self.waypoint = []
        self.global_goal = PoseStamped()
        self.cornerMarkers = MarkerArray()
        self.waypointMarker = MarkerArray()
        self.pub_markers = rospy.Publisher("corner_markers", MarkerArray, queue_size=10)
        rospy.sleep(1)

        rospy.loginfo("Deleting existing RVIZ markers")
        delete_marker = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        delete_marker.markers.append(marker)
        self.pub_markers.publish(delete_marker)
        del delete_marker

        if self.show_image:
            cv.namedWindow("Contour", cv.WINDOW_NORMAL)
            cv.resizeWindow("Contour", width=1000, height=1000)
            cv.imshow("Contour", self.final_img), cv.waitKey(0)
        rospy.loginfo("Corner Detector Initialized")

    def extractCorners(self):
        """Extract corners of given map."""
        rospy.loginfo("Extracting corners")

        # Building Binary Image including Static Obstacle
        ret, thresh = cv.threshold(self.img, 205, 255, cv.THRESH_BINARY_INV)
        thresh = cv.bitwise_xor(self.img, thresh)
        ret, thresh = cv.threshold(thresh, 205, 255, cv.THRESH_BINARY_INV)

        if self.show_image:
            img = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)

        # Building Contour , Convex hull
        contours, hierarchy = cv.findContours(thresh, 1, 2)
        cntPoints, cnvxPoints, area = [], [], []
        for cnts in contours:

            if self.USE_APPROXIMATION:
                epsilon = self.APPROX_PARAM * cv.arcLength(cnts, True)
                approx = cv.approxPolyDP(cnts, epsilon, True)

                area.append(cv.moments(cnts)["m00"])

                # Extract contour points
                if self.show_image:
                    img = cv.drawContours(img, [approx], -1, color=(255, 0, 0))

                for point in approx:
                    for data in point:
                        cntPoints.append(data)

            else:
                area.append(cv.moments(cnts)["m00"])

                # Extract contour points

                for point in cnts:
                    for data in point:
                        cntPoints.append(data)

        # Find convex hull with second largest area
        sortedArea = area[:]
        heapq.heapify(sortedArea)
        cnvxIdx = area.index(sortedArea[-self.CONVEX_HULL_IDX])
        hull = cv.convexHull(contours[cnvxIdx])

        if self.show_image:
            self.final_img = cv.drawContours(img, [hull], -1, color=(0, 0, 255))

        # Extract convex hull points
        for point in hull:
            for data in point:
                if self.show_image:
                    self.final_img = cv.circle(img, data, 4, (0, 0, 255), 1)
                cnvxPoints.append(data)

        # Remove contour points from convex hull points

        for i in cntPoints:
            isValid = True
            for j in cnvxPoints:
                if sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2) < self.DISTANCE_THRESHOLD:
                    isValid = False
                    break
            if isValid:

                i = i.astype(np.float32)

                if self.show_image:
                    img = cv.circle(img, (int(i[0]), int(i[1])), 4, (0, 255, 0), 1)

                # Map the corner to the world coordinate
                i[0] *= self.map_resolution  # resolution: meter/pixel
                i[1] *= self.map_resolution  # resolution: meter/pixel

                # Align to the origin
                i[0] += self.map_origin[0]
                i[1] += self.map_origin[1]

                i[1] = -i[1]
                self.corners.append(i)

        if self.corners == []:
            self.extracted = False
        else:
            self.extracted = True

        # Save corners for debug
        # with open("corners.txt", "w") as fileWrite:
        #     for point in self.corners:
        #         fileWrite.write(str(point[0]) + "\t" + str(point[1]) + "\n")

        rospy.loginfo("Creating markers")
        # Create Markers
        orientation = quaternion_from_euler(0, 0, self.map_origin[2])
        time = rospy.Time.now()

        for i in range(0, len(self.corners)):
            marker = Marker()
            marker.header.frame_id = "map"
            # marker.header.stamp = time
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.1
            marker.pose.position.x = self.corners[i][0]
            marker.pose.position.y = self.corners[i][1]
            marker.pose.position.z = 0.0
            marker.lifetime = rospy.Duration(0)
            [marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w] = orientation
            marker.color.r, marker.color.g, marker.color.b = 1, 1, 0
            marker.color.a = 0.5
            self.cornerMarkers.markers.append(marker)

        return self.extracted

    def pubMarkers(self):
        """Publish corner marker array."""
        self.pub_markers.publish(self.cornerMarkers)

    def getRobotLocation(self):
        """Return the current location of robot.

        Return:
            [Location_x, Location_y] based on map frame
        """
        loc = getTF("map", "base_link")
        self.robotLocation = [loc.transform.translation.x, loc.transform.translation.y]
        return self.robotLocation

    def checkPath(self):
        for corner in self.corners:
            for point in self.g_path:
                if calcDistance(corner, point) < self.CORNER_DIST_THRES:
                    self.inRangeCorners.append(corner)
        return

    def is_waypoint(self, goal: list):
        """Check if the received move_base goal is one of the waypoints.

        Args:
            goal: Received goal, [x, y]

        Returns:
            Boolean
        """
        # for point in self.waypoint:
        if self.global_goal == goal:
            return True
        else:
            return False

    def cbGlobalPath(self, path):
        rospy.loginfo("Received Global Path")
        x, y = [], []
        for data in path.poses:
            x.append(-data.pose.position.y)
            y.append(data.pose.position.x)
        tempList = []
        for i in range(0, len(x)):
            tempList.append([x[i], y[i]])
        self.g_path = tempList[:]

        self.checkPath()

    def cbGlobalGoal(self, data):
        rospy.loginfo("Received Global Goal")
        goal = [data.pose.position.x, data.pose.position.y]
        if self.is_waypoint(goal):
            return
        else:
            self.global_goal = goal

        # TODO: Check distance between in-range-corner and robot

    def checkDistance(self, corners: list):
        """Check distance between robot and corner.

        Args:
            corner: [x, y] value of target point

        Return:

        """
        for corner in corners:
            if calcDistance(corner, self.getRobotLocation()) < 0.2:
                self.filterWaypoint(corner)

        return

    def filterWaypoint(self, corner: list, waypointList: list):
        return

    # TODO: Calculate Arc Length
    def checkArcLength(self, waypoint: list):
        """Calculate arc length of area based on index.

        Args:
            waypoint: List of move_base goal

        """
        return

    def createWaypoint(self, center: list, radius: int, stepsize: int):

        waypoint = []
        rads = np.arange(-3.14, 3.14, toRad(stepsize))
        id = 0

        # Create circular waypoints
        for theta in rads:
            position = [center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), 0.0]
            orientation = quaternion_from_euler(0, 0, (theta + np.pi / 2.0))
            point = [position, orientation]
            waypoint.append(point)

        self.setWaypointDirection(waypoint)

        for i in range(0, len(waypoint)):
            marker = Marker()
            marker.header.frame_id = "odom"
            # marker.header.stamp = time
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.pose.position.x = waypoint[i][0][0]
            marker.pose.position.y = waypoint[i][0][1]
            marker.pose.position.z = 0.0
            marker.lifetime = rospy.Duration(0)
            [marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w] = waypoint[i][1]
            marker.color.r, marker.color.g, marker.color.b = 1, 1, 0
            marker.color.a = 0.5
            self.waypointMarker.markers.append(marker)
        return waypoint

    def setWaypointDirection(self, waypoint: list):
        return

    def setWaypoint(self, point: list):
        """Set waypoint and send to the actionlib, and wait until the action is done.

        Args:
            point: [[position], [orientation]]
        """
        client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = point[0][0]
        goal.target_pose.pose.position.y = point[0][1]
        goal.target_pose.pose.position.z = 0.0
        goal.target_pose.pose.orientation.x = point[1][0]
        goal.target_pose.pose.orientation.y = point[1][1]
        goal.target_pose.pose.orientation.z = point[1][2]
        goal.target_pose.pose.orientation.w = point[1][3]

        client.send_goal(goal)
        client.wait_for_result()

        return

    def prepare(self):
        self.extractCorners()
        self.pubMarkers()
