#! /usr/bin/env python3

from math import sqrt, cos, sin
from multiprocessing.spawn import get_executable
import sys
from tomlkit import string
import yaml
import heapq
import rospy
import tf2_ros
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, Path
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def ROSINFO(msg):
    rospy.loginfo(msg)


def ROSERR(msg):
    rospy.logerr(msg)


def calcOrientation(point1: list, point2: list):
    """Calculate difference between two point and return quarternion

    Args:
        point1: current point (t)
        point2: previous point (t -1)"""
    if point1[0] - point2[0] == 0:
        if (point1[1] - point2[1]) > 0:
            angle = toRad(90)
        elif (point1[1] - point2[1]) < 0:
            angle = toRad(-90)
        else:
            angle = 0.0
    else:
        angle = np.arctan((point1[1] - point2[1]) / (point1[0] - point2[0]))
    return quaternion_from_euler(0, 0, angle)


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


class Corner:
    def __init__(self):
        self.corner = PoseStamped()
        self.waypoints = []

    def get_euler_angle(self):
        """Returns the eualer angle of the corner"""
        q = self.corner.pose.orientation
        q = [q.x, q.y, q.z, q.w]
        return euler_from_quaternion(q)[2]

    def get_quaternion(self):
        return self.corner.pose.orientation

    def get_corner_location(self):
        """Returns the list of x, y position"""
        return [self.corner.pose.position.x, self.corner.pose.position.y]

    def get_waypoint_list(self):
        """Returns waypoint list of the corner"""
        return self.waypoints

    def set_corner_location(self, location: list):
        """Set x, y location"""
        self.corner.pose.position.x = location[0]
        self.corner.pose.position.y = location[1]

    def set_corner_orientation(self, orientation: Quaternion):
        """Set corner orientation"""
        self.corner.pose.orientation = orientation

    def set_corner_angle(self, angle: float):
        """Set corner angle (degree)"""
        [
            self.corner.pose.orientation.x,
            self.corner.pose.orientation.y,
            self.corner.pose.orientation.z,
            self.corner.pose.orientation.w,
        ] = quaternion_from_euler(0, 0, angle)

    def set_waypoints(self, waypoints: list):
        """Set waypont list"""
        self.waypoints = waypoints


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

        ROSINFO("Parameter loaded")

        # Load map image
        if self.map_file == "":
            ROSERR("No static map specified. Use same map file with navigation stack.")

        if self.map_file[-5:] == ".yaml":
            self.image_dir = self.map_file[:-5] + ".pgm"
        else:
            ROSERR("Invalid map_file name. Map file must be end with '.yaml'.")

        self.img = cv.imread(self.image_dir, cv.IMREAD_GRAYSCALE)

        # Load map data
        with open(self.map_file) as file:
            mapData = yaml.safe_load(file)

        self.map_resolution = mapData["resolution"]
        self.map_origin = mapData["origin"]

        self.extracted = False
        self.corners = []

        self.g_path = []  # Sequence of PoseStamped

        self.inRangeCorners = []
        self.global_goal = PoseStamped()
        self.currentGoal = PoseStamped()
        self.cornerMarkers = MarkerArray()
        self.waypointMarker = MarkerArray()
        self.pub_c_markers = rospy.Publisher("corner_markers", MarkerArray, queue_size=10)
        self.pub_w_markers = rospy.Publisher("waypoint_markers", MarkerArray, queue_size=10)
        self.sub_global_path = rospy.Subscriber(self.global_plan_name, Path, self.cbGlobalPath)
        self.sub_global_goal = rospy.Subscriber(self.global_goal_name, PoseStamped, self.cbGlobalGoal)
        rospy.sleep(1)

        ROSINFO("Deleting existing RVIZ markers")
        delete_marker = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        delete_marker.markers.append(marker)
        self.pub_c_markers.publish(delete_marker)
        del delete_marker

        if self.show_image:
            cv.namedWindow("Contour", cv.WINDOW_NORMAL)
            cv.resizeWindow("Contour", width=1000, height=1000)
            cv.imshow("Contour", self.final_img), cv.waitKey(0)
        rospy.loginfo("Corner Detector Initialized")

    def extractCorners(self):
        """Extract corners of given map."""
        ROSINFO("Extracting corners")

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

        ROSINFO("Creating markers")
        # Create Corner Markers
        orientation = quaternion_from_euler(0, 0, self.map_origin[2])
        time = rospy.Time.now()

        for i in range(0, len(self.corners)):
            marker = Marker()
            marker.header.frame_id = "map"
            # marker.header.stamp = time
            marker.ns = "corners"
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
            marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = orientation
            marker.color.r, marker.color.g, marker.color.b = 1, 1, 0
            marker.color.a = 0.5
            self.cornerMarkers.markers.append(marker)

        self.pub_c_markers.publish(self.cornerMarkers)
        return self.extracted

    def getRobotLocation(self):
        """Return the current location of robot.

        Return:
            [Location_x, Location_y] based on map frame
        """
        loc = getTF("map", "base_link")
        self.robotLocation = [loc.transform.translation.x, loc.transform.translation.y]
        return self.robotLocation

    def checkPath(self):
        """Find the corners need to be avoided

        Return:
            List of the corners
        """
        inRangeCorners = []
        for corner in self.corners:
            for i in range(0, len(self.g_path)):
                path = self.g_path[i]

                if calcDistance(corner, path) < self.CORNER_DIST_THRES:
                    point = Corner()
                    point.set_corner_location(corner)
                    if i < len(self.g_path) - 1:
                        nextPathPoint = self.g_path[i + 1]
                        angle = calcOrientation(nextPathPoint, path)
                    else:
                        prevPathPoint = self.g_path[i - 1]
                        angle = calcOrientation(path, prevPathPoint)
                    point.set_corner_angle(angle)
                    inRangeCorners.append(point)
                    break

        return inRangeCorners

    def is_waypoint(self, goal: PoseStamped):
        """Check if the received move_base goal is one of the waypoints.

        Args:
            goal: Goal pose to check

        Returns:
            Boolean
        """

        if self.inRangeCorners == []:
            return False
        for corner in self.inRangeCorners:
            for point in corner.get_waypoint_list():
                if point.pose == goal:
                    return True
                else:
                    continue
        return False

    def cbGlobalPath(self, path):
        ROSINFO("Received Global Path")
        x, y = [], []

        if self.is_waypoint(self.currentGoal):
            ROSINFO("Received goal belongs to waypoint array")
            return

        # Delete Existing waypoints
        delete_marker = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        delete_marker.markers.append(marker)
        self.pub_w_markers.publish(delete_marker)
        self.inRangeCorners = []

        # Save new global path
        for data in path.poses:
            x.append(data.pose.position.x)
            y.append(data.pose.position.y)

        tempList = []
        for i in range(0, len(x)):
            tempList.append([x[i], y[i]])

        self.g_path = tempList[:]

        # Create new waypoint
        self.waypointMarker = MarkerArray()

        inRangeCorners = self.checkPath()

        ROSINFO("Creating waypoint for " + str(len(inRangeCorners)) + " corners")

        # Set new waypoint array
        for corner in inRangeCorners:
            waypoint = self.createWaypoint(corner, self.CORNER_AVOID_RADIUS, self.WAYPOINT_STEPSIZE)
            corner.set_waypoints(waypoint)
            self.inRangeCorners.append(corner)

        self.pub_w_markers.publish(self.waypointMarker)

    def cbGlobalGoal(self, goal):
        ROSINFO("Received Global Goal")
        self.currentGoal = goal.pose
        if self.is_waypoint(goal):
            return
        else:
            self.global_goal = goal

    def checkDistance(self, corners: list):
        """Check distance between robot and corner.

        Args:
            corner: [x, y] value of target point

        Return:

        """
        currentLoc = self.getRobotLocation()

        # for corner in corners:
        #     if calcDistance(corner, self.getRobotLocation()) < 0.2:
        #         self.filterWaypoint(corner)

        return

    def filterWaypoint(self, waypointSet: list):
        """Discard unvalid waypoints

        Args:
            waypointSet: Set of corner and waypoints [corner, waypoints]

        Return:
            Valid wayoints of the corner
        """

        # TODO:

        return

    # TODO: Calculate Arc Length
    def checkArcLength(self, waypointSet: list):
        """Calculate arc length of area based on index.

        Args:
            waypointSet: Set of corner and waypoints [corner, waypoints]

        """

        return

    def createWaypoint(self, corner: Corner, radius: int, stepsize: int):

        waypoint = []
        rads = np.arange(-3.14, 3.14, toRad(stepsize))

        # Create circular waypoints
        center = corner.get_corner_location()
        for theta in rads:
            point = PoseStamped()
            point.pose.position.x, point.pose.position.y, point.pose.position.z = [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                0.0,
            ]
            if -np.pi / 2 <= (corner.get_euler_angle() - (theta + np.pi / 2.0)) <= np.pi / 2:
                [
                    point.pose.orientation.x,
                    point.pose.orientation.y,
                    point.pose.orientation.z,
                    point.pose.orientation.w,
                ] = quaternion_from_euler(0, 0, (theta + np.pi / 2.0))
            else:
                [
                    point.pose.orientation.x,
                    point.pose.orientation.y,
                    point.pose.orientation.z,
                    point.pose.orientation.w,
                ] = quaternion_from_euler(0, 0, (theta - np.pi / 2.0))
            waypoint.append(point)

        # Create Waypoint Markers
        for i in range(0, len(waypoint)):
            marker = Marker()
            marker.header.frame_id = "map"
            # marker.header.stamp = time
            marker.ns = "waypoint" + str(center[0] + center[1])
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.pose.position = waypoint[i].pose.position
            marker.lifetime = rospy.Duration(0)
            marker.pose.orientation = waypoint[i].pose.orientation
            marker.color.r, marker.color.g, marker.color.b = 1, 0, 0
            marker.color.a = 0.7
            self.waypointMarker.markers.append(marker)
        return waypoint

    def setWaypoint(self, point: PoseStamped):
        """Set waypoint and send to the actionlib, and wait until the action is done.

        Args:
            point: [[position], [orientation]]
        """
        client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        client.wait_for_server()

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = point
        client.send_goal(goal)
        client.wait_for_result()

        return

    def prepare(self):
        if not self.extractCorners():
            ROSERR("Extraction Failed")
