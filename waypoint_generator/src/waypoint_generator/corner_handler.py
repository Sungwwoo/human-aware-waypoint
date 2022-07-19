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

CORNER_LOCATION = 0
CORNER_ORIENTATION = 1
POINT_IN = 2
POINT_OUT = 3
WAYPOINT = 4

# Logging functions
def ROSINFO(msg):
    rospy.loginfo(msg)


def ROSERR(msg):
    rospy.logerr(msg)


def ROSWARN(msg):
    rospy.logwarn(msg)


def calcOrientation(prev: list, current: list, ret_deg=False):
    """Calculate orientation from previous point to current point

    Args:
        current point (t)
        previous point (t -1)"""
    # unit_vector = np.array([1, 0])
    # input_vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])
    # angle = np.arccos(input_vector.dot(unit_vector) / sqrt(input_vector[0] ** 2 + input_vector[1] ** 2))

    dx = current[0] - prev[0]
    dy = current[1] - prev[1]

    if dx > 0:
        if dy > 0:  # 1
            angle = np.arctan((current[1] - prev[1]) / (current[0] - prev[0]))
        else:  # 4
            angle = np.arctan((current[1] - prev[1]) / (current[0] - prev[0]))

    elif dx < 0:
        if dy > 0:  # 2
            angle = np.pi + np.arctan((current[1] - prev[1]) / (current[0] - prev[0]))
        else:  # 3
            angle = -np.pi + np.arctan((current[1] - prev[1]) / (current[0] - prev[0]))

    else:
        if (current[1] - prev[1]) > 0:
            angle = toRad(90)
        elif (current[1] - prev[1]) < 0:
            angle = toRad(-90)
        else:
            angle = 0.0

    if ret_deg:
        return angle

    q = Quaternion()
    dat = quaternion_from_euler(0, 0, angle)
    q.x = dat[0]
    q.y = dat[1]
    q.z = dat[2]
    q.w = dat[3]
    return q


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


def toDegree(rad: float):
    return rad / np.pi * 180.0


class Corner:
    def __init__(self):
        self.is_configured = [False for i in range(0, 5)]
        self.corner = PoseStamped()
        self.waypoints = []
        self.point_in = PoseStamped()
        self.point_out = PoseStamped()
        self.direction = 0  # 0: clockwise, 1: counterclockwise

    def get_euler_angle(self):
        """Returns the eualer angle of the corner"""
        if self.is_configured[CORNER_ORIENTATION]:
            q = [
                self.corner.pose.orientation.x,
                self.corner.pose.orientation.y,
                self.corner.pose.orientation.z,
                self.corner.pose.orientation.w,
            ]

            return euler_from_quaternion(q)[2]
        else:
            ROSWARN("No orientation configured")
            return

    def get_quaternion(self):
        if self.is_configured[CORNER_ORIENTATION]:
            return self.corner.pose.orientation
        else:
            ROSWARN("No orientation configured")
            return

    def get_corner_location(self):
        """Returns the list of x, y position"""
        if self.is_configured[CORNER_LOCATION]:
            return [self.corner.pose.position.x, self.corner.pose.position.y]
        else:
            ROSWARN("No location configured")

    def get_waypoint_list(self):
        """Returns waypoint list of the corner"""
        if self.is_configured[WAYPOINT]:
            return self.waypoints
        else:
            ROSWARN("No waypoint configured")

    def set_point_in(self, position: list, orientation: Quaternion):
        self.point_in.pose.position.x = position[0]
        self.point_in.pose.position.y = position[1]
        self.point_in.pose.position.z = 0.0
        self.point_in.pose.orientation = orientation
        self.is_configured[POINT_IN] = True

    def set_point_out(self, position: list, orientation: Quaternion):
        self.point_out.pose.position.x = position[0]
        self.point_out.pose.position.y = position[1]
        self.point_out.pose.position.z = 0.0
        self.point_out.pose.orientation = orientation
        self.is_configured[POINT_OUT] = True

    def set_corner_location(self, location: list):
        """Set x, y location"""
        self.corner.pose.position.x = location[0]
        self.corner.pose.position.y = location[1]
        self.corner.pose.position.z = 0.0
        self.is_configured[CORNER_LOCATION] = True

    def set_corner_orientation(self, orientation: Quaternion):
        """Set corner orientation"""
        self.corner.pose.orientation = orientation
        self.is_configured[CORNER_ORIENTATION] = True

    def set_corner_angle(self, angle: float):
        """Set corner angle (degree)"""
        [
            self.corner.pose.orientation.x,
            self.corner.pose.orientation.y,
            self.corner.pose.orientation.z,
            self.corner.pose.orientation.w,
        ] = quaternion_from_euler(0, 0, angle)
        self.is_configured[CORNER_LOCATION] = True

    def set_waypoints(self, radius: int, stepsize: int):
        """Set waypont list"""

        if self.is_configured[WAYPOINT]:
            ROSWARN("Overwriting previous waypoint")

        start_angle = [
            self.point_in.pose.orientation.x,
            self.point_in.pose.orientation.y,
            self.point_in.pose.orientation.z,
            self.point_in.pose.orientation.w,
        ]
        start_angle = euler_from_quaternion(start_angle)[2]

        end_angle = [
            self.point_out.pose.orientation.x,
            self.point_out.pose.orientation.y,
            self.point_out.pose.orientation.z,
            self.point_out.pose.orientation.w,
        ]
        end_angle = euler_from_quaternion(end_angle)[2]

        # print("Angle_in :", toDegree(start_angle), "\nAngle_out:", toDegree(end_angle), "Dir: ", self.direction)
        if self.direction == 0:  # clockwise

            if start_angle > end_angle:
                rads = np.arange(start_angle, end_angle, -toRad(stepsize))

            else:
                if start_angle < 0:
                    rads = np.arange(start_angle, -3.14, -toRad(stepsize))
                    tmp = np.arange(3.14, end_angle, -toRad(stepsize))
                    rads = np.append(rads, tmp)

            # Create circular waypoints
            center = self.get_corner_location()
            for theta in rads:

                point = PoseStamped()
                point.pose.position.x, point.pose.position.y, point.pose.position.z = [
                    center[0] + radius * np.cos(theta),
                    center[1] + radius * np.sin(theta),
                    0.0,
                ]

                [
                    point.pose.orientation.x,
                    point.pose.orientation.y,
                    point.pose.orientation.z,
                    point.pose.orientation.w,
                ] = quaternion_from_euler(0, 0, (theta - np.pi / 2.0))
                self.waypoints.append(point)

        else:  # Counterclockwise
            if start_angle > end_angle:
                if start_angle > 0:
                    rads = np.arange(start_angle, 3.14, toRad(stepsize))
                    tmp = np.arange(-3.14, end_angle, toRad(stepsize))
                    rads = np.append(rads, tmp)
                else:
                    rads = np.arange(start_angle, 0, toRad(stepsize))
                    tmp = np.arange(0, end_angle, toRad(stepsize))
                    rads = np.append(rads, tmp)
                    if end_angle < 0:
                        tmp = np.arange(-3.14, end_angle, toRad(stepsize))
                        rads = np.append(rads, tmp)

            else:
                rads = np.arange(start_angle, end_angle, toRad(stepsize))

            # Create circular waypoints
            center = self.get_corner_location()
            for theta in rads:

                point = PoseStamped()
                point.pose.position.x, point.pose.position.y, point.pose.position.z = [
                    center[0] + radius * np.cos(theta),
                    center[1] + radius * np.sin(theta),
                    0.0,
                ]

                [
                    point.pose.orientation.x,
                    point.pose.orientation.y,
                    point.pose.orientation.z,
                    point.pose.orientation.w,
                ] = quaternion_from_euler(0, 0, (theta + np.pi / 2.0))
                self.waypoints.append(point)
        self.is_configured[WAYPOINT] = True

    def set_direction(self, dir: int):
        if dir == 0 or dir == 1:
            self.direction = dir
        else:
            ROSWARN("Invalid direction.")


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
        self.WAYPOINT_STEPSIZE = rospy.get_param("corner_avoid/step_size", default=25)

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

        delete_marker = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        delete_marker.markers.append(marker)
        self.pub_w_markers.publish(delete_marker)
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
            ROSINFO("No corners detected. Shutting down...")
            return False

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
        return True

    def cbGlobalPath(self, path):

        if self.is_waypoint(self.currentGoal):
            return

        ROSINFO("Received Global Path")
        x, y = [], []

        # Delete Existing waypoints
        delete_marker = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.action = Marker.DELETEALL
        delete_marker.markers.append(marker)
        self.pub_w_markers.publish(delete_marker)
        del delete_marker

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
        if inRangeCorners == []:
            ROSINFO("No corners along the global path")
            return
        ROSINFO("Creating waypoint for " + str(len(inRangeCorners)) + " corners")

        # Set new waypoint array
        for corner in inRangeCorners:
            corner.set_waypoints(self.CORNER_AVOID_RADIUS, self.WAYPOINT_STEPSIZE)
            self.inRangeCorners.append(corner)

        self.markWaypoint()

    def cbGlobalGoal(self, goal):
        ROSINFO("Received Global Goal")
        self.currentGoal = goal.pose
        if self.is_waypoint(goal):
            return
        else:
            self.global_goal = goal
        return

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
            point_found = False
            for i in range(0, len(self.g_path)):
                p_in = self.g_path[i]

                point = Corner()
                if abs(calcDistance(corner, p_in) - self.CORNER_AVOID_RADIUS) < 0.01:

                    # Calculate orientation of point_in
                    if i < len(self.g_path) - 1:
                        angle_in = calcOrientation(corner, p_in)
                    else:
                        # if point_in is the last point of the global plan
                        # there is no need to avoid the corner
                        break

                    # Now find point_out
                    for j in range(i + 1, len(self.g_path)):
                        p_out = self.g_path[j]
                        if abs(calcDistance(corner, p_out) - self.CORNER_AVOID_RADIUS) < 0.01 and calcDistance(p_in, p_out) > (
                            self.CORNER_AVOID_RADIUS / sqrt(2)
                        ):
                            angle_out = calcOrientation(corner, p_out)
                            point_found = True
                            break

                if point_found:
                    point.set_corner_location(corner)
                    point.set_point_in(p_in, angle_in)
                    point.set_point_out(p_out, angle_out)

                    # Find direction
                    middle_idx = int((i + j) / 2)
                    p_mid = self.g_path[middle_idx]
                    angle_in = calcOrientation(corner, p_in, True)
                    angle_mid = calcOrientation(corner, p_mid, True)

                    if angle_in < 0:
                        angle_in = 2 * np.pi + angle_in
                    if angle_mid < 0:
                        angle_mid = 2 * np.pi + angle_mid
                    print(angle_in, angle_mid)
                    if angle_in > angle_mid:
                        if angle_in > np.pi and angle_mid < np.pi:
                            point.set_direction(1)
                        else:
                            point.set_direction(0)

                    else:  # Counterclockwise
                        if angle_in < np.pi and angle_mid > np.pi:
                            point.set_direction(0)
                        else:
                            point.set_direction(1)

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

    def markWaypoint(self):
        waypointMarker = MarkerArray()
        # Create Waypoint Markers
        id = 0
        for i in range(0, len(self.inRangeCorners)):
            corner = self.inRangeCorners[i]

            loc = corner.get_corner_location()
            marker = Marker()
            marker.header.frame_id = "map"
            # marker.header.stamp = time
            marker.ns = "waypoint" + str(i)
            marker.id = id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            marker.pose.position = corner.point_in.pose.position
            marker.lifetime = rospy.Duration(0)
            marker.pose.orientation = corner.point_in.pose.orientation
            marker.color.r, marker.color.g, marker.color.b = 0, 0, 1
            marker.color.a = 0.7
            waypointMarker.markers.append(marker)
            id += 1

            for point in corner.get_waypoint_list():
                marker = Marker()
                marker.header.frame_id = "map"
                # marker.header.stamp = time
                marker.ns = "waypoint" + str(i)
                marker.id = id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.03
                marker.scale.z = 0.03
                marker.pose.position = point.pose.position
                marker.lifetime = rospy.Duration(0)
                marker.pose.orientation = point.pose.orientation
                marker.color.r, marker.color.g, marker.color.b = 1, 0, 0
                marker.color.a = 0.7
                waypointMarker.markers.append(marker)
                id += 1

            marker = Marker()
            marker.header.frame_id = "map"
            # marker.header.stamp = time
            marker.ns = "waypoint" + str(i)
            marker.id = id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.1
            marker.pose.position = corner.point_out.pose.position
            marker.lifetime = rospy.Duration(0)
            marker.pose.orientation = corner.point_out.pose.orientation
            marker.color.r, marker.color.g, marker.color.b = 1, 0, 0
            marker.color.a = 0.7
            waypointMarker.markers.append(marker)
            id += 1

        self.pub_w_markers.publish(waypointMarker)
        return

    def Waypoint(self, point: PoseStamped):
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
