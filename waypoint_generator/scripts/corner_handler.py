from math import sqrt
import sys
import heapq
import rospy
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import numpy as np
import cv2 as cv
import cv_bridge
import matplotlib.pyplot as plt


class CornerHandler:
    def __init__(self):

        # Load params
        self.map_file = rospy.get_param("/corner_detector/map_file", default="")
        self.USE_APPROXIMATION = rospy.get_param("/corner_detect/use_approximation", default=True)
        self.APPROX_PARAM = rospy.get_param("/corner_detect/approx_param", default=0.0005)
        self.DISTANCE_THRESHOLD = rospy.get_param("/corner_detect/distance_threshold", default=5)
        self.CONVEX_HULL_IDX = rospy.get_param("/corner_detect/convex_hull_idx", default=5)

        if self.map_file == "":
            rospy.logerr("No static map specified.")
            exit()

        # TODO
        # Get static map data
        # rospy.wait_for_service("static_map")
        # try:
        #     get_map = rospy.ServiceProxy("static_map", GetMap)
        #     self._map = get_map().map
        # except rospy.ServiceException as e:
        #     rospy.logerr("Map service call failed: %s" % e)

        rospy.loginfo("Corner Detector Initialized")

    def extractCorners(self):
        # Build CV image from occupancy grid map
        img = cv.imread(self.map_file, cv.IMREAD_GRAYSCALE)

        # Building Binary Image including Static Obstacle
        ret, thresh = cv.threshold(img, 205, 255, cv.THRESH_BINARY_INV)
        thresh = cv.bitwise_xor(img, thresh)
        ret, thresh = cv.threshold(thresh, 205, 255, cv.THRESH_BINARY_INV)

        # Converting to color image for colorized contour representation
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

        # Building Contour , Convex hull
        contours, hierarchy = cv.findContours(thresh, 1, 2)
        cntPoints, cnvxPoints, area = [], [], []
        for cnts in contours:

            if self.USE_APPROXIMATION:
                epsilon = self.APPROX_PARAM * cv.arcLength(cnts, True)
                approx = cv.approxPolyDP(cnts, epsilon, True)

                area.append(cv.moments(approx)["m00"])

                # Extract contour points
                for point in approx:
                    for data in point:
                        cntPoints.append(data)

            else:
                area.append(cv.moments(cnts)["m00"])

                # Extract contour points

                for point in approx:
                    for data in point:
                        cntPoints.append(data)

        # Find convex hull with second largest area
        sortedArea = area[:]
        heapq.heapify(sortedArea)
        cnvxIdx = area.index(sortedArea[-self.CONVEX_HULL_IDX])
        hull = cv.convexHull(contours[cnvxIdx])

        # Extract convex hull points
        for point in hull:
            for data in point:
                cnvxPoints.append(data)

        # Remove contour points from convex hull points
        self.corners = []
        for i in cntPoints:
            isValid = True
            for j in cnvxPoints:
                if sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2) < self.DISTANCE_THRESHOLD:
                    isValid = False
                    break
            if isValid:
                self.corners.append(i)
        print(self.corners)

    def createAvoidingArea(self):
        return

    def checkPath(self):
        return

    def avoidArea(self):
        return


if __name__ == "__main__":
    rospy.init_node("waypoint_generator", disable_signals=True)

    cornerHandler = CornerHandler()

    cornerHandler.extractCorners()
