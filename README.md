# human-aware-waypoint


## Algorithm workflow
1. Detect corners of the static map using CV algorithms  

2. Assign possible-collision areas to the corners 

3. Start moving if the global goal is passed\
    3-1. Look for every doors while moving, assigning possible-collision areas to the doors

4. If the robot enters the possible-collision area, back up the global goal and follow the boundary of the area using sequence of waypoints.

## Process
### Decisions to be made
- Map loading method: Directly loading .pgm / Get occupancy grid data from map_server
- Door detection algorithm: lidar-based / depth camera-based / both
### TODO

- [x] Corner detection
- [ ] Map the detected corner with world coordinate and visualize corners with RVIZ (Corner localization)
- [ ] Create possible-collision areas for corners
- [ ] Door detection
- [ ] Door localization
- [ ] Create possible-collision areas for doors
- [ ] Check if the robot is entering possible-collision areas
- [ ] Area Avoidance algorithm
