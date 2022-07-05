# human-aware-waypoint


## Algorithm overview
1. Detect corners of the static map using CV algorithms  

2. Assign possible-collision areas to the corners 

3. Start moving if the global goal is passed\
    3-1. Look for every doors and assign possible-collision areas to the doors while moving
 
4. If the robot enters the possible-collision area, back up the global goal and follow the boundary of the area using sequence of waypoints.

## Process
### Main Contribution?
TBD
### TODO

- [x] Corner detection
- [X] Map the detected corner with world coordinate and visualize corners with RVIZ (Corner localization)
- [ ] Create possible-collision areas for corners
- [ ] Door detection
- [ ] Door localization
- [ ] Create possible-collision areas for doors
- [ ] Check if the robot is entering possible-collision areas
- [ ] Area Avoidance algorithm
