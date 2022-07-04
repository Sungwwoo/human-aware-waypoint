# human-aware-waypoint


## Algorithm workflow
1. Detect corners of the static map using CV algorithms  

2. Assign possible-collision areas to the corners 

3. Start moving if the global goal is passed\
    3-1. Look for every doors while moving, assigning possible-collision areas to the doors

4. If the robot enters the possible-collision area, back up the global goal and follow the boundary of the area using sequence of waypoints.

## Process

### TODO

- [x] Corner detection
- [ ] Map the detected corner with world coordinate and visualize corners with RVIZ
- [ ] Create possible-collision areas
- [ ] Door detection
- [ ] Door localization
