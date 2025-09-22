# WA Coding Challenge

## Method

### Ego-Trajectory Estimation

The vehicle's path can be calculated by tracking the traffic light. A 5x5 pixel patch around the traffic light gives 3d positions from depth data, which then can be used to find the cars's forward velocity (v_x) and angular velocity (ω_z):

```
ω_z = -dY/dt / X
v_x = -dX/dt + ω_z * Y
```

The car path is built by integrating these numbers over time.

### Detection

- **YOLO Detection**: YOLOv11 detects and tracks people and vehicles with persistent object IDs (Though I could only map the golf cart and 1 barrel)
- **Color Detection**: Color filtering allowed identification of orange barrels
- **3D Coordinate Transformation**: The detections were transformed into 3d coordinate positions

## Assumptions

- Traffic light is static and serves as world reference
- Vehicles move on a flat plane without slipping/rotating
- Constant time between frames

## Results

The script generated a bird's-eye view plot (`bev.png`) showing:

- The car's trajectory
- One barrel detected
- Golf cart that is ahead of the car 
- Traffic light at origin

