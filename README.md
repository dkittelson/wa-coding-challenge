# Computer Vision Challenge: Ego-Trajectory & Bird’s-Eye View Mapping

## Method

The vehicle's path was calculated by inferring its motion from the movement of the traffic light. The process can be divided into three main stages:

1.  **3D Feature Extraction**: The 3D position of the traffic light relative to the camera was determined for each video frame. To ensure an accurate result, a 5x5 pixel patch around the traffic light's bounding box center was extracted from the depth data. Invalid points were filtered out, and the remaining 3D coordinates were averaged.

2.  **Kinematic Velocity Calculation**: The vehicle's forward speed ($v_x$) and its turning rate ($\omega_z$) were calculated using a kinematic model based on the time derivatives of the light's measured 3D position ($X, Y$). The relevant equations are:
    $$\omega_z = -\frac{\dot{Y}}{X} \quad , \quad v_x = -\dot{X} + \omega_z Y$$

3.  **Trajectory Integration**: The final path was then constructed by integrating these velocities over time. Starting from an initial position and heading defined by the first measurement, the vehicle's state was updated step-by-step using Euler integration to build the complete trajectory.

## Assumptions

* The **traffic light is perfectly static** and serves as a fixed point in the world.
* The vehicle moves on a **flat 2D plane**.
* The vehicle cannot slip or move sideways.
* The time interval between video frames is constant.

## Results

The script generates a plot of the vehicle's trajectory, saved as `trajectory.png`.

!(trajectory.png)

The picture shows the vehicle starting on the right side of the frame and moving towards the traffic light (located at the origin). The curve in the path shows that the model detected the vehicle's rotation (turning) as it drove forward. The smoothness of the trajectory shows the effectiveness of the patch-averaging technique, as it reduced noise. 