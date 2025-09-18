import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

DATASET_PATH = Path("dataset")
BBOX_FILE = DATASET_PATH / "bbox_light.csv"
XYZ_DIR = DATASET_PATH / "xyz/"

# Load Data
bbox_df = pd.read_csv(BBOX_FILE)

# Create List to Store all XYZ points
xyz_list = []

# Iterate through DataFrame to add Coordinates to xyz_list
for index, row in bbox_df.iterrows():

    # Check for empty coordinates
    if row['x2'] == 0:
        continue   

    # Find centers for each plane
    u = (row['x1'] + row['x2']) / 2
    v = (row['y1'] + row['y2']) / 2

    

    frame_number = row['frame'] 
    formatted_id = f"depth{frame_number:06d}"
    path = XYZ_DIR / f"{formatted_id}.npz"

    # Check if files exist
    if not path.exists():
        continue

    xyz_points = np.load(path)['xyz']   

    # Define Patch Size
    PATCH_SIZE = 5

    # Calculate Boundaries
    H, W, _ = xyz_points.shape
    half_patch = PATCH_SIZE // 2
    v_start = int(v - half_patch)
    v_end = int(v + half_patch + 1)
    u_start = int(u - half_patch)
    u_end = int(u + half_patch + 1)

    # Clamp Boundaries
    v_start = max(0, v_start)
    v_end = min(H, v_end)
    u_start = max(0, u_start)
    u_end = min(W, u_end)   

    # Generate Patch
    patch = xyz_points[v_start:v_end, u_start:u_end]

    # Clean Data
    condition = np.any(patch != 0, axis=-1)
    valid_points = patch[condition]

    # Average Valid Points
    if valid_points.shape[0] > 0:
        final_xyz = np.mean(valid_points, axis=0)
    # Use Single Pixel
    else:
        final_xyz = xyz_points[int(v), int(u)]

    # Add Coordinate
    xyz_list.append(final_xyz)

# Convert xyz_list into an np array
p_cam = np.array(xyz_list)

# Find Derivatives
dt = 10.0 / len(bbox_df)
dx_dt = np.gradient(p_cam[:, 0], dt)
dy_dt = np.gradient(p_cam[:, 1], dt)

# Calculate Angular Velocity
x_safe = np.where(np.abs(p_cam[:, 0]) < 1e-2, 1e-2, p_cam[:, 0])
omega_z = -dy_dt / x_safe 

# Calculate Forward Velocity
v_x = -dx_dt + omega_z * p_cam[:, 1]

# Set Starting Point
x0 = p_cam[0, 0]
y0 = p_cam[0, 1]
theta0 = np.pi
trajectory = [[x0, y0]]
thetas = [theta0]

# Build the Path
for i in range(1, len(v_x)):
    
    # Get State from Previous Time Step
    old_x, old_y = trajectory[-1]
    old_theta = thetas[-1]

    # Update Heading
    new_theta = old_theta + omega_z[i] * dt
    thetas.append(new_theta)

    # Calculate Motion in World Frame
    v_x_world = v_x[i] * np.cos(old_theta)
    v_y_world = v_x[i] * np.sin(old_theta)

    # Update Position
    new_x = old_x + v_x_world * dt
    new_y = old_y + v_y_world * dt 
    trajectory.append([new_x, new_y])

trajectory = np.array(trajectory)


### Visualization ###

# Create figure and axes object
fig, ax = plt.subplots(figsize=(10, 8))

# Plot Traffic Light
ax.plot(trajectory[:, 0], trajectory[:, 1], marker='.', linestyle='-', color='dodgerblue', label='Ego-vehicle Path')

# Mark Start + End Points
ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', color='limegreen', markersize=12, label='Start')
ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color='gold', markersize=12, label='End')

# Label + Title + Grid
ax.set_title("Ego-Vehicle Trajectory", fontsize=16, fontweight='bold')
ax.set_xlabel("X-position (m)", fontsize=12)
ax.set_ylabel("Y-position (m)", fontsize=12)
ax.grid(True)
ax.legend()

# Make Aspect Ratio Equal
ax.set_aspect('equal', adjustable='box')

# Save Plot and Display it
plt.savefig("trajectory.png", dpi=300, bbox_inches='tight')
plt.show()



     


















