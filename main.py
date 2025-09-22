import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

# Setup
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

DATASET_PATH = Path("dataset")
BBOX_FILE = DATASET_PATH / "bboxes_light.csv"
XYZ_DIR = DATASET_PATH / "xyz/"
RGB_DIR = DATASET_PATH / "rgb/"
bbox_df = pd.read_csv(BBOX_FILE)

# ================= EGO TRAJECTORY (PART A) =================
def load_ego_trajectory():
    """Load ego vehicle trajectory from Part A"""
    xyz_list = []
    
    # Extracting 3d positions
    for index, row in bbox_df.iterrows():
        if row['x2'] == 0:
            continue
        
        u = (row['x1'] + row['x2']) / 2
        v = (row['y1'] + row['y2']) / 2
        frame_number = row['frame']
        path = XYZ_DIR / f"depth{frame_number:06d}.npz"
        
        if not path.exists():
            continue
        
        try:
            npz_data = np.load(path)
            if 'xyz' in npz_data:
                xyz_points = npz_data['xyz']
            elif 'points' in npz_data:
                xyz_points = npz_data['points']
            else:
                available_keys = list(npz_data.keys())
                if len(available_keys) > 0:
                    xyz_points = npz_data[available_keys[0]]
                else:
                    continue
        except Exception:
            continue
        
        # Find 3d position using patch
        PATCH_SIZE = 5
        H, W = xyz_points.shape[0], xyz_points.shape[1]
        half_patch = PATCH_SIZE // 2
        v_start = max(0, int(v - half_patch))
        v_end = min(H, int(v + half_patch + 1))
        u_start = max(0, int(u - half_patch))
        u_end = min(W, int(u + half_patch + 1))
        
        patch = xyz_points[v_start:v_end, u_start:u_end]
        condition = np.any(patch != 0, axis=-1)
        valid_points = patch[condition]
        
        if valid_points.shape[0] > 0:
            final_xyz = np.mean(valid_points, axis=0)
        else:
            final_xyz = xyz_points[int(v), int(u)]
        
        xyz_list.append(final_xyz)
    
    # Calculate trajectory
    p_cam = np.array(xyz_list)
    dt = 10.0 / len(bbox_df)
    dx_dt = np.gradient(p_cam[:, 0], dt)
    dy_dt = np.gradient(p_cam[:, 1], dt)
    x_safe = np.where(np.abs(p_cam[:, 0]) < 1e-2, 1e-2, p_cam[:, 0])
    omega_z = -dy_dt / x_safe
    v_x = -dx_dt + omega_z * p_cam[:, 1]
    
    # Build the trajectory
    x0, y0, theta0 = p_cam[0, 0], p_cam[0, 1], np.pi
    trajectory = [[x0, y0]]
    thetas = [theta0]
    
    for i in range(1, len(v_x)):
        old_x, old_y = trajectory[-1]
        old_theta = thetas[-1]
        new_theta = old_theta + omega_z[i] * dt
        thetas.append(new_theta)
        v_x_world = v_x[i] * np.cos(old_theta)
        v_y_world = v_x[i] * np.sin(old_theta)
        new_x = old_x + v_x_world * dt
        new_y = old_y + v_y_world * dt
        trajectory.append([new_x, new_y])
    
    return np.array(trajectory)

# ================= OBJECT DETECTION =================
def detect_objects():
    """Detect objects using YOLO and color-based methods"""
    if not YOLO_AVAILABLE:
        return {}
    
    model = YOLO("yolo11n.pt")
    object_tracks = defaultdict(list)
    rgb_files = sorted(list(RGB_DIR.glob("*.png")))
    
    for i, rgb_file in enumerate(rgb_files):
        image = cv2.imread(str(rgb_file))
        results = model.track(image, persist=True, verbose=False, conf=0.5)
        
        # Filter and categorize detections
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            names = results[0].names
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.arange(len(boxes))
            
            for box, cls, conf, track_id in zip(boxes, classes, confidences, track_ids):
                class_name = names[int(cls)]
                
                if class_name in ['person', 'car', 'truck', 'bicycle', 'motorcycle'] and conf > 0.6:
                    x_center, y_center, width, height = box
                    
                    # Filter by size and pos
                    if width < 20 or height < 20:
                        continue
                    image_height = image.shape[0]
                    if y_center < image_height * 0.3:
                        continue
                    
                    # Determine object type
                    if class_name == 'person':
                        obj_type = 'pedestrian'
                    elif class_name in ['car', 'truck']:
                        obj_type = 'golf_cart'
                    else:
                        obj_type = 'bicycle'
                    
                    object_tracks[f"{obj_type}_{int(track_id)}"].append({
                        'frame': i, 'x_pixel': x_center, 'y_pixel': y_center, 'type': obj_type
                    })
        
        # Detect barrels using color
        if i % 5 == 0:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([8, 120, 120])
            upper_orange = np.array([20, 255, 255])
            barrel_mask = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # Clean mask, find shapes, fill gaps, etc
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            barrel_mask = cv2.morphologyEx(barrel_mask, cv2.MORPH_CLOSE, kernel)
            barrel_mask = cv2.morphologyEx(barrel_mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(barrel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for j, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if 500 < area < 3000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.5 < aspect_ratio < 2.0:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            if cy > image_height * 0.4:
                                object_tracks[f"barrel_static_{j}"].append({
                                    'frame': i, 'x_pixel': cx, 'y_pixel': cy, 'type': 'barrel'
                                })
    
    # Filter tracks by minimum detections
    filtered_tracks = {}
    for obj_id, detections in object_tracks.items():
        obj_type = detections[0]['type'] if detections else 'unknown'
        min_detections = {'barrel': 2, 'pedestrian': 5, 'golf_cart': 5, 'bicycle': 5}
        if len(detections) >= min_detections.get(obj_type, 5):
            filtered_tracks[obj_id] = detections
    
    return filtered_tracks

# ================= COORDINATE TRANSFORMATION =================
def convert_to_world_coordinates(object_tracks, ego_trajectory):
    """Convert pixel detections to world coordinates"""
    world_tracks = defaultdict(list)
    
    for obj_id, detections in object_tracks.items():
        for detection in detections:
            frame_idx = detection['frame']
            x_pixel = detection['x_pixel']
            y_pixel = detection['y_pixel']
            
            # Load depth data
            frame_number = frame_idx + 1
            xyz_path = XYZ_DIR / f"depth{frame_number:06d}.npz"
            if not xyz_path.exists():
                continue
            
            try:
                npz_data = np.load(xyz_path)
                available_keys = list(npz_data.keys())
                
                if 'xyz' in available_keys:
                    xyz_points = npz_data['xyz']
                elif 'points' in available_keys:
                    xyz_points = npz_data['points']
                elif len(available_keys) == 1:
                    xyz_points = npz_data[available_keys[0]]
                else:
                    continue
                
                # Extract 3d position using patch
                PATCH_SIZE = 7
                H, W = xyz_points.shape[0], xyz_points.shape[1]
                half_patch = PATCH_SIZE // 2
                v_center = max(0, min(int(y_pixel), H - 1))
                u_center = max(0, min(int(x_pixel), W - 1))
                v_start = max(0, v_center - half_patch)
                v_end = min(H, v_center + half_patch + 1)
                u_start = max(0, u_center - half_patch)
                u_end = min(W, u_center + half_patch + 1)
                
                patch = xyz_points[v_start:v_end, u_start:u_end]
                condition = np.any(patch != 0, axis=-1)
                valid_points = patch[condition]
                
                if valid_points.shape[0] > 3:
                    final_xyz = np.mean(valid_points, axis=0)
                else:
                    final_xyz = xyz_points[v_center, u_center]
                
                x_cam, y_cam, z_cam = final_xyz[0], final_xyz[1], final_xyz[2]
                
                # Skip invalid depths
                if (x_cam == 0 and y_cam == 0 and z_cam == 0) or np.any(np.isnan([x_cam, y_cam, z_cam])):
                    continue
                
                # Distance filtering
                distance = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
                if distance < 1.0 or distance > 80.0:
                    continue
                
                # Object-specific distance filtering
                obj_type = detection['type']
                if obj_type == 'golf_cart' and (x_cam < 2.0 or distance > 40.0):
                    continue
                elif obj_type == 'pedestrian' and distance > 35.0:
                    continue
                elif obj_type == 'barrel' and distance > 50.0:
                    continue
                
                # Transform to world coordinates
                if frame_idx < len(ego_trajectory):
                    ego_x, ego_y = ego_trajectory[frame_idx]
                    world_x = ego_x - x_cam  # Objects ahead are closer to traffic light
                    world_y = ego_y + y_cam
                    
                    world_tracks[obj_id].append({
                        'frame': frame_idx, 'x_world': world_x, 'y_world': world_y, 'type': obj_type
                    })
                    
            except Exception:
                continue
    
    # Final coordinate filtering
    filtered_tracks = {}
    min_positions = {'barrel': 1, 'pedestrian': 2, 'golf_cart': 3, 'bicycle': 2}
    for obj_id, positions in world_tracks.items():
        obj_type = positions[0]['type'] if positions else 'unknown'
        if len(positions) >= min_positions.get(obj_type, 2):
            x_coords = [pos['x_world'] for pos in positions]
            y_coords = [pos['y_world'] for pos in positions]
            if (max(x_coords) < 200 and min(x_coords) > -100 and 
                max(y_coords) < 100 and min(y_coords) > -100):
                filtered_tracks[obj_id] = positions
    
    return filtered_tracks

# ================= VISUALIZATION =================
def create_bev_plot(ego_trajectory, world_tracks):
    """Create Bird's Eye View plot"""
    plt.figure(figsize=(14, 10))
    
    # Ego vehicle trajectory
    plt.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], 
             'b-o', linewidth=3, markersize=3, label='Ego Vehicle', alpha=0.8)
    plt.plot(ego_trajectory[0, 0], ego_trajectory[0, 1], 
             'go', markersize=15, label='Ego Start', markeredgecolor='black')
    plt.plot(ego_trajectory[-1, 0], ego_trajectory[-1, 1], 
             'rs', markersize=15, label='Ego End', markeredgecolor='black')
    
    # Object styles
    styles = {
        'pedestrian': {'color': 'orange', 'marker': '^', 'size': 120},
        'golf_cart': {'color': 'purple', 'marker': 's', 'size': 150},
        'barrel': {'color': 'brown', 'marker': 'o', 'size': 180},
        'bicycle': {'color': 'cyan', 'marker': 'D', 'size': 100}
    }
    
    plotted_types = set()
    for obj_id, positions in world_tracks.items():
        obj_type = positions[0]['type']
        style = styles.get(obj_type, {'color': 'gray', 'marker': 'x', 'size': 80})
        
        x_coords = [pos['x_world'] for pos in positions]
        y_coords = [pos['y_world'] for pos in positions]
        
        # Determine if static or dynamic
        movement = np.sqrt((x_coords[-1] - x_coords[0])**2 + (y_coords[-1] - y_coords[0])**2) if len(positions) > 1 else 0
        is_static = movement < 2.0
        
        if is_static and len(x_coords) > 1:
            # Static: show avg position
            plt.scatter(np.mean(x_coords), np.mean(y_coords), 
                       c=style['color'], marker=style['marker'], s=style['size'], 
                       alpha=0.9, edgecolors='black', linewidth=2)
        else:
            # Dynamic: show trajectory
            plt.plot(x_coords, y_coords, color=style['color'], marker=style['marker'], 
                    linestyle='-', linewidth=3, markersize=8, alpha=0.9, markeredgecolor='black')
        
        # Legend
        if obj_type not in plotted_types:
            movement_type = 'Static' if is_static else 'Dynamic'
            plt.scatter([], [], c=style['color'], marker=style['marker'], s=120, 
                       edgecolors='black', label=f"{obj_type.title()} ({movement_type})")
            plotted_types.add(obj_type)
    
    # Traffic light at origin
    plt.plot(0, 0, '*', color='gold', markersize=25, markeredgecolor='black', 
             markeredgewidth=3, label='Traffic Light (Origin)')
    
    # Formatting
    plt.title("Bird's Eye View", fontsize=16, fontweight='bold')
    plt.xlabel("X Position (m)", fontsize=12)
    plt.ylabel("Y Position (m)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("bev.png", dpi=300, bbox_inches='tight')
    plt.show()

# ================= EXECUTION =================
def main():
    ego_trajectory = load_ego_trajectory()
    object_tracks = detect_objects()
    
    if object_tracks:
        world_tracks = convert_to_world_coordinates(object_tracks, ego_trajectory)
        create_bev_plot(ego_trajectory, world_tracks)
    else:
        create_bev_plot(ego_trajectory, {})

if __name__ == "__main__":
    main()