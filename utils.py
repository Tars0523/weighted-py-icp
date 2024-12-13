import numpy as np
import open3d as o3d
import matplotlib.cm as cm  # 색상 맵을 사용하기 위한 임포트
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
from scipy.spatial.transform import Rotation as R

def save_poses_to_txt(filepath, poses, description):
    with open(filepath, 'w') as file:
        for t, pose in enumerate(poses):
            # Extract translation
            x, y, z = pose[:3, 3]
            
            # Extract quaternion (rotation matrix to quaternion)
            r = R.from_matrix(pose[:3, :3])
            qx, qy, qz, qw = r.as_quat()  # Quaternion in (x, y, z, w) format
            
            # Write to file: timestep x y z qx qy qz qw
            file.write(f"{t} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    print(f"{description} saved to {filepath}")

def deskew_points(points, times, T_prev):
    
    mid_pose_timestamp = 0.5

    # Compute delta transformation (SE3 to se3 using logarithm map)
    rotation_matrix = T_prev[:3, :3]
    translation_vector = T_prev[:3, 3]

    # Logarithmic map for rotation
    delta_rotation_vector = R.from_matrix(rotation_matrix).as_rotvec()

    deskewed_points = []
    for point, time in zip(points, times):
        # Interpolate rotation and translation based on time
        interpolated_rotation = R.from_rotvec((time - mid_pose_timestamp) * delta_rotation_vector).as_matrix()
        interpolated_translation = (time - mid_pose_timestamp) * translation_vector

        # Construct interpolated transformation matrix
        motion = np.eye(4)
        motion[:3, :3] = interpolated_rotation
        motion[:3, 3] = interpolated_translation

        # Apply the motion correction
        point_homog = np.append(point, 1)  # Convert to homogeneous coordinates
        corrected_point = motion @ point_homog
        deskewed_points.append(corrected_point[:3])

    return np.array(deskewed_points)

def visualize_map(global_map, voxel_size=0.5, point_size=0.5, colormap_name='Blues', background_color=(0, 0, 0)):

    all_points = np.vstack(global_map)  # N_total x 3


    if not all_points.dtype == np.float64:
        all_points = all_points.astype(np.float64)


    mask = np.isfinite(all_points).all(axis=1)
    all_points = all_points[mask]


    z_values = all_points[:, 2]  


    z_values = all_points[:, 2]
    z_min, z_max = 0, 1
    z_norm = (z_values - z_min) / (z_max - z_min)


    colormap = cm.get_cmap(colormap_name)  
    colors = colormap(z_norm)[:, :3]  


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)  


    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = point_size  
    render_option.background_color = np.array(background_color)  

    vis.run()
    vis.destroy_window()
    