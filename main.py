import pykitti
import numpy as np
from wicp import WeightedICP
from tqdm import tqdm
import numpy as np
import open3d as o3d
import os
from utils import save_poses_to_txt, deskew_points , visualize_map

if __name__ == "__main__":

    # KITTI dataset directory and configuration
    basedir = '/home/jiwoo/Documents/Dataset/KITTI'
    date = '2011_09_29'
    drive = '0004'

    start_frame = 0
    end_frame = 300
    frame_range = range(start_frame, end_frame + 1)  

    rl_icp = WeightedICP()

    # Load the data
    dataset = pykitti.raw(basedir, date, drive)
    velo_data = [dataset.get_velo(i) for i in frame_range]
    oxts_data = dataset.oxts  # Ground truth data

    imu_T_velo = np.array([
        [ 9.999976e-01,  7.553071e-04, -2.035826e-03, -8.086759e-01],
        [-7.854027e-04,  9.998898e-01, -1.482298e-02,  3.195559e-01],
        [ 2.024406e-03,  1.482454e-02,  9.998881e-01, -7.997231e-01],
        [ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]
    ])

    def extract_gt_pose(oxts_data, frame_range):
        """Extract ground truth pose from KITTI oxts data."""
        pose_gt = np.eye(4)
        poses = []
        for i in frame_range:
            pose_t_0 = oxts_data[i].T_w_imu   
            pose_t_1 = oxts_data[i + 1].T_w_imu
            pose = np.linalg.inv(pose_t_0) @ pose_t_1
            pose = (np.linalg.inv(imu_T_velo) @ (pose @ imu_T_velo))  # L_{t-1}_T_L_{t}
            pose_gt = pose_gt @ pose
            poses.append(pose_gt)
        return np.array(poses)

    # Extract ground truth poses
    gt_poses = extract_gt_pose(oxts_data, frame_range)

    T_est = np.eye(4)  # Initial transformation matrix (G_T_L_{0})
    T_prev_est = np.eye(4)  # Previous estimated transformation matrix (G_T_L_{t-1})

    # Initialize lists for plotting
    foo = np.eye(4)
    est_path = []
    est_path.append(foo)
    gt_path = []
    gt_path.append(foo)

    # Global Mapping
    global_map = []
    
    use_deskew = False
    
    # Loop through the frames and update the plot
    for t in tqdm(range(len(velo_data) - 1)):
        
        if t == 0:
            P = velo_data[t][:, :3]  # N x 3
            ones = np.ones((P.shape[0], 1))  # N x 1
            P_homog = np.hstack((P, ones)).T  # 4 x N
            transformed_P_homog = T_est @ P_homog  # 4 x N
            transformed_P = transformed_P_homog[:3, :].T  # N x 3
            global_map.append(transformed_P)

        else:
            Q = velo_data[t]  # N x 4
            Q_times = np.linspace(0, 1, len(Q))
            Q_ = Q[:, :3]  # N x 3
            
            if use_deskew:
                Q = deskew_points(Q_, Q_times, T_prev_est) # N x 3
            else:
                Q = Q_
            
            T, iter_ = rl_icp.run(
                P, Q, 
                alpha=0.9, 
                max_corr_distance=1.0, 
                k=10, 
                planar_threshold=0.01, 
                max_iterations=50, 
                tolerance=1e-4, 
                voxel_size=0.8,
                initial_transformation = T_prev_est
            )  # L_{t}_T_L_{t-1}

            T_est = T_est @ np.linalg.inv(T)  # G_T_L_{t} = G_T_L_{t-1} @ L_{t-1}_T_L_{t}
            T_prev_est = T 
            P = Q

            # Append estimated and ground truth positions
            est_path.append(T_est)
            gt_path.append(gt_poses[t])

            # Mapping
            ones = np.ones((P.shape[0], 1))  # N x 1
            P_homog = np.hstack((P, ones)).T  # 4 x N
            transformed_P_homog = T_est @ P_homog  # 4 x N
            transformed_P = transformed_P_homog[:3, :].T  # N x 3
            global_map.append(transformed_P)

   # File paths for saving
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # Save ground truth poses
    gt_filepath = os.path.join(output_dir, 'ground_truth_poses.txt')
    save_poses_to_txt(gt_filepath, gt_path, "Ground truth poses")

    # Save estimated poses
    est_filepath = os.path.join(output_dir, 'estimated_poses.txt')
    save_poses_to_txt(est_filepath, est_path, "Estimated poses")
    visualize_map(global_map, voxel_size=0.1, point_size=2.0, colormap_name='Blues', background_color=(0.2, 0.2, 0.2))
        