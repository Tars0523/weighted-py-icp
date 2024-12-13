import os
import numpy as np
import matplotlib.pyplot as plt

# 데이터 파일 경로 설정
base_dir = "/home/jiwoo/Documents/slam/wicp/output"
truth_file = os.path.join(base_dir, "truth", "kitti_ex.txt")
algorithm_dirs = [
    os.path.join(base_dir, "algorithms", "alpha_01", "kitti_ex", "estimated_poses.txt"),
    os.path.join(base_dir, "algorithms", "alpha_05", "kitti_ex", "estimated_poses.txt"),
    os.path.join(base_dir, "algorithms", "alpha_09", "kitti_ex", "estimated_poses.txt")
]

# 데이터 로드 함수
def load_trajectory(file_path):
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Ground Truth 데이터 로드
truth_data = load_trajectory(truth_file)
if truth_data is None:
    raise ValueError("Failed to load Ground Truth data")

# 알고리즘 결과 데이터 로드
algorithm_data = []
labels = ["Alpha 0.1", "Alpha 0.5", "Alpha 0.9"]
for algo_path in algorithm_dirs:
    data = load_trajectory(algo_path)
    if data is not None:
        algorithm_data.append(data)

# 에러 계산
def compute_errors(truth, estimate):
    errors = np.abs(truth[:, 1:4] - estimate[:, 1:4])
    return errors

errors = [compute_errors(truth_data, algo_data) for algo_data in algorithm_data]

# 에러 시각화
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
error_labels = ["X Error", "Y Error", "Z Error"]
colors = ["red", "blue", "green"]

for i in range(3):  # x, y, z 에러 각각 플롯
    for j, error in enumerate(errors):
        axs[i].plot(error[:, i], label=f"{labels[j]} - {error_labels[i]}", color=colors[j])
    axs[i].set_ylabel(f"{error_labels[i]} [m]", fontsize=12)
    axs[i].grid(True)
    axs[i].legend(fontsize=10)

axs[2].set_xlabel("Time [frame]", fontsize=14)
plt.suptitle("Trajectory Error Comparison", fontsize=16)
plt.show()