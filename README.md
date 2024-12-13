# WICP Project

This repository contains the implementation and evaluation of the Weighted Iterative Closest Point (WICP) algorithm. Below is a summary of the contents of this project and the associated visualizations.

## Directory Overview

- **main.py**: The main script for running the WICP algorithm.
- **utils.py**: Contains utility functions used throughout the project.
- **wicp.py**: Implements the WICP algorithm.
- **traj_plot.py**: Generates trajectory plots for visualization.
- **output/**: Directory for storing output files.
- **__pycache__/**: Cache directory for Python compiled files.

## Visualizations

### 1. Error Comparison
- **File:** `error_comparsion.png`
- **Description:** This plot compares the error metrics of different iterations or configurations of the WICP algorithm.
- ![Error Comparison](error_comparsion.png)

### 2. Map Visualization
- **File:** `map.png`
- **Description:** A visualization of the map generated or used during the execution of the WICP algorithm.
- ![Map](map.png)

### 3. Normal Candidates
- **File:** `normal_candidate.png`
- **Description:** Displays the normal candidates used in the point cloud alignment process.
- ![Normal Candidates](normal_candidate.png)

### 4. Trajectory Evaluation
- **File:** `traj_eval.png`
- **Description:** A plot showing the evaluation metrics for the trajectory estimated by the WICP algorithm.
- ![Trajectory Evaluation](traj_eval.png)

## How to Run

1. Ensure you have all the dependencies installed.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Visualize the results using the provided scripts and images.

## Notes

- The `output/` directory will contain intermediate results and final outputs.
- Use `traj_plot.py` to generate custom trajectory plots for different datasets.

Feel free to explore and modify the code to suit your project needs.

