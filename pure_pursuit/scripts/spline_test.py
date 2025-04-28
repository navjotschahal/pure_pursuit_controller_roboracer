from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import csv

waypoints = []

def smooth_waypoints(waypoints):
    waypoints = np.array(waypoints)
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    t = np.linspace(0, 1, len(x))

    # 2D B-Spline
    tck, u = splprep([x, y], s=1.0, k=2, per=True)  # k=3 for cubic spline

    t_smooth = np.linspace(0, 1, 3000)  # Increase the number of points for smoother curve
    x_smooth, y_smooth = splev(t_smooth, tck)

    smoothed_waypoints = np.vstack((x_smooth, y_smooth)).T

    return smoothed_waypoints.tolist()

def load_waypoints(csv_file_path):
    waypoints = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x, y = float(row[0]), float(row[1])
            # if float(row[3]) != 0:
            waypoints.append((x, y))
    return waypoints

def visualize_waypoints(original_waypoints, smoothed_waypoints):
    original_waypoints = np.array(original_waypoints)
    smoothed_waypoints = np.array(smoothed_waypoints)

    plt.figure(figsize=(10, 6))
    plt.plot(original_waypoints[:, 0], original_waypoints[:, 1], 'r-', label='Original Waypoints')
    plt.plot(original_waypoints[:, 0], original_waypoints[:, 1], 'ro', markersize=5, label='Original Waypoints')
    plt.plot(smoothed_waypoints[:, 0], smoothed_waypoints[:, 1], 'b-', label='Smoothed Waypoints')
    plt.legend()
    plt.title('Waypoints Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()



if __name__ == "__main__":
    # Example usage
    csv_file_path: Path = Path(__file__).parent / '../config/WP_try1.csv'
    waypoints = load_waypoints(csv_file_path)
    smoothed_waypoints = smooth_waypoints(waypoints)
    visualize_waypoints(waypoints, smoothed_waypoints)
