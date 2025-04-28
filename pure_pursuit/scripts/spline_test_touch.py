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
    tck, u = splprep([x, y], s=8.0, k=2, per=True)  # k=3 for cubic spline

    t_smooth = np.linspace(0, 1, 100)  # Increase the number of points for smoother curve
    x_smooth, y_smooth = splev(t_smooth, tck)

    smoothed_waypoints = np.vstack((x_smooth, y_smooth)).T

    return smoothed_waypoints.tolist()

def load_waypoints(csv_file_path, skip_header=0):
    """
    Load waypoints from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        skip_header: Number of header rows to skip (default: 0)
    
    Returns:
        List of waypoint tuples (x, y)
    """
    waypoints = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        # Skip header rows if specified
        for _ in range(skip_header):
            next(reader, None)
            
        for row in reader:
            x, y = float(row[0]), float(row[1])
            if float(row[3]) != 0:
                waypoints.append((x, y))
    return waypoints

def calculate_cumulative_distances(waypoints):
    """Calculate cumulative distance along the waypoints."""
    distances = [0]  # Start with 0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        distance = np.sqrt(dx**2 + dy**2)
        distances.append(distances[-1] + distance)
    return distances

def visualize_waypoints_with_sectors(original_waypoints, smoothed_waypoints, sector_lengths_dict):
    original_waypoints = np.array(original_waypoints)
    smoothed_waypoints = np.array(smoothed_waypoints)
    
    # Calculate cumulative distances along the track
    distances = calculate_cumulative_distances(smoothed_waypoints)
    total_distance = distances[-1]
    
    plt.figure(figsize=(12, 8))
    plt.plot(original_waypoints[:, 0], original_waypoints[:, 1], 'k--', alpha=0.5, label='Original Waypoints')
    
    # Sort sectors by position
    sorted_sectors = sorted(sector_lengths_dict.items(), key=lambda x: x[1])
    
    # Generate colors for each sector
    colors = plt.cm.jet(np.linspace(0, 1, len(sorted_sectors) + 1))
    
    # Starting point
    start_idx = 0
    current_sector = -1
    
    # Plot each sector with different color
    for i, (sector_num, length) in enumerate(sorted_sectors):
        # Find index where we cross this sector length
        end_idx = next((j for j, d in enumerate(distances) if d >= length), len(distances) - 1)
        
        # Plot this segment
        segment = smoothed_waypoints[start_idx:end_idx+1]
        plt.plot(segment[:, 0], segment[:, 1], '-', color=colors[i], linewidth=3, 
                 label=f'Sector {sector_num} (Length: {length:.1f}m)')
        
        # Mark sector boundary
        plt.plot(smoothed_waypoints[end_idx, 0], smoothed_waypoints[end_idx, 1], 'o', 
                 color='black', markersize=8)
        plt.annotate(f"S{sector_num}", (smoothed_waypoints[end_idx, 0], smoothed_waypoints[end_idx, 1]), 
                     fontsize=10, fontweight='bold', xytext=(5, 5), textcoords="offset points")
        
        # Update start index for next segment
        start_idx = end_idx
    
    plt.title('Track Visualization with Sectors', fontsize=14)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Makes circles look like circles
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    # Dictionary of sector lengths
    sector_to_track_lengths_dict = {
    0: 10.1,
    1: 19.5,
    2: 23.6,
    3: 29.0,
    4: 31.8,
    5: 35.6,
    6: 46.9,
    7: 50.1,
    8: 57.9,
    9: 61.1,
    10: 67.1,
    11: 71.8,
    12: 74.6
}
    
    # Example usage
    csv_file_path: Path = Path(__file__).parent / '../config/race3-manual-fitted.csv'
    waypoints = load_waypoints(csv_file_path, skip_header=1)
    smoothed_waypoints = smooth_waypoints(waypoints)
    visualize_waypoints_with_sectors(waypoints, smoothed_waypoints, sector_to_track_lengths_dict)
