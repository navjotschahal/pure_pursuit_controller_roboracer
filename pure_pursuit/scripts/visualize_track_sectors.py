import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import splprep, splev
from pathlib import Path

# Define sector distances along the track
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

def load_track_data(csv_file_path):
    """Load track data from CSV file, skipping header row."""
    track_data = []
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header row
        next(reader, None)
        
        for row in reader:
            if len(row) < 8:  # Basic validation
                continue
                
            try:
                print('logging row : ', row)
                track_data.append({
                    'x': float(row[0]),  # x_ref_m
                    'y': float(row[1]),  # y_ref_m
                    'width_left': float(row[2]),  # width_left_m
                    'width_right': float(row[3]),  # width_right_m
                    'psi': float(row[4]),  # psi_racetraj_rad
                    's': float(row[5]),   # s_racetraj_m
                    'kappa': float(row[6]),  # kappa_racetraj_radpm
                    'vx': float(row[7])   # vx_racetraj_mps
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Couldn't process row: {row}. Error: {e}")
                continue
                
    return track_data

def assign_sectors_to_points(track_data, sector_lengths):
    """Assign each track point to a sector based on its s-coordinate."""
    points_by_sector = {sector: [] for sector in sector_lengths}
    
    for point in track_data:
        s = point['s']
        assigned = False
        
        # Find which sector this point belongs to
        for sector in sorted(sector_lengths.keys()):
            if sector == 0:
                start = 0
            else:
                start = sector_lengths[sector-1]
            end = sector_lengths[sector]
            
            if start <= s < end:
                points_by_sector[sector].append(point)
                assigned = True
                break
        
        # Handle points beyond the last sector
        if not assigned:
            last_sector = max(sector_lengths.keys())
            points_by_sector[last_sector].append(point)
    
    return points_by_sector

def smooth_sector(points, s=1.0, k=3, n_points=200):
    """Apply spline smoothing to points in a sector."""
    if len(points) < 4:  # Need at least 4 points for cubic spline
        return [p['x'] for p in points], [p['y'] for p in points]
    
    x = [p['x'] for p in points]
    y = [p['y'] for p in points]
    
    try:
        # Create spline representation
        tck, u = splprep([x, y], s=s, k=k, per=False)
        
        # Evaluate the spline on a finer grid
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = splev(u_new, tck)
        
        return x_new, y_new
    except Exception as e:
        print(f"Spline smoothing failed for sector: {e}")
        return x, y

def visualize_track_sectors(track_data, sector_to_track_lengths_dict):
    """Visualize the track with different colors for each sector."""
    # Assign points to sectors
    points_by_sector = assign_sectors_to_points(track_data, sector_to_track_lengths_dict)
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Use a colormap for different sectors
    num_sectors = len(sector_to_track_lengths_dict)
    colors = plt.cm.jet(np.linspace(0, 1, num_sectors))
    
    # Plot each sector
    for sector, points in points_by_sector.items():
        if not points:
            continue
        
        # Get x,y coordinates for this sector
        x, y = [p['x'] for p in points], [p['y'] for p in points]
        
        # Apply spline smoothing
        if len(points) >= 4:
            x_smooth, y_smooth = smooth_sector(points)
            plt.plot(x_smooth, y_smooth, '-', color=colors[sector], linewidth=3, 
                    label=f'Sector {sector} (s={sector_to_track_lengths_dict[sector]}m)')
        else:
            # Not enough points for smoothing
            plt.plot(x, y, '-', color=colors[sector], linewidth=3, 
                    label=f'Sector {sector} (s={sector_to_track_lengths_dict[sector]}m)')
    
    # Plot original waypoints as small dots
    x_all = [p['x'] for p in track_data]
    y_all = [p['y'] for p in track_data]
    plt.plot(x_all, y_all, 'ko', markersize=1, alpha=0.3)
    
    # Mark start/finish
    start_point = track_data[0]
    plt.plot(start_point['x'], start_point['y'], 'go', markersize=10, label='Start/Finish')
    
    # Finalize the plot
    plt.title('Track Visualization by Sectors')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set the path to your specific CSV file
    csv_file_path = Path("/home/navjot/Desktop/work/sim_ws/src/lab5/slam-and-pure-pursuit-team13/pure_pursuit/config/race3-manual-fitted.csv")
    
    # Alternative if you want to use the path relative to the workspace
    # workspace_path = Path("/home/navjot/Desktop/work/sim_ws")
    # csv_file_path = workspace_path / "src/lab5/slam-and-pure-pursuit-team13/pure_pursuit/config/race3-manual-fitted.csv"
    
    print(f"Using CSV file: {csv_file_path}")
    
    # Load track data
    try:
        track_data = load_track_data(csv_file_path)
        print(f"Loaded {len(track_data)} waypoints from {csv_file_path}")
        
        # Visualize track sectors
        visualize_track_sectors(track_data, sector_to_track_lengths_dict)
    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        print("Please check that the CSV has 8 columns with the expected data types")
