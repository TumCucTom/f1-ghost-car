#!/usr/bin/env python3
"""
F1 Track Position Data Extraction Script

This script retrieves track XY positions for drivers at every time step during qualifying
using the Fast F1 API. It can compare two drivers' positions throughout the session.

Usage:
    python qualifying_positions.py --year 2024 --race "Monaco" --driver1 "VER" --driver2 "HAM"
    python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --export
"""

import argparse
import sys
import os
import fastf1
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_track_position_data(year: int, race: str, driver1: str, driver2: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Get track XY position data for two drivers throughout the qualifying session.
    
    Args:
        year: The year of the race
        race: The name of the Grand Prix
        driver1: Three-letter driver code for first driver
        driver2: Three-letter driver code for second driver
    
    Returns:
        Tuple of (driver1_position_data, driver2_position_data) or (None, None) if error
    """
    try:
        # Create cache directory if it doesn't exist
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"Created cache directory: {cache_dir}")
        
        # Enable caching for better performance
        fastf1.Cache.enable_cache(cache_dir)
        
        # Load the qualifying session
        print(f"Loading qualifying session for {year} {race}...")
        session = fastf1.get_session(year, race, 'Q')
        session.load()
        
        # Debug: Show available drivers and their codes
        results = session.results
        print(f"Available drivers in session:")
        for _, driver in results.iterrows():
            print(f"  {driver['Abbreviation']} ({driver['FullName']}) - Position: {driver['Position']}")
        
        # Get position data for both drivers
        position_data1 = get_driver_position_data(session, driver1)
        position_data2 = get_driver_position_data(session, driver2)
        
        return position_data1, position_data2
        
    except Exception as e:
        print(f"Error loading session data: {e}")
        return None, None


def get_driver_position_data(session, driver_code: str) -> Optional[pd.DataFrame]:
    """
    Get track position data (X, Y coordinates) for a specific driver throughout the session.
    
    Args:
        session: FastF1 session object
        driver_code: Three-letter driver code
    
    Returns:
        DataFrame with position data or None if not found
    """
    try:
        # Get the session results to find the driver
        results = session.results
        driver_result = results.loc[results['Abbreviation'] == driver_code]
        
        if driver_result.empty:
            print(f"Driver {driver_code} not found in session")
            return None
        
        # Get the driver number
        driver_number = driver_result.iloc[0]['DriverNumber']
        
        # Get position data for this driver
        position_data = session.pos_data[driver_number]
        
        if position_data.empty:
            print(f"No position data found for driver {driver_code}")
            return None
        
        # Add driver information to the data
        position_data = position_data.copy()
        position_data['Driver'] = driver_code
        position_data['DriverNumber'] = driver_number
        
        print(f"Retrieved {len(position_data)} position data points for {driver_code}")
        
        return position_data
        
    except Exception as e:
        print(f"Error getting position data for driver {driver_code}: {e}")
        return None


def analyze_position_data(driver1: str, data1: Optional[pd.DataFrame], 
                         driver2: str, data2: Optional[pd.DataFrame]) -> None:
    """
    Analyze and display track position data for two drivers.
    
    Args:
        driver1: Name/code of first driver
        data1: Position data DataFrame for first driver
        driver2: Name/code of second driver
        data2: Position data DataFrame for second driver
    """
    print("\n" + "="*60)
    print("TRACK POSITION DATA ANALYSIS")
    print("="*60)
    
    if data1 is None or data2 is None:
        print("Unable to retrieve position data for analysis.")
        return
    
    # Display basic statistics
    print(f"\n{driver1} Position Data:")
    print(f"  Total data points: {len(data1)}")
    print(f"  Time range: {data1.index[0]} to {data1.index[-1]}")
    print(f"  X coordinate range: {data1['X'].min():.2f} to {data1['X'].max():.2f}")
    print(f"  Y coordinate range: {data1['Y'].min():.2f} to {data1['Y'].max():.2f}")
    
    print(f"\n{driver2} Position Data:")
    print(f"  Total data points: {len(data2)}")
    print(f"  Time range: {data2.index[0]} to {data2.index[-1]}")
    print(f"  X coordinate range: {data2['X'].min():.2f} to {data2['X'].max():.2f}")
    print(f"  Y coordinate range: {data2['Y'].min():.2f} to {data2['Y'].max():.2f}")
    
    # Show sample data
    print(f"\nSample data for {driver1} (first 5 points):")
    print(data1[['X', 'Y', 'Status']].head())
    
    print(f"\nSample data for {driver2} (first 5 points):")
    print(data2[['X', 'Y', 'Status']].head())


def export_position_data(driver1: str, data1: Optional[pd.DataFrame], 
                        driver2: str, data2: Optional[pd.DataFrame], 
                        year: int, race: str) -> None:
    """
    Export position data to CSV files.
    
    Args:
        driver1: Name/code of first driver
        data1: Position data DataFrame for first driver
        driver2: Name/code of second driver
        data2: Position data DataFrame for second driver
        year: Year of the race
        race: Name of the race
    """
    if data1 is None or data2 is None:
        print("Cannot export data - missing position data.")
        return
    
    # Create export directory
    export_dir = f"position_data_{year}_{race.replace(' ', '_')}"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"Created export directory: {export_dir}")
    
    # Export individual driver data
    filename1 = f"{export_dir}/{driver1}_position_data.csv"
    data1.to_csv(filename1)
    print(f"Exported {driver1} data to: {filename1}")
    
    filename2 = f"{export_dir}/{driver2}_position_data.csv"
    data2.to_csv(filename2)
    print(f"Exported {driver2} data to: {filename2}")
    
    # Export combined data
    combined_data = pd.concat([
        data1.assign(Driver=driver1),
        data2.assign(Driver=driver2)
    ], ignore_index=True)
    
    combined_filename = f"{export_dir}/combined_position_data.csv"
    combined_data.to_csv(combined_filename, index=False)
    print(f"Exported combined data to: {combined_filename}")
    
    # Export metadata
    metadata = {
        "race": race,
        "year": year,
        "drivers": [driver1, driver2],
        "export_timestamp": datetime.now().isoformat(),
        "data_points": {
            driver1: len(data1),
            driver2: len(data2)
        }
    }
    
    metadata_filename = f"{export_dir}/metadata.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported metadata to: {metadata_filename}")


def get_fastest_lap_data(session, driver_code: str) -> Optional[pd.DataFrame]:
    """
    Get position data for a driver's fastest qualifying lap.
    
    Args:
        session: FastF1 session object
        driver_code: Three-letter driver code
    
    Returns:
        DataFrame with fastest lap position data or None if not found
    """
    try:
        # Get the session results to find the driver
        results = session.results
        driver_result = results.loc[results['Abbreviation'] == driver_code]
        
        if driver_result.empty:
            print(f"Driver {driver_code} not found in session")
            return None
        
        # Get the driver number
        driver_number = driver_result.iloc[0]['DriverNumber']
        
        # Get all laps for this driver
        driver_laps = session.laps.pick_drivers(driver_number)
        
        if driver_laps.empty:
            print(f"No laps found for driver {driver_code}")
            return None
        
        # Find the fastest lap
        fastest_lap = driver_laps.pick_fastest()
        fastest_lap_time = fastest_lap['LapTime']
        
        print(f"Fastest lap for {driver_code}: {fastest_lap_time}")
        
        # Prefer telemetry which often contains X/Y; fall back to merging with position data
        telemetry = fastest_lap.get_telemetry()
        if telemetry is None or telemetry.empty:
            print(f"No telemetry found for fastest lap of {driver_code}")
            return None

        # Debug available columns
        print(f"Available columns in telemetry: {list(telemetry.columns)}")

        has_xy = ('X' in telemetry.columns) and ('Y' in telemetry.columns)
        if has_xy:
            fastest_lap_positions = pd.DataFrame({
                'X': telemetry['X'],
                'Y': telemetry['Y'],
                'Status': 'OnTrack'
            })
        else:
            # Try to merge with session position data using SessionTime or Date
            pos_df = session.pos_data.get(driver_number)
            if pos_df is None or pos_df.empty:
                print(f"Position data not available to reconstruct X/Y for {driver_code}")
                return None

            left_time_key = None
            right_time_key = None

            if 'SessionTime' in telemetry.columns:
                left_time_key = 'SessionTime'
            elif 'Date' in telemetry.columns:
                left_time_key = 'Date'

            if 'SessionTime' in pos_df.columns:
                right_time_key = 'SessionTime'
            elif 'Date' in pos_df.columns:
                right_time_key = 'Date'

            if not left_time_key or not right_time_key:
                print(f"Could not align telemetry and position data for {driver_code}")
                return None

            telem = telemetry[[left_time_key]].copy()
            pos = pos_df[[right_time_key, 'X', 'Y']].copy()

            telem_sorted = telem.sort_values(by=left_time_key)
            pos_sorted = pos.sort_values(by=right_time_key)

            merged = pd.merge_asof(
                telem_sorted,
                pos_sorted,
                left_on=left_time_key,
                right_on=right_time_key,
                direction='nearest'
            )

            if merged[['X', 'Y']].isna().all().all():
                print(f"Failed to merge telemetry with position data for {driver_code}")
                return None

            fastest_lap_positions = pd.DataFrame({
                'X': merged['X'],
                'Y': merged['Y'],
                'Status': 'OnTrack'
            })
        
        # Add driver information
        fastest_lap_positions['Driver'] = driver_code
        fastest_lap_positions['DriverNumber'] = driver_number
        fastest_lap_positions['LapTime'] = fastest_lap_time
        
        if fastest_lap_positions.empty:
            print(f"No position data found for fastest lap of {driver_code}")
            return None
        
        # Add driver information
        fastest_lap_positions = fastest_lap_positions.copy()
        fastest_lap_positions['Driver'] = driver_code
        fastest_lap_positions['DriverNumber'] = driver_number
        fastest_lap_positions['LapTime'] = fastest_lap_time
        
        print(f"Retrieved {len(fastest_lap_positions)} position data points for {driver_code}'s fastest lap")
        
        return fastest_lap_positions
        
    except Exception as e:
        print(f"Error getting fastest lap data for driver {driver_code}: {e}")
        return None


def create_track_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                     driver1: str, driver2: str, 
                     year: int, race: str, session=None) -> None:
    """
    Create a static plot showing both drivers' fastest lap paths overlaid on the track.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
        session: FastF1 session object for track layout
    """
    plt.figure(figsize=(15, 10))
    
    # Plot track layout if available
    if session is not None:
        try:
            track = session.track
            plt.plot(track.x, track.y, 'k-', linewidth=3, alpha=0.7, label='Track')
        except:
            pass
    
    # Filter data to only show on-track positions
    data1_track = data1[data1['Status'] == 'OnTrack']
    data2_track = data2[data2['Status'] == 'OnTrack']
    
    # Plot driver paths
    plt.plot(data1_track['X'], data1_track['Y'], 
             color='blue', linewidth=3, alpha=0.9, 
             label=f'{driver1} Fastest Lap', marker='o', markersize=2)
    
    plt.plot(data2_track['X'], data2_track['Y'], 
             color='red', linewidth=3, alpha=0.9, 
             label=f'{driver2} Fastest Lap', marker='s', markersize=2)
    
    # Mark start and end points
    if len(data1_track) > 0:
        plt.scatter(data1_track['X'].iloc[0], data1_track['Y'].iloc[0], 
                   color='blue', s=150, marker='o', edgecolor='black', linewidth=2,
                   label=f'{driver1} Lap Start', zorder=5)
        plt.scatter(data1_track['X'].iloc[-1], data1_track['Y'].iloc[-1], 
                   color='blue', s=150, marker='X', edgecolor='black', linewidth=2,
                   label=f'{driver1} Lap End', zorder=5)
    
    if len(data2_track) > 0:
        plt.scatter(data2_track['X'].iloc[0], data2_track['Y'].iloc[0], 
                   color='red', s=150, marker='s', edgecolor='black', linewidth=2,
                   label=f'{driver2} Lap Start', zorder=5)
        plt.scatter(data2_track['X'].iloc[-1], data2_track['Y'].iloc[-1], 
                   color='red', s=150, marker='X', edgecolor='black', linewidth=2,
                   label=f'{driver2} Lap End', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'{year} {race} - {driver1} vs {driver2} Fastest Lap Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save the plot
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_fastest_laps.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved fastest lap plot: {filename}")
    
    plt.show()


def create_animated_track_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                              driver1: str, driver2: str, 
                              year: int, race: str, session=None,
                              follow: bool = False, window_size: float = 300.0,
                              gif_seconds: Optional[float] = None,
                              default_fps: int = 30) -> None:
    """
    Create an animated plot showing both drivers' fastest lap positions over time.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
        session: FastF1 session object for track layout
    """
    # Filter data to only show on-track positions
    data1_track = data1[data1['Status'] == 'OnTrack'].reset_index()
    data2_track = data2[data2['Status'] == 'OnTrack'].reset_index()
    
    if len(data1_track) == 0 or len(data2_track) == 0:
        print("No on-track data available for animation")
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot track layout if available
    if session is not None:
        try:
            track = session.track
            ax.plot(track.x, track.y, 'k-', linewidth=3, alpha=0.7, label='Track')
        except:
            pass
    
    # Initialize empty lines and points
    line1, = ax.plot([], [], 'b-', linewidth=3, alpha=0.8, label=f'{driver1} Fastest Lap')
    line2, = ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label=f'{driver2} Fastest Lap')
    point1, = ax.plot([], [], 'bo', markersize=10, label=f'{driver1} Current')
    point2, = ax.plot([], [], 'rs', markersize=10, label=f'{driver2} Current')
    
    # Set initial axis limits
    all_x = pd.concat([data1_track['X'], data2_track['X']])
    all_y = pd.concat([data1_track['Y'], data2_track['Y']])

    if follow:
        # Center on initial positions
        x_points = []
        y_points = []
        if len(data1_track) > 0:
            x_points.append(float(data1_track['X'].iloc[0]))
            y_points.append(float(data1_track['Y'].iloc[0]))
        if len(data2_track) > 0:
            x_points.append(float(data2_track['X'].iloc[0]))
            y_points.append(float(data2_track['Y'].iloc[0]))
        if x_points and y_points:
            cx = sum(x_points) / len(x_points)
            cy = sum(y_points) / len(y_points)
            pad = window_size * 1.2
            ax.set_xlim(cx - pad, cx + pad)
            ax.set_ylim(cy - pad, cy + pad)
        else:
            ax.set_xlim(all_x.min() - 50, all_x.max() + 50)
            ax.set_ylim(all_y.min() - 50, all_y.max() + 50)
    else:
        ax.set_xlim(all_x.min() - 50, all_x.max() + 50)
        ax.set_ylim(all_y.min() - 50, all_y.max() + 50)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'{year} {race} - {driver1} vs {driver2} Fastest Lap Animation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Determine fps/interval to target desired gif duration if provided
    max_frames = max(len(data1_track), len(data2_track))
    fps = default_fps
    if gif_seconds and gif_seconds > 0:
        fps = max(1, int(round(max_frames / gif_seconds)))
    interval_ms = int(round(1000.0 / fps))

    # Animation function
    def animate(frame):
        # Update lines (show path up to current frame)
        if frame < len(data1_track):
            line1.set_data(data1_track['X'].iloc[:frame+1], data1_track['Y'].iloc[:frame+1])
            point1.set_data([data1_track['X'].iloc[frame]], [data1_track['Y'].iloc[frame]])
        
        if frame < len(data2_track):
            line2.set_data(data2_track['X'].iloc[:frame+1], data2_track['Y'].iloc[:frame+1])
            point2.set_data([data2_track['X'].iloc[frame]], [data2_track['Y'].iloc[frame]])

        if follow:
            # Recenter axis around current positions
            x_points = []
            y_points = []
            if frame < len(data1_track):
                x_points.append(float(data1_track['X'].iloc[frame]))
                y_points.append(float(data1_track['Y'].iloc[frame]))
            if frame < len(data2_track):
                x_points.append(float(data2_track['X'].iloc[frame]))
                y_points.append(float(data2_track['Y'].iloc[frame]))
            if x_points and y_points:
                cx = sum(x_points) / len(x_points)
                cy = sum(y_points) / len(y_points)
                pad = window_size * 1.2
                ax.set_xlim(cx - pad, cx + pad)
                ax.set_ylim(cy - pad, cy + pad)
        
        return line1, line2, point1, point2
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                 interval=interval_ms, blit=(not follow), repeat=True)
    
    # Save animation
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_fastest_laps_animated.gif"
    anim.save(filename, writer='pillow', fps=fps)
    print(f"Saved fastest lap animation: {filename}")
    
    plt.show()


def main():
    """Main function to handle command line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Extract F1 track position data for two drivers during qualifying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qualifying_positions.py --year 2024 --race "Monaco" --driver1 "VER" --driver2 "HAM"
  python qualifying_positions.py --year 2023 --race "Silverstone" --driver1 "LEC" --driver2 "NOR" --export
  python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --plot
  python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --animate
        """
    )
    
    parser.add_argument('--year', type=int, required=True,
                       help='Year of the race (e.g., 2024)')
    parser.add_argument('--race', type=str, required=True,
                       help='Name of the Grand Prix (e.g., "Monaco", "Silverstone")')
    parser.add_argument('--driver1', type=str, required=True,
                       help='Three-letter driver code for first driver (e.g., "VER", "HAM")')
    parser.add_argument('--driver2', type=str, required=True,
                       help='Three-letter driver code for second driver (e.g., "LEC", "NOR")')
    parser.add_argument('--export', action='store_true',
                       help='Export position data to CSV files')
    parser.add_argument('--plot', action='store_true',
                       help='Create static plot of fastest lap track positions')
    parser.add_argument('--animate', action='store_true',
                       help='Create animated plot of fastest lap track positions')
    parser.add_argument('--follow', action='store_true',
                       help='Enable camera follow: keep axes centered on cars')
    parser.add_argument('--follow-window', type=float, default=200.0,
                       help='Half-size of follow window in meters (default: 200)')
    parser.add_argument('--gif-seconds', type=float,
                       help='Target GIF duration in seconds (auto-adjust fps)')
    
    args = parser.parse_args()
    
    # Validate year
    if args.year < 2018 or args.year > 2024:
        print(f"Warning: Year {args.year} may not have complete data available.")
        print("FastF1 typically has data from 2018 onwards.")
    
    # Convert driver codes to uppercase
    driver1 = args.driver1.upper()
    driver2 = args.driver2.upper()
    
    print(f"Extracting track position data for {driver1} vs {driver2}")
    print(f"Race: {args.year} {args.race}")
    
    # Get track position data
    data1, data2 = get_track_position_data(
        args.year, args.race, driver1, driver2
    )
    
    # Display results
    if data1 is not None and data2 is not None:
        analyze_position_data(driver1, data1, driver2, data2)
        
        if args.export:
            export_position_data(driver1, data1, driver2, data2, args.year, args.race)
        
        if args.plot or args.animate:
            # Load session for track layout and fastest lap data
            try:
                cache_dir = 'cache'
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                fastf1.Cache.enable_cache(cache_dir)
                
                session = fastf1.get_session(args.year, args.race, 'Q')
                session.load()
                
                # Get fastest lap data instead of all position data
                print(f"\nExtracting fastest lap data...")
                fastest_lap_data1 = get_fastest_lap_data(session, driver1)
                fastest_lap_data2 = get_fastest_lap_data(session, driver2)
                
                if fastest_lap_data1 is not None and fastest_lap_data2 is not None:
                    if args.animate:
                        create_animated_track_plot(
                            fastest_lap_data1, fastest_lap_data2,
                            driver1, driver2, args.year, args.race, session,
                            follow=args.follow, window_size=args.follow_window,
                            gif_seconds=args.gif_seconds
                        )
                    else:
                        create_track_plot(fastest_lap_data1, fastest_lap_data2, driver1, driver2, args.year, args.race, session)
                else:
                    print("Could not extract fastest lap data for plotting")
                    
            except Exception as e:
                print(f"Error loading session for plotting: {e}")
                session = None
    else:
        print("\n❌ Failed to retrieve position data.")
        print("Please check:")
        print("- Year and race name are correct")
        print("- Driver codes are valid (3-letter format)")
        print("- Internet connection is available")
        sys.exit(1)


if __name__ == "__main__":
    main()
