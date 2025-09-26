#!/usr/bin/env python3
"""
F1 Track Position Visualization Script

This script plots track XY positions for two drivers over time, overlaying their paths
on the same graph to compare their racing lines and positions throughout the session.

Usage:
    python plot_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA"
    python plot_positions.py --data-dir "position_data_2024_Hungary" --driver1 "NOR" --driver2 "PIA"
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import json
from typing import Optional, Tuple
import fastf1
from datetime import datetime, timedelta


def load_position_data_from_csv(data_dir: str, driver1: str, driver2: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load position data from exported CSV files.
    
    Args:
        data_dir: Directory containing the CSV files
        driver1: First driver code
        driver2: Second driver code
    
    Returns:
        Tuple of (driver1_data, driver2_data) or (None, None) if error
    """
    try:
        # Load individual driver data
        file1 = os.path.join(data_dir, f"{driver1}_position_data.csv")
        file2 = os.path.join(data_dir, f"{driver2}_position_data.csv")
        
        if not os.path.exists(file1) or not os.path.exists(file2):
            print(f"CSV files not found in {data_dir}")
            return None, None
        
        data1 = pd.read_csv(file1, index_col=0, parse_dates=True)
        data2 = pd.read_csv(file2, index_col=0, parse_dates=True)
        
        print(f"Loaded {len(data1)} data points for {driver1}")
        print(f"Loaded {len(data2)} data points for {driver2}")
        
        return data1, data2
        
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return None, None


def get_track_layout(year: int, race: str) -> Optional[fastf1.Track]:
    """
    Get track layout information for visualization.
    
    Args:
        year: Year of the race
        race: Name of the race
    
    Returns:
        Track object or None if not available
    """
    try:
        # Enable caching
        fastf1.Cache.enable_cache('cache')
        
        # Load session to get track info
        session = fastf1.get_session(year, race, 'Q')
        session.load()
        
        return session.track
        
    except Exception as e:
        print(f"Could not load track layout: {e}")
        return None


def create_static_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                      driver1: str, driver2: str, 
                      year: int, race: str, 
                      track: Optional[fastf1.Track] = None) -> None:
    """
    Create a static plot showing both drivers' paths overlaid.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
        track: Track layout information
    """
    plt.figure(figsize=(15, 10))
    
    # Plot track layout if available
    if track is not None:
        try:
            # Plot track boundaries
            plt.plot(track.x, track.y, 'k-', linewidth=3, alpha=0.7, label='Track')
        except:
            pass
    
    # Filter data to only show on-track positions
    data1_track = data1[data1['Status'] == 'OnTrack']
    data2_track = data2[data2['Status'] == 'OnTrack']
    
    # Plot driver paths
    plt.plot(data1_track['X'], data1_track['Y'], 
             color='blue', linewidth=2, alpha=0.8, 
             label=f'{driver1} Path', marker='o', markersize=1)
    
    plt.plot(data2_track['X'], data2_track['Y'], 
             color='red', linewidth=2, alpha=0.8, 
             label=f'{driver2} Path', marker='s', markersize=1)
    
    # Mark start and end points
    if len(data1_track) > 0:
        plt.scatter(data1_track['X'].iloc[0], data1_track['Y'].iloc[0], 
                   color='blue', s=100, marker='o', edgecolor='black', 
                   label=f'{driver1} Start', zorder=5)
        plt.scatter(data1_track['X'].iloc[-1], data1_track['Y'].iloc[-1], 
                   color='blue', s=100, marker='X', edgecolor='black', 
                   label=f'{driver1} End', zorder=5)
    
    if len(data2_track) > 0:
        plt.scatter(data2_track['X'].iloc[0], data2_track['Y'].iloc[0], 
                   color='red', s=100, marker='s', edgecolor='black', 
                   label=f'{driver2} Start', zorder=5)
        plt.scatter(data2_track['X'].iloc[-1], data2_track['Y'].iloc[-1], 
                   color='red', s=100, marker='X', edgecolor='black', 
                   label=f'{driver2} End', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'{year} {race} - {driver1} vs {driver2} Track Positions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save the plot
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_track_positions.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved static plot: {filename}")
    
    plt.show()


def create_animated_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                        driver1: str, driver2: str, 
                        year: int, race: str, 
                        track: Optional[fastf1.Track] = None) -> None:
    """
    Create an animated plot showing both drivers' positions over time.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
        track: Track layout information
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
    if track is not None:
        try:
            ax.plot(track.x, track.y, 'k-', linewidth=3, alpha=0.7, label='Track')
        except:
            pass
    
    # Initialize empty lines and points
    line1, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label=f'{driver1} Path')
    line2, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7, label=f'{driver2} Path')
    point1, = ax.plot([], [], 'bo', markersize=8, label=f'{driver1} Current')
    point2, = ax.plot([], [], 'rs', markersize=8, label=f'{driver2} Current')
    
    # Set axis limits
    all_x = pd.concat([data1_track['X'], data2_track['X']])
    all_y = pd.concat([data1_track['Y'], data2_track['Y']])
    
    ax.set_xlim(all_x.min() - 50, all_x.max() + 50)
    ax.set_ylim(all_y.min() - 50, all_y.max() + 50)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'{year} {race} - {driver1} vs {driver2} Animated Track Positions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Animation function
    def animate(frame):
        # Update lines (show path up to current frame)
        if frame < len(data1_track):
            line1.set_data(data1_track['X'].iloc[:frame+1], data1_track['Y'].iloc[:frame+1])
            point1.set_data([data1_track['X'].iloc[frame]], [data1_track['Y'].iloc[frame]])
        
        if frame < len(data2_track):
            line2.set_data(data2_track['X'].iloc[:frame+1], data2_track['Y'].iloc[:frame+1])
            point2.set_data([data2_track['X'].iloc[frame]], [data2_track['Y'].iloc[frame]])
        
        return line1, line2, point1, point2
    
    # Create animation
    max_frames = max(len(data1_track), len(data2_track))
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                 interval=50, blit=True, repeat=True)
    
    # Save animation
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_animated.gif"
    anim.save(filename, writer='pillow', fps=20)
    print(f"Saved animated plot: {filename}")
    
    plt.show()


def create_time_series_plot(data1: pd.DataFrame, data2: pd.DataFrame, 
                           driver1: str, driver2: str, 
                           year: int, race: str) -> None:
    """
    Create time series plots showing position changes over time.
    
    Args:
        data1: Position data for first driver
        data2: Position data for second driver
        driver1: First driver code
        driver2: Second driver code
        year: Year of the race
        race: Name of the race
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Filter to on-track data
    data1_track = data1[data1['Status'] == 'OnTrack']
    data2_track = data2[data2['Status'] == 'OnTrack']
    
    # Plot X position over time
    ax1.plot(data1_track.index, data1_track['X'], 'b-', linewidth=2, label=f'{driver1} X Position')
    ax1.plot(data2_track.index, data2_track['X'], 'r-', linewidth=2, label=f'{driver2} X Position')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title(f'{year} {race} - X Position Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Y position over time
    ax2.plot(data1_track.index, data1_track['Y'], 'b-', linewidth=2, label=f'{driver1} Y Position')
    ax2.plot(data2_track.index, data2_track['Y'], 'r-', linewidth=2, label=f'{driver2} Y Position')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title(f'{year} {race} - Y Position Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"{year}_{race.replace(' ', '_')}_{driver1}_vs_{driver2}_time_series.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved time series plot: {filename}")
    
    plt.show()


def main():
    """Main function to handle command line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Plot F1 track position data for two drivers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from live data
  python plot_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA"
  
  # Plot from exported CSV data
  python plot_positions.py --data-dir "position_data_2024_Hungary" --driver1 "NOR" --driver2 "PIA"
  
  # Create animated plot
  python plot_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --animate
        """
    )
    
    # Data source options (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--year', type=int,
                           help='Year of the race (e.g., 2024)')
    data_group.add_argument('--data-dir', type=str,
                           help='Directory containing exported CSV files')
    
    parser.add_argument('--race', type=str,
                       help='Name of the Grand Prix (e.g., "Monaco", "Silverstone")')
    parser.add_argument('--driver1', type=str, required=True,
                       help='Three-letter driver code for first driver (e.g., "VER", "HAM")')
    parser.add_argument('--driver2', type=str, required=True,
                       help='Three-letter driver code for second driver (e.g., "LEC", "NOR")')
    parser.add_argument('--animate', action='store_true',
                       help='Create animated plot instead of static plot')
    parser.add_argument('--time-series', action='store_true',
                       help='Create time series plot showing position over time')
    
    args = parser.parse_args()
    
    # Convert driver codes to uppercase
    driver1 = args.driver1.upper()
    driver2 = args.driver2.upper()
    
    # Load data
    if args.data_dir:
        # Load from CSV files
        print(f"Loading position data from {args.data_dir}...")
        data1, data2 = load_position_data_from_csv(args.data_dir, driver1, driver2)
        year = 2024  # Default year for CSV data
        race = "Unknown"  # Will be extracted from directory name if possible
    else:
        # Load from FastF1 API (this would require importing the main script)
        print("Live data loading not implemented in this version.")
        print("Please use --data-dir to load from exported CSV files.")
        sys.exit(1)
    
    if data1 is None or data2 is None:
        print("Failed to load position data.")
        sys.exit(1)
    
    # Get track layout if possible
    track = None
    if args.year and args.race:
        track = get_track_layout(args.year, args.race)
    
    # Create plots
    if args.time_series:
        create_time_series_plot(data1, data2, driver1, driver2, year, race)
    elif args.animate:
        create_animated_plot(data1, data2, driver1, driver2, year, race, track)
    else:
        create_static_plot(data1, data2, driver1, driver2, year, race, track)


if __name__ == "__main__":
    main()
