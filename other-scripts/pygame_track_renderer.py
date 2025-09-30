#!/usr/bin/env python3
"""
Pygame Track Renderer for F1 Data Visualization

This script creates a realistic track visualization using FastF1 track data
with tarmac, grass, and curbs rendered in pygame.

Usage:
    python pygame_track_renderer.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA"
"""

import argparse
import sys
import os
import fastf1
import pandas as pd
import numpy as np
import pygame
import math
from typing import Tuple, List, Optional
import json
from datetime import datetime


class TrackRenderer:
    def __init__(self, width: int = 1920, height: int = 1080):
        """Initialize the pygame track renderer."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("F1 Track Visualization")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.TARMAC_COLOR = (60, 60, 60)  # Dark gray
        self.GRASS_COLOR = (34, 139, 34)  # Forest green
        self.CURB_RED = (220, 20, 20)     # Red
        self.CURB_WHITE = (255, 255, 255) # White
        self.TRACK_LINE = (255, 255, 0)   # Yellow center line
        self.BACKGROUND = (0, 100, 0)     # Dark green background
        
        # Track data
        self.track_points = None
        self.track_bounds = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
    
    def _create_track_from_positions(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Create an approximate track outline from position data."""
        if not positions:
            return []
        
        # Convert to numpy arrays for easier processing
        positions_array = np.array(positions)
        x_coords = positions_array[:, 0]
        y_coords = positions_array[:, 1]
        
        # Find the convex hull to get the outer boundary
        from scipy.spatial import ConvexHull
        
        try:
            # Get convex hull
            hull = ConvexHull(positions_array)
            hull_points = positions_array[hull.vertices]
            
            # Convert back to list of tuples
            track_points = [(float(x), float(y)) for x, y in hull_points]
            
            # Add some points to make it smoother
            smoothed_points = []
            for i in range(len(track_points)):
                smoothed_points.append(track_points[i])
                if i < len(track_points) - 1:
                    # Add intermediate points for smoother curves
                    x1, y1 = track_points[i]
                    x2, y2 = track_points[i + 1]
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    smoothed_points.append((mid_x, mid_y))
            
            return smoothed_points
            
        except ImportError:
            # Fallback: create a simple bounding box
            print("scipy not available, using simple bounding box")
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            
            # Create a rectangular track
            margin = 50  # meters
            return [
                (min_x - margin, min_y - margin),
                (max_x + margin, min_y - margin),
                (max_x + margin, max_y + margin),
                (min_x - margin, max_y + margin)
            ]
        
    def load_track_data(self, session) -> bool:
        """Load track outline data from FastF1 session."""
        try:
            # Try multiple methods to get track data
            track_points = None
            
            # Method 1: Try session.track
            if hasattr(session, 'track') and session.track is not None:
                track = session.track
                if hasattr(track, 'x') and hasattr(track, 'y'):
                    track_points = list(zip(track.x, track.y))
                    print("Loaded track data from session.track")
            
            # Method 2: Try to get track from circuit info
            if not track_points and hasattr(session, 'event'):
                try:
                    circuit_info = session.event.get_circuit_info()
                    if hasattr(circuit_info, 'corners'):
                        # Use corner points to create track outline
                        corners = circuit_info.corners
                        if len(corners) > 0:
                            track_points = [(corner['X'], corner['Y']) for corner in corners]
                            print("Loaded track data from circuit corners")
                except:
                    pass
            
            # Method 3: Try to extract from position data (create approximate track)
            if not track_points:
                print("No direct track data found, creating approximate track from position data...")
                # Get all position data to create an approximate track outline
                all_positions = []
                for driver_num in session.drivers:
                    try:
                        pos_data = session.pos_data[driver_num]
                        if not pos_data.empty and 'X' in pos_data.columns and 'Y' in pos_data.columns:
                            # Filter to on-track positions
                            on_track = pos_data[pos_data['Status'] == 'OnTrack']
                            if not on_track.empty:
                                all_positions.extend(list(zip(on_track['X'], on_track['Y'])))
                    except:
                        continue
                
                if all_positions:
                    # Create a simplified track outline by finding the outer boundary
                    track_points = self._create_track_from_positions(all_positions)
                    print(f"Created approximate track from {len(all_positions)} position points")
            
            if not track_points:
                print("No track data available in session")
                return False
            
            self.track_points = track_points
            
            if not self.track_points:
                print("No track points found")
                return False
            
            # Calculate track bounds
            x_coords = [p[0] for p in self.track_points]
            y_coords = [p[1] for p in self.track_points]
            
            self.track_bounds = {
                'min_x': min(x_coords),
                'max_x': max(x_coords),
                'min_y': min(y_coords),
                'max_y': max(y_coords)
            }
            
            # Calculate scale and offset to fit track on screen
            track_width = self.track_bounds['max_x'] - self.track_bounds['min_x']
            track_height = self.track_bounds['max_y'] - self.track_bounds['min_y']
            
            # Add padding
            padding = 100
            scale_x = (self.width - 2 * padding) / track_width
            scale_y = (self.height - 2 * padding) / track_height
            self.scale_factor = min(scale_x, scale_y)
            
            # Center the track
            self.offset_x = (self.width - track_width * self.scale_factor) / 2 - self.track_bounds['min_x'] * self.scale_factor
            self.offset_y = (self.height - track_height * self.scale_factor) / 2 - self.track_bounds['min_y'] * self.scale_factor
            
            print(f"Track loaded: {len(self.track_points)} points")
            print(f"Bounds: {self.track_bounds}")
            print(f"Scale factor: {self.scale_factor:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error loading track data: {e}")
            return False
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale_factor + self.offset_x)
        screen_y = int(y * self.scale_factor + self.offset_y)
        return screen_x, screen_y
    
    def draw_grass_background(self):
        """Draw grass background covering the entire screen."""
        self.screen.fill(self.GRASS_COLOR)
    
    def draw_track_surface(self):
        """Draw the tarmac track surface."""
        if not self.track_points:
            return
        
        # Create a polygon for the track surface
        # We'll make the track wider by offsetting the centerline
        track_width = 12  # meters (typical F1 track width)
        half_width = track_width / 2
        
        # Create left and right edges of the track
        left_edge = []
        right_edge = []
        
        for i, (x, y) in enumerate(self.track_points):
            # Calculate direction vector
            if i == 0:
                # First point - use direction to next point
                next_x, next_y = self.track_points[1]
                dx = next_x - x
                dy = next_y - y
            elif i == len(self.track_points) - 1:
                # Last point - use direction from previous point
                prev_x, prev_y = self.track_points[i-1]
                dx = x - prev_x
                dy = y - prev_y
            else:
                # Middle point - average direction
                prev_x, prev_y = self.track_points[i-1]
                next_x, next_y = self.track_points[i+1]
                dx = (next_x - prev_x) / 2
                dy = (next_y - prev_y) / 2
            
            # Normalize direction vector
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
            
            # Calculate perpendicular vector (normal)
            nx = -dy
            ny = dx
            
            # Offset points
            left_x = x + nx * half_width
            left_y = y + ny * half_width
            right_x = x - nx * half_width
            right_y = y - ny * half_width
            
            left_edge.append((left_x, left_y))
            right_edge.append((right_x, right_y))
        
        # Create track polygon (left edge + reversed right edge)
        track_polygon = left_edge + list(reversed(right_edge))
        
        # Convert to screen coordinates
        screen_points = [self.world_to_screen(x, y) for x, y in track_polygon]
        
        # Draw the track surface
        if len(screen_points) >= 3:
            pygame.draw.polygon(self.screen, self.TARMAC_COLOR, screen_points)
    
    def draw_curbs(self):
        """Draw red and white curbs at track edges."""
        if not self.track_points:
            return
        
        curb_width = 0.5  # meters
        curb_height = 0.1  # meters (visual height)
        
        for i, (x, y) in enumerate(self.track_points):
            # Calculate direction and normal (same as track surface)
            if i == 0:
                next_x, next_y = self.track_points[1]
                dx = next_x - x
                dy = next_y - y
            elif i == len(self.track_points) - 1:
                prev_x, prev_y = self.track_points[i-1]
                dx = x - prev_x
                dy = y - prev_y
            else:
                prev_x, prev_y = self.track_points[i-1]
                next_x, next_y = self.track_points[i+1]
                dx = (next_x - prev_x) / 2
                dy = (next_y - prev_y) / 2
            
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
            
            nx = -dy
            ny = dx
            
            # Determine curb color (alternate red/white)
            curb_color = self.CURB_RED if (i // 10) % 2 == 0 else self.CURB_WHITE
            
            # Draw left curb
            left_x = x + nx * (6 + curb_width/2)  # 6m from centerline
            left_y = y + ny * (6 + curb_width/2)
            left_screen = self.world_to_screen(left_x, left_y)
            
            # Draw right curb
            right_x = x - nx * (6 + curb_width/2)
            right_y = y - ny * (6 + curb_width/2)
            right_screen = self.world_to_screen(right_x, right_y)
            
            # Draw curb rectangles
            curb_rect_size = int(curb_width * self.scale_factor)
            if curb_rect_size > 0:
                pygame.draw.rect(self.screen, curb_color, 
                               (left_screen[0] - curb_rect_size//2, left_screen[1] - curb_rect_size//2,
                                curb_rect_size, curb_rect_size))
                pygame.draw.rect(self.screen, curb_color, 
                               (right_screen[0] - curb_rect_size//2, right_screen[1] - curb_rect_size//2,
                                curb_rect_size, curb_rect_size))
    
    def draw_center_line(self):
        """Draw the yellow center line."""
        if not self.track_points:
            return
        
        screen_points = [self.world_to_screen(x, y) for x, y in self.track_points]
        
        # Draw dashed center line
        dash_length = 10
        gap_length = 5
        
        for i in range(len(screen_points) - 1):
            start = screen_points[i]
            end = screen_points[i + 1]
            
            # Calculate distance and number of dashes
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                num_dashes = int(distance / (dash_length + gap_length))
                
                for j in range(num_dashes):
                    t1 = j * (dash_length + gap_length) / distance
                    t2 = (j * (dash_length + gap_length) + dash_length) / distance
                    
                    if t2 > 1:
                        t2 = 1
                    
                    dash_start = (int(start[0] + dx * t1), int(start[1] + dy * t1))
                    dash_end = (int(start[0] + dx * t2), int(start[1] + dy * t2))
                    
                    pygame.draw.line(self.screen, self.TRACK_LINE, dash_start, dash_end, 2)
    
    def draw_cars(self, car1_pos: Optional[Tuple[float, float]], 
                  car2_pos: Optional[Tuple[float, float]],
                  car1_color: Tuple[int, int, int] = (0, 100, 255),
                  car2_color: Tuple[int, int, int] = (255, 100, 100)):
        """Draw cars at their positions."""
        car_size = int(3 * self.scale_factor)  # 3 meter car length
        
        if car1_pos:
            screen_pos = self.world_to_screen(car1_pos[0], car1_pos[1])
            pygame.draw.circle(self.screen, car1_color, screen_pos, car_size)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, car_size, 2)
        
        if car2_pos:
            screen_pos = self.world_to_screen(car2_pos[0], car2_pos[1])
            pygame.draw.circle(self.screen, car2_color, screen_pos, car_size)
            pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, car_size, 2)
    
    def render_frame(self, car1_pos: Optional[Tuple[float, float]] = None,
                    car2_pos: Optional[Tuple[float, float]] = None):
        """Render a complete frame."""
        self.draw_grass_background()
        self.draw_track_surface()
        self.draw_curbs()
        self.draw_center_line()
        self.draw_cars(car1_pos, car2_pos)
    
    def save_frame(self, filename: str):
        """Save current frame to file."""
        pygame.image.save(self.screen, filename)
    
    def run_animation(self, car1_data: pd.DataFrame, car2_data: pd.DataFrame,
                     driver1: str, driver2: str, fps: int = 30):
        """Run animation with car data."""
        if car1_data.empty or car2_data.empty:
            print("No car data available for animation")
            return
        
        running = True
        frame_count = 0
        
        # Filter to on-track data
        car1_track = car1_data[car1_data['Status'] == 'OnTrack']
        car2_track = car2_data[car2_data['Status'] == 'OnTrack']
        
        max_frames = max(len(car1_track), len(car2_track))
        
        print(f"Starting animation: {max_frames} frames at {fps} FPS")
        
        while running and frame_count < max_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get car positions for current frame
            car1_pos = None
            car2_pos = None
            
            if frame_count < len(car1_track):
                car1_pos = (car1_track['X'].iloc[frame_count], car1_track['Y'].iloc[frame_count])
            
            if frame_count < len(car2_track):
                car2_pos = (car2_track['X'].iloc[frame_count], car2_track['Y'].iloc[frame_count])
            
            # Render frame
            self.render_frame(car1_pos, car2_pos)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(fps)
            
            frame_count += 1
        
        pygame.quit()


def get_fastest_lap_data(session, driver_code: str) -> Optional[pd.DataFrame]:
    """Get fastest lap data for a driver."""
    try:
        results = session.results
        driver_result = results.loc[results['Abbreviation'] == driver_code]
        
        if driver_result.empty:
            print(f"Driver {driver_code} not found in session")
            return None
        
        driver_number = driver_result.iloc[0]['DriverNumber']
        driver_laps = session.laps.pick_drivers(driver_number)
        
        if driver_laps.empty:
            print(f"No laps found for driver {driver_code}")
            return None
        
        fastest_lap = driver_laps.pick_fastest()
        telemetry = fastest_lap.get_telemetry()
        
        if telemetry is None or telemetry.empty:
            print(f"No telemetry found for fastest lap of {driver_code}")
            return None
        
        # Check for X/Y coordinates
        if 'X' in telemetry.columns and 'Y' in telemetry.columns:
            return pd.DataFrame({
                'X': telemetry['X'],
                'Y': telemetry['Y'],
                'Status': 'OnTrack'
            })
        else:
            print(f"No X/Y coordinates in telemetry for {driver_code}")
            return None
            
    except Exception as e:
        print(f"Error getting fastest lap data for driver {driver_code}: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Render F1 track with pygame visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pygame_track_renderer.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA"
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
    parser.add_argument('--width', type=int, default=1920,
                       help='Screen width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Screen height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Animation FPS (default: 30)')
    
    args = parser.parse_args()
    
    # Convert driver codes to uppercase
    driver1 = args.driver1.upper()
    driver2 = args.driver2.upper()
    
    print(f"Loading F1 session data for {args.year} {args.race}...")
    
    # Load session data
    try:
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(cache_dir)
        
        session = fastf1.get_session(args.year, args.race, 'Q')
        session.load()
        
        # Get fastest lap data
        car1_data = get_fastest_lap_data(session, driver1)
        car2_data = get_fastest_lap_data(session, driver2)
        
        if car1_data is None or car2_data is None:
            print("Failed to load car data")
            sys.exit(1)
        
        # Create track renderer
        renderer = TrackRenderer(args.width, args.height)
        
        # Load track data
        if not renderer.load_track_data(session):
            print("Failed to load track data")
            sys.exit(1)
        
        # Run animation
        renderer.run_animation(car1_data, car2_data, driver1, driver2, args.fps)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
