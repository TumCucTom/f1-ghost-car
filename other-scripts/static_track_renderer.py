#!/usr/bin/env python3
"""
Static Pygame Track Renderer for F1 Data Visualization

This script creates a realistic static track visualization using FastF1 track data
with tarmac, grass, and curbs rendered in pygame.

Usage:
    python static_track_renderer.py --year 2024 --race "Hungary"
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


class StaticTrackRenderer:
    def __init__(self, width: int = 1920, height: int = 1080):
        """Initialize the pygame track renderer."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("F1 Track Visualization")
        
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
        try:
            from scipy.spatial import ConvexHull
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
    
    def _create_detailed_track_from_positions(self, positions: List[Tuple[float, float]], 
                                            position_details: List[dict]) -> List[Tuple[float, float]]:
        """Create a detailed track outline from comprehensive position data."""
        if not positions:
            return []
        
        print(f"Creating detailed track from {len(positions)} position points...")
        
        # Convert to numpy arrays
        positions_array = np.array(positions)
        
        # Method 1: Use DBSCAN clustering to find track centerline
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            # Cluster positions to find the main track path
            scaler = StandardScaler()
            positions_scaled = scaler.fit_transform(positions_array)
            
            # Use DBSCAN to find clusters (track segments)
            clustering = DBSCAN(eps=0.1, min_samples=10).fit(positions_scaled)
            labels = clustering.labels_
            
            # Find the largest cluster (main track)
            unique_labels = np.unique(labels)
            largest_cluster_size = 0
            main_cluster_label = -1
            
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                cluster_size = np.sum(labels == label)
                if cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    main_cluster_label = label
            
            if main_cluster_label != -1:
                main_track_points = positions_array[labels == main_cluster_label]
                print(f"Found main track cluster with {len(main_track_points)} points")
                
                # Sort points to create a continuous track
                track_centerline = self._sort_track_points(main_track_points)
                return track_centerline
                
        except ImportError:
            print("sklearn not available, using alternative method")
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
        
        # Method 2: Use position density to find track centerline
        try:
            # Create a grid and find high-density areas
            x_coords = positions_array[:, 0]
            y_coords = positions_array[:, 1]
            
            # Create bins for density analysis
            x_bins = np.linspace(x_coords.min(), x_coords.max(), 100)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), 100)
            
            # Count positions in each bin
            hist, x_edges, y_edges = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
            
            # Find high-density areas (track centerline)
            threshold = np.percentile(hist, 80)  # Top 20% density
            high_density_mask = hist > threshold
            
            # Get coordinates of high-density areas
            y_indices, x_indices = np.where(high_density_mask)
            track_points = []
            
            for i, j in zip(x_indices, y_indices):
                x = (x_edges[j] + x_edges[j+1]) / 2
                y = (y_edges[i] + y_edges[i+1]) / 2
                track_points.append((x, y))
            
            if track_points:
                track_array = np.array(track_points)
                sorted_track = self._sort_track_points(track_array)
                print(f"Created track from density analysis: {len(sorted_track)} points")
                return sorted_track
                
        except Exception as e:
            print(f"Density analysis failed: {e}")
        
        # Method 3: Fallback to convex hull with more detail
        return self._create_track_from_positions(positions)
    
    def _sort_track_points(self, points: np.ndarray) -> List[Tuple[float, float]]:
        """Sort track points to create a continuous path."""
        if len(points) < 2:
            return [(float(x), float(y)) for x, y in points]
        
        # Start with the point closest to the center
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        start_idx = np.argmin(distances)
        
        sorted_points = [points[start_idx]]
        remaining_points = np.delete(points, start_idx, axis=0)
        
        # Greedily find the next closest point
        while len(remaining_points) > 0:
            last_point = sorted_points[-1]
            distances = np.linalg.norm(remaining_points - last_point, axis=1)
            next_idx = np.argmin(distances)
            sorted_points.append(remaining_points[next_idx])
            remaining_points = np.delete(remaining_points, next_idx, axis=0)
        
        return [(float(x), float(y)) for x, y in sorted_points]
    
    def _interpolate_track_from_corners(self, corner_points: List[Tuple[float, float]], points_per_segment: int = 20) -> List[Tuple[float, float]]:
        """Create a detailed track by interpolating between corner points."""
        if len(corner_points) < 2:
            return corner_points
        
        detailed_points = []
        
        # Close the track by adding the first point at the end
        closed_corners = corner_points + [corner_points[0]]
        
        for i in range(len(closed_corners) - 1):
            start_point = closed_corners[i]
            end_point = closed_corners[i + 1]
            
            # Add the start point
            detailed_points.append(start_point)
            
            # Interpolate between start and end points
            for j in range(1, points_per_segment):
                t = j / points_per_segment
                x = start_point[0] + t * (end_point[0] - start_point[0])
                y = start_point[1] + t * (end_point[1] - start_point[1])
                detailed_points.append((x, y))
        
        return detailed_points
    
    def _reconstruct_track_from_driver_positions(self, session) -> Optional[List[Tuple[float, float]]]:
        """Reconstruct track outline from sampled driver position data."""
        try:
            print("Collecting sampled driver position data...")
            all_positions = []
            sample_rate = 10  # Take every 10th point to reduce data size
            
            for driver_num in session.drivers:
                try:
                    pos_data = session.pos_data[driver_num]
                    if not pos_data.empty and 'X' in pos_data.columns and 'Y' in pos_data.columns:
                        # Get all on-track positions
                        on_track = pos_data[pos_data['Status'] == 'OnTrack']
                        if not on_track.empty:
                            # Sample the data to reduce size
                            sampled_data = on_track.iloc[::sample_rate]
                            positions = list(zip(sampled_data['X'], sampled_data['Y']))
                            all_positions.extend(positions)
                            print(f"  Driver {driver_num}: {len(positions)} sampled positions (from {len(on_track)})")
                except Exception as e:
                    print(f"  Error processing driver {driver_num}: {e}")
                    continue
            
            if not all_positions:
                print("No position data found")
                return None
            
            print(f"Total sampled position points: {len(all_positions)}")
            
            # Use the advanced track reconstruction method
            track_points = self._create_detailed_track_from_positions(all_positions, [])
            return track_points
            
        except Exception as e:
            print(f"Error reconstructing track from driver positions: {e}")
            return None
    
    def _extract_track_from_object(self, obj) -> Optional[List[Tuple[float, float]]]:
        """Try to extract track coordinates from various object types."""
        try:
            # If it's a list/array of points
            if hasattr(obj, '__len__') and len(obj) > 0:
                first_item = obj[0]
                
                # Dictionary format: {'X': x, 'Y': y}
                if isinstance(first_item, dict):
                    if 'X' in first_item and 'Y' in first_item:
                        return [(point['X'], point['Y']) for point in obj]
                    elif 'x' in first_item and 'y' in first_item:
                        return [(point['x'], point['y']) for point in obj]
                
                # Object with x/y attributes
                elif hasattr(first_item, 'x') and hasattr(first_item, 'y'):
                    return [(point.x, point.y) for point in obj]
                
                # Tuple/list format: [(x, y), ...]
                elif isinstance(first_item, (tuple, list)) and len(first_item) >= 2:
                    return [(point[0], point[1]) for point in obj]
            
            # If it's a single object with x/y attributes
            elif hasattr(obj, 'x') and hasattr(obj, 'y'):
                return [(obj.x, obj.y)]
            
            # If it has x/y arrays
            elif hasattr(obj, 'x') and hasattr(obj, 'y'):
                x_data = obj.x
                y_data = obj.y
                if hasattr(x_data, '__len__') and hasattr(y_data, '__len__'):
                    return list(zip(x_data, y_data))
            
            return None
            
        except Exception as e:
            print(f"Error extracting track from object: {e}")
            return None
        
    def load_track_data(self, session) -> bool:
        """Load official track outline data from FastF1 API."""
        try:
            track_points = None
            
            print("Attempting to load official track outline from FastF1 API...")
            
            # Method 1: Try session.track (most direct)
            if hasattr(session, 'track') and session.track is not None:
                track = session.track
                print(f"Session track object: {type(track)}")
                print(f"Track attributes: {[attr for attr in dir(track) if not attr.startswith('_')]}")
                
                if hasattr(track, 'x') and hasattr(track, 'y'):
                    track_points = list(zip(track.x, track.y))
                    print(f"Loaded track data from session.track: {len(track_points)} points")
                else:
                    print("Track object doesn't have x/y coordinates")
            
            # Method 2: Try circuit info for track outline (session has get_circuit_info)
            if not track_points:
                try:
                    circuit_info = session.get_circuit_info()
                    print(f"Circuit info type: {type(circuit_info)}")
                    print(f"Circuit info attributes: {[attr for attr in dir(circuit_info) if not attr.startswith('_')]}")
                    
                    # Try different ways to get track outline - prioritize marshal data for more detail
                    track_data_sources = [
                        ('marshal_sectors', 'marshal sectors'),
                        ('marshal_lights', 'marshal lights'), 
                        ('corners', 'corners')
                    ]
                    
                    for data_attr, data_name in track_data_sources:
                        if hasattr(circuit_info, data_attr):
                            data = getattr(circuit_info, data_attr)
                            print(f"{data_name.title()} data: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                            
                            if hasattr(data, 'columns'):
                                print(f"{data_name.title()} columns: {data.columns.tolist()}")
                                if len(data) > 0:
                                    print(f"First {data_name[:-1]}: {data.iloc[0].to_dict()}")
                                    
                                    if 'X' in data.columns and 'Y' in data.columns:
                                        # Get points from this data source
                                        points = [(row['X'], row['Y']) for _, row in data.iterrows()]
                                        
                                        if data_attr == 'corners':
                                            # For corners, interpolate for smooth curves
                                            track_points = self._interpolate_track_from_corners(points, points_per_segment=15)
                                            print(f"Loaded track from {data_name}: {len(points)} points, {len(track_points)} interpolated")
                                        else:
                                            # For marshal data, use directly as it's more detailed
                                            track_points = points
                                            print(f"Loaded track from {data_name}: {len(track_points)} points")
                                            
                                            # If we have marshal data but want even more detail, try driver reconstruction
                                            if len(track_points) < 50:  # If not very detailed
                                                print("Marshal data found but may not be detailed enough, trying driver position reconstruction...")
                                                driver_track = self._reconstruct_track_from_driver_positions(session)
                                                if driver_track and len(driver_track) > len(track_points):
                                                    track_points = driver_track
                                                    print(f"Using driver-reconstructed track instead: {len(track_points)} points")
                                        break
                            elif isinstance(data[0], dict) and 'X' in data[0] and 'Y' in data[0]:
                                track_points = [(point['X'], point['Y']) for point in data]
                                print(f"Loaded track from {data_name}: {len(track_points)} points")
                                break
                    
                    # Try to find track outline in other attributes
                    for attr_name in ['track_outline', 'circuit_outline', 'track_boundary', 'track_perimeter', 'track_points', 'waypoints', 'segments', 'sectors']:
                        if hasattr(circuit_info, attr_name):
                            data = getattr(circuit_info, attr_name)
                            print(f"Found {attr_name}: {type(data)}")
                            if data is not None and hasattr(data, '__len__') and len(data) > 0:
                                # Try to extract coordinates
                                if isinstance(data[0], dict):
                                    if 'X' in data[0] and 'Y' in data[0]:
                                        track_points = [(point['X'], point['Y']) for point in data]
                                        print(f"Loaded track from {attr_name}: {len(track_points)} points")
                                        break
                                elif hasattr(data[0], 'x') and hasattr(data[0], 'y'):
                                    track_points = [(point.x, point.y) for point in data]
                                    print(f"Loaded track from {attr_name}: {len(track_points)} points")
                                    break
                    
                    # Check if we can get more detailed track data
                    print(f"Circuit info detailed inspection:")
                    for attr in dir(circuit_info):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(circuit_info, attr)
                                if hasattr(value, '__len__') and len(value) > 0:
                                    print(f"  {attr}: {type(value)} with {len(value)} items")
                                    if hasattr(value, 'columns'):
                                        print(f"    Columns: {value.columns.tolist()}")
                                else:
                                    print(f"  {attr}: {type(value)} = {value}")
                            except Exception as e:
                                print(f"  {attr}: Error accessing - {e}")
                    
                except Exception as e:
                    print(f"Circuit info method failed: {e}")
            
            # Method 3: Try to access track data through different session attributes
            if not track_points:
                print("Checking session for track-related attributes...")
                session_attrs = [attr for attr in dir(session) if not attr.startswith('_')]
                print(f"Session attributes: {session_attrs}")
                
                for attr_name in ['circuit', 'track_layout', 'track_data', 'circuit_data']:
                    if hasattr(session, attr_name):
                        data = getattr(session, attr_name)
                        print(f"Found {attr_name}: {type(data)}")
                        if data is not None:
                            # Try to extract track points from this data
                            track_points = self._extract_track_from_object(data)
                            if track_points:
                                print(f"Loaded track from {attr_name}: {len(track_points)} points")
                                break
            
            # Method 4: Try event-level track data
            if not track_points and hasattr(session, 'event'):
                try:
                    event = session.event
                    print(f"Event type: {type(event)}")
                    print(f"Event attributes: {[attr for attr in dir(event) if not attr.startswith('_')]}")
                    
                    # Try to get track data from event
                    for attr_name in ['track', 'circuit', 'track_layout']:
                        if hasattr(event, attr_name):
                            data = getattr(event, attr_name)
                            print(f"Event {attr_name}: {type(data)}")
                            track_points = self._extract_track_from_object(data)
                            if track_points:
                                print(f"Loaded track from event.{attr_name}: {len(track_points)} points")
                                break
                except Exception as e:
                    print(f"Event track data access failed: {e}")
            
            # Method 5: Fallback to driver position data to reconstruct track outline
            if not track_points:
                print("No official track outline found, trying to reconstruct from driver positions...")
                track_points = self._reconstruct_track_from_driver_positions(session)
                if track_points:
                    print(f"Reconstructed track from driver positions: {len(track_points)} points")
            
            if not track_points:
                print("No track outline data found in FastF1 API")
                print("Available data sources checked:")
                print("- session.track")
                print("- session.get_circuit_info()")
                print("- session circuit/track attributes")
                print("- event track attributes")
                print("- driver position reconstruction")
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
        curb_spacing = 2.0  # meters between curb segments
        
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
            
            # Determine curb color (alternate red/white based on distance along track)
            distance_along_track = i * curb_spacing
            curb_color = self.CURB_RED if int(distance_along_track / 10) % 2 == 0 else self.CURB_WHITE
            
            # Draw left curb
            left_x = x + nx * (6 + curb_width/2)  # 6m from centerline
            left_y = y + ny * (6 + curb_width/2)
            left_screen = self.world_to_screen(left_x, left_y)
            
            # Draw right curb
            right_x = x - nx * (6 + curb_width/2)
            right_y = y - ny * (6 + curb_width/2)
            right_screen = self.world_to_screen(right_x, right_y)
            
            # Draw curb rectangles
            curb_rect_size = max(1, int(curb_width * self.scale_factor))
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
    
    def render_track(self):
        """Render the complete track."""
        self.draw_grass_background()
        self.draw_track_surface()
        self.draw_curbs()
        self.draw_center_line()
    
    def save_track_image(self, filename: str):
        """Save current track image to file."""
        pygame.image.save(self.screen, filename)
        print(f"Track image saved to: {filename}")
    
    def show_track(self):
        """Display the track and wait for user to close."""
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        # Save image when 's' is pressed
                        filename = f"track_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        self.save_track_image(filename)
            
            # Keep the display updated
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Render static F1 track with pygame visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python static_track_renderer.py --year 2024 --race "Hungary"
  python static_track_renderer.py --year 2024 --race "Monaco" --width 2560 --height 1440
        """
    )
    
    parser.add_argument('--year', type=int, required=True,
                       help='Year of the race (e.g., 2024)')
    parser.add_argument('--race', type=str, required=True,
                       help='Name of the Grand Prix (e.g., "Monaco", "Silverstone")')
    parser.add_argument('--width', type=int, default=1920,
                       help='Screen width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080,
                       help='Screen height (default: 1080)')
    parser.add_argument('--save', action='store_true',
                       help='Save track image and exit')
    
    args = parser.parse_args()
    
    print(f"Loading F1 session data for {args.year} {args.race}...")
    
    # Load session data
    try:
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(cache_dir)
        
        session = fastf1.get_session(args.year, args.race, 'Q')
        session.load()
        
        # Create track renderer
        renderer = StaticTrackRenderer(args.width, args.height)
        
        # Load track data
        if not renderer.load_track_data(session):
            print("Failed to load track data")
            sys.exit(1)
        
        # Render track
        renderer.render_track()
        
        if args.save:
            # Save and exit
            filename = f"track_{args.year}_{args.race.replace(' ', '_')}.png"
            renderer.save_track_image(filename)
        else:
            # Show interactive window
            print("Track rendered! Press 's' to save, ESC or close window to exit")
            renderer.show_track()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
