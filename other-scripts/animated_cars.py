#!/usr/bin/env python3
"""
Animated F1 Cars on Track
Animates car sprites moving around the track using FastF1 position data.
"""

import argparse
import os
import sys
import pygame
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import fastf1
from fastf1 import get_session

# Initialize pygame
pygame.init()

class CarSprite:
    """Simple car sprite for animation."""
    
    def __init__(self, color: Tuple[int, int, int], size: int = 8):
        self.color = color
        self.size = size
        self.x = 0
        self.y = 0
        self.angle = 0
        
    def draw(self, screen: pygame.Surface):
        """Draw the car sprite."""
        # Draw car body (rectangle)
        car_rect = pygame.Rect(self.x - self.size//2, self.y - self.size//3, self.size, self.size//1.5)
        pygame.draw.rect(screen, self.color, car_rect)
        
        # Draw car direction indicator (small triangle)
        points = [
            (self.x + self.size//2, self.y),
            (self.x + self.size//2 + 4, self.y - 2),
            (self.x + self.size//2 + 4, self.y + 2)
        ]
        pygame.draw.polygon(screen, (255, 255, 255), points)

class AnimatedTrackRenderer:
    """Renders animated cars on track."""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("F1 Cars Animation")
        
        # Colors
        self.BACKGROUND_COLOR = (255, 255, 255)  # White background
        self.TRACK_COLOR = (60, 60, 60)  # Dark grey track
        self.CENTER_LINE_COLOR = (255, 255, 0)  # Yellow center line
        
        # Car colors
        self.NOR_COLOR = (255, 165, 0)  # Orange for NOR
        self.PIA_COLOR = (80, 80, 80)   # Dark grey for PIA
        
        # Track data
        self.track_points = []
        self.track_bounds = {}
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Animation data
        self.nor_positions = []
        self.pia_positions = []
        self.current_frame = 0
        self.max_frames = 0
        
        # Car sprites
        self.nor_car = CarSprite(self.NOR_COLOR, 12)
        self.pia_car = CarSprite(self.PIA_COLOR, 12)
        
        # Trail data
        self.nor_trail = []
        self.pia_trail = []
        self.max_trail_length = 50
        
    def load_session_data(self, year: int, race: str) -> bool:
        """Load F1 session data."""
        try:
            print(f"Loading F1 session data for {year} {race}...")
            
            # Enable caching
            cache_dir = 'cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            fastf1.Cache.enable_cache(cache_dir)
            
            # Get session
            session = get_session(year, race, 'Q')
            session.load()
            
            print(f"Session loaded: {session.name}")
            print(f"Drivers: {session.drivers}")
            
            # Load track data
            if not self.load_track_data(session):
                return False
            
            # Load driver position data
            if not self.load_driver_data(session, 'NOR', 'PIA'):
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading session data: {e}")
            return False
    
    def load_track_data(self, session) -> bool:
        """Load track outline data."""
        try:
            # Try to get track data from circuit info
            circuit_info = session.get_circuit_info()
            
            # Use marshal sectors for track outline (most detailed)
            if hasattr(circuit_info, 'marshal_sectors'):
                marshal_data = circuit_info.marshal_sectors
                if not marshal_data.empty and 'X' in marshal_data.columns and 'Y' in marshal_data.columns:
                    self.track_points = [(row['X'], row['Y']) for _, row in marshal_data.iterrows()]
                    print(f"Loaded track from marshal sectors: {len(self.track_points)} points")
                else:
                    # Fallback to corners
                    corners = circuit_info.corners
                    if not corners.empty and 'X' in corners.columns and 'Y' in corners.columns:
                        corner_points = [(row['X'], row['Y']) for _, row in corners.iterrows()]
                        self.track_points = self.interpolate_track(corner_points)
                        print(f"Loaded track from corners: {len(corner_points)} corners, {len(self.track_points)} interpolated")
            else:
                print("No track data available")
                return False
            
            if not self.track_points:
                return False
            
            # Calculate bounds and scaling
            self.calculate_track_scaling()
            return True
            
        except Exception as e:
            print(f"Error loading track data: {e}")
            return False
    
    def interpolate_track(self, corner_points: List[Tuple[float, float]], points_per_segment: int = 15) -> List[Tuple[float, float]]:
        """Interpolate between corner points for smooth track."""
        if len(corner_points) < 2:
            return corner_points
        
        detailed_points = []
        closed_corners = corner_points + [corner_points[0]]
        
        for i in range(len(closed_corners) - 1):
            start_point = closed_corners[i]
            end_point = closed_corners[i + 1]
            
            detailed_points.append(start_point)
            
            for j in range(1, points_per_segment):
                t = j / points_per_segment
                x = start_point[0] + t * (end_point[0] - start_point[0])
                y = start_point[1] + t * (end_point[1] - start_point[1])
                detailed_points.append((x, y))
        
        return detailed_points
    
    def calculate_track_scaling(self):
        """Calculate scaling and offset to fit track on screen."""
        if not self.track_points:
            return
        
        x_coords = [point[0] for point in self.track_points]
        y_coords = [point[1] for point in self.track_points]
        
        self.track_bounds = {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
        
        # Calculate scale with padding
        track_width = self.track_bounds['max_x'] - self.track_bounds['min_x']
        track_height = self.track_bounds['max_y'] - self.track_bounds['min_y']
        
        padding = 100
        scale_x = (self.width - 2 * padding) / track_width
        scale_y = (self.height - 2 * padding) / track_height
        self.scale_factor = min(scale_x, scale_y)
        
        # Center the track
        self.offset_x = (self.width - track_width * self.scale_factor) / 2 - self.track_bounds['min_x'] * self.scale_factor
        self.offset_y = (self.height - track_height * self.scale_factor) / 2 - self.track_bounds['min_y'] * self.scale_factor
        
        print(f"Track bounds: {self.track_bounds}")
        print(f"Scale factor: {self.scale_factor:.3f}")
    
    def load_driver_data(self, session, driver1: str, driver2: str) -> bool:
        """Load position data for two drivers."""
        try:
            # Get driver numbers
            driver1_num = None
            driver2_num = None
            
            for driver_num in session.drivers:
                try:
                    driver_info = session.get_driver(driver_num)
                    if driver_info['Abbreviation'] == driver1:
                        driver1_num = driver_num
                    elif driver_info['Abbreviation'] == driver2:
                        driver2_num = driver_num
                except:
                    continue
            
            if not driver1_num or not driver2_num:
                print(f"Could not find drivers {driver1} and {driver2}")
                return False
            
            print(f"Found drivers: {driver1} (#{driver1_num}), {driver2} (#{driver2_num})")
            
            # Load position data
            self.nor_positions = self.get_driver_positions(session, driver1_num)
            self.pia_positions = self.get_driver_positions(session, driver2_num)
            
            if not self.nor_positions or not self.pia_positions:
                print("No position data found")
                return False
            
            # Sync the data lengths
            min_length = min(len(self.nor_positions), len(self.pia_positions))
            self.nor_positions = self.nor_positions[:min_length]
            self.pia_positions = self.pia_positions[:min_length]
            self.max_frames = min_length
            
            print(f"Loaded {len(self.nor_positions)} positions for {driver1}")
            print(f"Loaded {len(self.pia_positions)} positions for {driver2}")
            print(f"Animation will have {self.max_frames} frames")
            
            return True
            
        except Exception as e:
            print(f"Error loading driver data: {e}")
            return False
    
    def get_driver_positions(self, session, driver_num: str) -> List[Tuple[float, float]]:
        """Get position data for a specific driver."""
        try:
            pos_data = session.pos_data[driver_num]
            if pos_data.empty or 'X' not in pos_data.columns or 'Y' not in pos_data.columns:
                return []
            
            # Get on-track positions
            on_track = pos_data[pos_data['Status'] == 'OnTrack']
            if on_track.empty:
                return []
            
            # Sample every 5th point to reduce data size
            sampled_data = on_track.iloc[::5]
            positions = [(row['X'], row['Y']) for _, row in sampled_data.iterrows()]
            
            return positions
            
        except Exception as e:
            print(f"Error getting positions for driver {driver_num}: {e}")
            return []
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.scale_factor + self.offset_x)
        screen_y = int(y * self.scale_factor + self.offset_y)
        return screen_x, screen_y
    
    def draw_track(self):
        """Draw the track outline."""
        if not self.track_points:
            return
        
        # Convert track points to screen coordinates
        screen_points = [self.world_to_screen(x, y) for x, y in self.track_points]
        
        # Draw track surface
        if len(screen_points) > 2:
            pygame.draw.polygon(self.screen, self.TRACK_COLOR, screen_points)
        
        # Draw center line
        if len(screen_points) > 2:
            pygame.draw.lines(self.screen, self.CENTER_LINE_COLOR, True, screen_points, 2)
    
    def draw_trails(self):
        """Draw car trails."""
        # Draw NOR trail (orange)
        if len(self.nor_trail) > 1:
            pygame.draw.lines(self.screen, self.NOR_COLOR, False, self.nor_trail, 3)
        
        # Draw PIA trail (dark grey)
        if len(self.pia_trail) > 1:
            pygame.draw.lines(self.screen, self.PIA_COLOR, False, self.pia_trail, 3)
    
    def update_cars(self):
        """Update car positions and trails."""
        if self.current_frame >= self.max_frames:
            self.current_frame = 0  # Loop animation
        
        # Update NOR car
        if self.current_frame < len(self.nor_positions):
            nor_x, nor_y = self.nor_positions[self.current_frame]
            screen_x, screen_y = self.world_to_screen(nor_x, nor_y)
            self.nor_car.x = screen_x
            self.nor_car.y = screen_y
            
            # Add to trail
            self.nor_trail.append((screen_x, screen_y))
            if len(self.nor_trail) > self.max_trail_length:
                self.nor_trail.pop(0)
        
        # Update PIA car
        if self.current_frame < len(self.pia_positions):
            pia_x, pia_y = self.pia_positions[self.current_frame]
            screen_x, screen_y = self.world_to_screen(pia_x, pia_y)
            self.pia_car.x = screen_x
            self.pia_car.y = screen_y
            
            # Add to trail
            self.pia_trail.append((screen_x, screen_y))
            if len(self.pia_trail) > self.max_trail_length:
                self.pia_trail.pop(0)
        
        self.current_frame += 1
    
    def draw_cars(self):
        """Draw the car sprites."""
        self.nor_car.draw(self.screen)
        self.pia_car.draw(self.screen)
    
    def draw_info(self):
        """Draw animation info."""
        font = pygame.font.Font(None, 36)
        
        # Frame counter
        frame_text = font.render(f"Frame: {self.current_frame}/{self.max_frames}", True, (0, 0, 0))
        self.screen.blit(frame_text, (10, 10))
        
        # Driver labels
        nor_text = font.render("NOR", True, self.NOR_COLOR)
        self.screen.blit(nor_text, (10, 50))
        
        pia_text = font.render("PIA", True, self.PIA_COLOR)
        self.screen.blit(pia_text, (10, 90))
    
    def run_animation(self):
        """Run the animation loop."""
        clock = pygame.time.Clock()
        running = True
        
        print("Starting animation...")
        print("Controls:")
        print("- SPACE: Pause/Resume")
        print("- R: Reset animation")
        print("- ESC: Exit")
        
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self.current_frame = 0
                        self.nor_trail = []
                        self.pia_trail = []
            
            if not paused:
                self.update_cars()
            
            # Draw everything
            self.screen.fill(self.BACKGROUND_COLOR)
            self.draw_track()
            self.draw_trails()
            self.draw_cars()
            self.draw_info()
            
            if paused:
                # Draw pause indicator
                font = pygame.font.Font(None, 72)
                pause_text = font.render("PAUSED", True, (255, 0, 0))
                text_rect = pause_text.get_rect(center=(self.width//2, self.height//2))
                self.screen.blit(pause_text, text_rect)
            
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Animate F1 cars on track')
    parser.add_argument('--year', type=int, default=2024, help='F1 season year')
    parser.add_argument('--race', type=str, default='Hungary', help='Race name')
    parser.add_argument('--driver1', type=str, default='NOR', help='First driver code')
    parser.add_argument('--driver2', type=str, default='PIA', help='Second driver code')
    parser.add_argument('--width', type=int, default=1200, help='Window width')
    parser.add_argument('--height', type=int, default=800, help='Window height')
    
    args = parser.parse_args()
    
    # Create renderer
    renderer = AnimatedTrackRenderer(args.width, args.height)
    
    # Load data
    if not renderer.load_session_data(args.year, args.race):
        print("Failed to load session data")
        sys.exit(1)
    
    # Run animation
    renderer.run_animation()

if __name__ == "__main__":
    main()
