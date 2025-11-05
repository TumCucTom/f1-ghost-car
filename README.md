# F1 Qualifying Visual Overhead Comparisons

![Follow Car Hungary](media/hungary_2024_follow.gif)

A Python script that visualizes Formula 1 qualifying sessions with animated car tracking and realistic track layouts.

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Track Data
Download track CSV files from the [TUM FTM Racetrack Database](https://github.com/TUMFTM/racetrack-database/tree/master):
- Place track CSV files in the project root (e.g., `Budapest.csv`, `Spa.csv`)
- Files should have format: `x_m,y_m,w_tr_right_m,w_tr_left_m`

### 3. Get Car Images
- Create `assets/cars/` directory
- Add car images as PNG files named by driver code (e.g., `nor.png`, `pia.png`)
- Images should show the car from above, with the car taking up about half the image width
- Found by googling "F1 car top view" or similar

### 4. Create Cache Directory
```bash
mkdir cache
```

## Usage

### Basic Command
```bash
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA"
```

### All Flags

#### Required Arguments
- `--year`: Year of the race (2018-2024)
- `--race`: Race name (e.g., "Hungary", "Belgium", "Spa")
- `--driver1`: First driver code (e.g., "NOR", "PIA", "VER")
- `--driver2`: Second driver code

#### Optional Arguments
- `--animate`: Create animated GIF instead of static plot
- `--follow`: Camera follows the cars (only with --animate)
- `--follow-window`: Follow window size in meters (default: 200)
- `--gif-seconds`: Target GIF duration in seconds (auto-adjusts fps)
- `--road`: Add realistic track surface based on track data
- `--no-lines`: Make racing lines white (hide lines, keep cars)

## Examples

### Static Plot
```bash
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA"
```

### Animated Overview
```bash
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --animate
```

### Follow Mode with Track
```bash
python qualifying_positions.py --year 2024 --race "Belgium" --driver1 "NOR" --driver2 "PIA" --animate --follow --road
```

### Cars Only (No Racing Lines)
```bash
python qualifying_positions.py --year 2024 --race "Spa" --driver1 "NOR" --driver2 "PIA" --animate --follow --road --no-lines
```

## Output

- **Static plots**: Displayed and saved as PNG
- **Animations**: Saved as GIF files
- **Filenames**: `{year}_{race}_{driver1}_vs_{driver2}_fastest_laps_animated.gif`

