# F1 Track Position Data Extraction Script

This script extracts track XY position data for drivers at every time step during qualifying using the Fast F1 API. It can analyze and compare two drivers' positions throughout the session.

## Features

- Extract complete track position data (X, Y coordinates) for drivers
- Support for any F1 race from 2018 onwards
- Automatic caching for improved performance
- Data analysis with statistics and sample data
- Export functionality to CSV files
- Comprehensive error handling

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python qualifying_positions.py --year 2024 --race "Monaco" --driver1 "VER" --driver2 "HAM"
```

### Command Line Arguments

- `--year`: Year of the race (e.g., 2024)
- `--race`: Name of the Grand Prix (e.g., "Monaco", "Silverstone", "Spa")
- `--driver1`: Three-letter driver code for first driver (e.g., "VER", "HAM")
- `--driver2`: Three-letter driver code for second driver (e.g., "LEC", "NOR")
- `--export`: Export position data to CSV files (optional)

### Examples

```bash
# Extract position data for Verstappen vs Hamilton at Monaco 2024
python qualifying_positions.py --year 2024 --race "Monaco" --driver1 "VER" --driver2 "HAM"

# Extract and export position data for Leclerc vs Norris at Silverstone 2023
python qualifying_positions.py --year 2023 --race "Silverstone" --driver1 "LEC" --driver2 "NOR" --export

# Extract and export position data for Norris vs Piastri at Hungary 2024
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --export
```

## Driver Codes

Common F1 driver codes (as of 2024):
- VER - Max Verstappen
- HAM - Lewis Hamilton
- LEC - Charles Leclerc
- NOR - Lando Norris
- RUS - George Russell
- PER - Sergio Perez
- SAI - Carlos Sainz
- ALO - Fernando Alonso
- OCO - Esteban Ocon
- GAS - Pierre Gasly

## Race Names

Use the official Grand Prix names:
- "Monaco"
- "Silverstone"
- "Spa"
- "Monza"
- "Suzuka"
- "Interlagos"
- "Melbourne"
- "Bahrain"
- "Jeddah"
- And many more...

## Output

The script will display:
- Loading progress
- Available drivers in the session
- Track position data analysis with statistics
- Sample position data for both drivers
- Export information (if --export flag is used)

Example output:
```
Extracting track position data for NOR vs PIA
Race: 2024 Hungary
Loading qualifying session for 2024 Hungary...
Available drivers in session:
  NOR (Lando Norris) - Position: 2
  PIA (Oscar Piastri) - Position: 5
Retrieved 1250 position data points for NOR
Retrieved 1180 position data points for PIA

============================================================
TRACK POSITION DATA ANALYSIS
============================================================

NOR Position Data:
  Total data points: 1250
  Time range: 0 days 00:00:00 to 0 days 00:45:30
  X coordinate range: -500.00 to 500.00
  Y coordinate range: -300.00 to 300.00

PIA Position Data:
  Total data points: 1180
  Time range: 0 days 00:00:00 to 0 days 00:45:30
  X coordinate range: -500.00 to 500.00
  Y coordinate range: -300.00 to 300.00

Sample data for NOR (first 5 points):
      X      Y  Status
0  100.5  200.3  OnTrack
1  102.1  201.8  OnTrack
2  103.7  203.2  OnTrack
3  105.2  204.6  OnTrack
4  106.8  206.1  OnTrack
```

## Data Export

When using the `--export` flag, the script creates a directory with:
- Individual CSV files for each driver's position data
- Combined CSV file with both drivers' data
- Metadata JSON file with session information

The exported data includes:
- Timestamp (index)
- X coordinate
- Y coordinate
- Status (OnTrack, InPit, etc.)
- Driver information

## Notes

- The script uses FastF1's caching system to improve performance on subsequent runs
- Data is available from 2018 onwards
- Internet connection is required for the first run of each race
- Driver codes are case-insensitive (automatically converted to uppercase)

## Troubleshooting

If you encounter issues:
1. Ensure you have a stable internet connection
2. Verify the year, race name, and driver codes are correct
3. Check that the race exists in the specified year
4. Make sure all dependencies are installed correctly

# Create static plot of fastest laps
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --plot

# Create animated plot of fastest laps
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --animate

# Extract data and create fastest lap plot
python qualifying_positions.py --year 2024 --race "Hungary" --driver1 "NOR" --driver2 "PIA" --export --plot