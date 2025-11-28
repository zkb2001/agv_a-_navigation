# AGV Multi-Agent Path Planning and Task Scheduling System

## Project Overview

This project implements a comprehensive multi-agent path planning and task scheduling system for Automated Guided Vehicles (AGVs) in a warehouse environment. The system coordinates 12 AGVs to efficiently complete 102 tasks across a 21×21 grid warehouse, ensuring collision-free operations and optimal task allocation.

## Competition Rules

This project was developed for the **2025 Siemens Xcelerator Open Competition - MioVerse Track JCIIOT Developer Contest**. The following rules outline the problem requirements and constraints.

### Problem Description

A courier company needs to optimize task allocation and AGV (Automated Guided Vehicle) route planning for an automated sorting unit to achieve efficient transportation of packages from pickup stations to destination sorting outlets. AGVs must complete transportation tasks in the shortest time while avoiding collisions and satisfying all constraint conditions.

#### Given Information

1. **Pickup Stations**: 6 stations where packages arrive. AGVs pick up packages at designated pickup points. Station names: `["Tiger", "Dragon", "Horse", "Rabbit", "Ox", "Monkey"]`. Each station has one pickup point.

2. **Sorting Outlets**: 16 outlets corresponding to destination flows. Each outlet has 4 adjacent unloading points.

3. **AGV Fleet**: 12 AGVs, all available. Names: `["Optimus", "Bumblebee", ..., "Jazz"]`. AGVs transport packages from pickup station pickup points to outlet unloading points. After reaching the designated unloading point of the destination outlet, AGVs automatically unload packages into the outlet.

4. **AGV Properties**:
   - **Dimensions**: 40cm × 40cm × 30cm
   - **Speed**: 1 m/s (acceleration/deceleration not considered)
   - **Turning**: Only in-place rotation, turning angle must be multiples of 90 degrees, each turn takes 1 second
   - **Loading/Unloading**: Each operation takes 1 second
   - **Waiting**: AGVs can wait in place for collision avoidance, waiting time must be integer multiples of 1 second
   - **Initial State**: Each AGV has initial coordinates [x, y] and orientation pitch (in degrees: 0° = +X axis, 90° = +Y axis, 180° = -X axis, 270° = -Y axis)

5. **Map Layout**:
   - The area map consists of a 20m × 20m region, each grid cell is 1m × 1m
   - AGVs can only move along grid cell centers

6. **Package Sequence**: Random sequence matching outlet flows, including pickup station number, destination outlet number, priority type (high-priority urgent or normal), and remaining valid time for high-priority packages.

### Constraints

1. Each AGV can only transport one package at a time
2. AGVs can only pick up packages when available at the station, and each pickup station can only complete one AGV pickup per second
3. AGVs pick up packages at the unique pickup point on the left/right side of the pickup station (e.g., Horse pickup point at (2,14); Monkey pickup point at (19,14)). AGVs can unload at any of the 4 unloading points adjacent to the destination outlet (unloading points are in the 4 adjacent grid cells, e.g., if destination Hangzhou is at (6,16), unloading points are at (6,15), (6,17), (5,16), (7,16)). AGVs cannot move beyond the 20m × 20m map boundaries
4. Packages must be picked up in the order specified by each pickup station's package sequence
5. High-priority packages must be transported to the destination within the specified remaining time (constraint satisfaction: +10 points bonus; violation: -5 points penalty)
6. Packages must be unloaded at the designated unloading points of the destination outlet
7. AGVs can only move along X and Y axes, one grid per second, no diagonal movement
8. All blank areas except pickup stations and outlets are passable. Collisions occur when AGVs appear at the same position at the same time, or when AGVs swap positions while moving in opposite directions
9. AGV in-place turning can only be ±90° or 180°, any angle turn takes 1 second to complete. Turning cannot occur at the same time as loading/unloading operations

### Scoring Rules

#### Preliminary Round: Automated Evaluation (Full Score: 120 points, including 20 bonus points)

- **Basic Scoring**: Within 5 minutes, count all packages successfully delivered to correct destinations. Each package = 1 point. Tasks violating constraints receive no points
- **High-Priority Task Bonus/Penalty**: High-priority packages unloaded at destination outlet within specified remaining time: +10 points per task. If high-priority task not completed within time limit: -5 points per task
- **AGV Collision Penalty**: If AGVs collide: -10 points. Collided AGVs disappear and cannot be used further. Package tasks are lost and cannot continue
- **Tie-Breaking**: In case of ties, compare the completion time of the last package task. Earlier completion wins. If still tied, additional random task inputs are used, considering evaluation score, task completion time, and program runtime

#### Final Round: Expert Subjective Evaluation (Full Score: 100 points)

Expert evaluation considers the following factors:

| Evaluation Criteria | Key Points | Weight |
| :--: | :--: | :--: |
| AI/LLM Technology Application | - Depth and innovation of LLM integration in the system<br>- Reasonableness of AI technology application scenarios<br>- Effectiveness of AI solutions<br>- Explainability of intelligent decisions | 40% |
| Algorithm Innovation | - Breakthrough improvements compared to traditional methods<br>- Originality of algorithm ideas<br>- Innovation level of technical approach | 30% |
| Practicality and Scalability | - Adaptation to real scenarios<br>- System scalability difficulty<br>- Deployment and maintenance costs<br>- Computational overhead | 20% |
| Code Quality | - Code standardization<br>- Maintainability | 10% |

**Final Score = Automated Evaluation Score (60%) + Expert Evaluation Score (40%)**

## Features

- **Multi-Agent Path Planning**: A* algorithm with temporal constraints for collision-free pathfinding
- **Dynamic Task Allocation**: Cost-matrix-based greedy assignment with priority handling
- **Conflict Resolution**: Prevents collisions and path swaps through forward-looking prediction
- **Real-time Visualization**: Interactive pygame-based visualization with video export capability
- **Task Sequence Management**: Ensures tasks are picked up in correct order from each pickup point

## Demo Video

以下是系统运行时的演示视频：

[![AGV Simulation Demo](https://img.youtube.com/vi/l5caN4atO44/maxresdefault.jpg)](https://youtube.com/shorts/l5caN4atO44?feature=share)

**点击上方图片观看完整演示视频** | [直接访问 YouTube](https://youtube.com/shorts/l5caN4atO44?feature=share)

## Project Structure

```
final_version/
├── agv_position.csv          # Warehouse layout and initial AGV positions
├── agv_task.csv              # Task definitions (102 tasks)
├── agv_trajectory.csv         # Generated AGV trajectories (output)
├── naviagation.py             # Main simulation algorithm
├── display.py                 # Visualization module
├── project_report.ipynb       # Jupyter notebook with full analysis
├── README.md                  # This file
└── agv_simulation.mp4         # Generated simulation video (output)
```

## Requirements

### Python Packages

- **csv**: Built-in module for CSV file handling
- **heapq**: Built-in module for priority queue (A* algorithm)
- **collections.defaultdict**: Built-in module for efficient dictionary operations
- **pandas**: Data manipulation and analysis (`pip install pandas`)
- **numpy**: Numerical computing (`pip install numpy`)
- **matplotlib**: Data visualization (`pip install matplotlib`)
- **seaborn**: Statistical visualization (`pip install seaborn`)
- **pygame**: Real-time visualization (`pip install pygame`)
- **imageio**: Video export (`pip install imageio`)

### Installation

```bash
pip install pandas numpy matplotlib seaborn pygame imageio
```

## Usage

### 1. Run the Simulation

Execute the main simulation to generate AGV trajectories:

```bash
python "navigation.py"
```

This will:
- Load warehouse layout from `agv_position.csv`
- Load tasks from `agv_task.csv`
- Run the simulation with path planning and task allocation
- Generate `agv_trajectory.csv` with complete AGV movements
- Perform conflict checking to verify no collisions or path swaps

### 2. Visualize the Simulation

Run the visualization tool to see the AGV movements:

```bash
python display.py
```

Options:
- **Speed control**: `python display.py 2` (2x speed)
- **Disable recording**: `python display.py --no-record`
- **Custom video filename**: `python display.py --record output.mp4`

The visualization shows:
- AGV positions and orientations (arrows)
- Cargo states (colored circles)
- Pickup points (blue squares)
- Delivery points (green triangles)
- Remaining task counts
- Current timestamp

### 3. Analyze Results

Open `project_report.ipynb` in Jupyter Notebook to:
- Explore data characteristics
- Analyze task distributions
- Visualize AGV trajectories
- Review performance metrics

## Algorithm Details

### A* Pathfinding

- **Heuristic**: Manhattan distance
- **Cost Function**: Movement (1 step) + Turn (1 additional step)
- **Constraints**: 
  - Static obstacles (pickup/delivery points)
  - Dynamic obstacles (other AGVs)
  - Swap prevention (no path crossing)

### Task Allocation

- **Method**: Greedy cost-matrix approach
- **Cost Calculation**: Manhattan distance from AGV to pickup point
- **Priority Weighting**: Urgent tasks receive 0.7× cost multiplier
- **Assignment**: Minimum cost AGV-task pair

### Conflict Resolution

- **Collision Detection**: Same position at same timestamp
- **Swap Detection**: Path crossing (position swap)
- **Task Order Validation**: Ensures correct task sequence

## Data Format

### agv_position.csv

```csv
type,name,x,y,pitch
start_point,Tiger,1,6,
end_point,Beijing,6,4,
agv,Optimus,3,1,90
```

- `type`: `start_point`, `end_point`, or `agv`
- `x`, `y`: Grid coordinates (1-21)
- `pitch`: Orientation in degrees (0, 90, 180, 270)

### agv_task.csv

```csv
task_id,start_point,end_point,priority,remaining_time
Tiger-1,Tiger,Xiamen,Normal,None
```

- `task_id`: Unique task identifier
- `start_point`: Pickup location name
- `end_point`: Delivery location name
- `priority`: `Normal` or `Urgent`
- `remaining_time`: Optional deadline (not used in current implementation)

### agv_trajectory.csv

```csv
timestamp,name,X,Y,pitch,loaded,destination,Emergency,task-id
0,Optimus,3,1,90,false,,false,
1,Optimus,3,2,90,false,,false,
```

- `timestamp`: Time step (0, 1, 2, ...)
- `name`: AGV identifier
- `X`, `Y`: Position coordinates
- `pitch`: Orientation
- `loaded`: `true` if carrying cargo
- `destination`: Delivery point name (when loaded)
- `Emergency`: `true` for urgent tasks
- `task-id`: Current task identifier

## Key Programming Packages

### csv
**Purpose**: Reading and writing CSV data files  
**Usage**: Loading warehouse layout, task definitions, and writing trajectory output

### heapq
**Purpose**: Priority queue implementation for A* algorithm  
**Usage**: Maintains the frontier in A* search, ensuring we always expand the most promising node first

### collections.defaultdict
**Purpose**: Dictionary with default values  
**Usage**: Efficiently groups tasks by start_point and manages task queues

### pandas
**Purpose**: Data manipulation and analysis  
**Usage**: Loading, cleaning, and analyzing CSV data in the Jupyter notebook

### matplotlib & seaborn
**Purpose**: Data visualization  
**Usage**: Creating plots to visualize warehouse layout, task distributions, and simulation results

### pygame
**Purpose**: Real-time visualization and animation  
**Usage**: Interactive visualization of AGV movements with frame-by-frame rendering and video export

### numpy
**Purpose**: Numerical computing  
**Usage**: Array operations for coordinate calculations and video frame processing

## Performance

- **Warehouse Size**: 21×21 grid
- **AGV Fleet**: 12 vehicles
- **Tasks**: 102 tasks
- **Pickup Points**: 6 locations
- **Delivery Points**: 16 destinations
- **Typical Simulation Time**: Varies based on task complexity (usually 200-400 time steps)

## Limitations

1. **Scalability**: May not scale well to very large fleets (50+ AGVs)
2. **Greedy Allocation**: Uses greedy approach, not globally optimal
3. **Static Environment**: Assumes static warehouse layout
4. **No Replanning**: Paths are fixed once assigned
5. **Limited Priority Handling**: Basic priority weighting, no strict deadlines

## Future Improvements

- Advanced MAPF algorithms (Conflict-Based Search, Push and Swap)
- Optimization-based task allocation (linear programming, constraint programming)
- Dynamic replanning capabilities
- Real-time adaptation to dynamic obstacles
- Comprehensive performance metrics
- Machine learning integration for adaptive planning


## License

This project is for educational purposes.



For detailed analysis and visualizations, please refer to `project_report.ipynb`.

