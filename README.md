# AGV Multi-Agent Path Planning and Task Scheduling System

## Project Overview

This project implements a comprehensive multi-agent path planning and task scheduling system for Automated Guided Vehicles (AGVs) in a warehouse environment. The system coordinates 12 AGVs to efficiently complete 102 tasks across a 21×21 grid warehouse, ensuring collision-free operations and optimal task allocation.

## Features

- **Multi-Agent Path Planning**: A* algorithm with temporal constraints for collision-free pathfinding
- **Dynamic Task Allocation**: Cost-matrix-based greedy assignment with priority handling
- **Conflict Resolution**: Prevents collisions and path swaps through forward-looking prediction
- **Real-time Visualization**: Interactive pygame-based visualization with video export capability
- **Task Sequence Management**: Ensures tasks are picked up in correct order from each pickup point

## Demo Video

以下是系统运行时的演示视频：

<video width="800" controls>
  <source src="agv_simulation.mp4" type="video/mp4">
  您的浏览器不支持视频标签。
</video>

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

