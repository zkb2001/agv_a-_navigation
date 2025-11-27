
import csv
import heapq
import sys
import io
from collections import defaultdict
import os

def get_agv_state(agv_list):
    agv_states = {
        agv["id"]: {
            # "pos": tuple(agv["pose"]),
            # "pitch": agv["pitch"],
            # "time": 0,
            "state": agv["pose"] + (0,agv["pitch"]),
            "task_id": None,
            "path": [],
            "load_point": None,
            "end_point": None,
            "priority": False
        } for agv in agv_list
    }
    # print(agv_states)
    return agv_states

def get_end_points(end_point_name,end_point):
    possible_pos = [
            (1, 0),    # 右
            (-1, 0), # 左
            (0, 1),   # 上
            (0, -1)  # 下
        ]
    # print(end_point)
    # print(end_point_name)
    unload_point = end_point[end_point_name]
    possible_end_points = []
    for x,y in possible_pos:
        possible_end_points.append((x+unload_point[0],y+unload_point[1]))
    # print(possible_end_point)
    return unload_point,possible_end_points

def get_pickup_coord(start_point_name, original_coord):
    if start_point_name in ["Tiger", "Dragon", "Horse"]:
        return (original_coord[0] + 1, original_coord[1])
    else:
        return (original_coord[0] - 1, original_coord[1])
    
def get_task_list(agv_task_path):
    all_tasks = {}
    with open(agv_task_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["start_point"] not in all_tasks:
                all_tasks[row["start_point"]] = []
            all_tasks[row["start_point"]].append({
                "task_id": row["task_id"],
                "start_point": row["start_point"].strip(),
                "end_point": row["end_point"].strip(),
                "priority": row["priority"],
                "remaining_time": int(row["remaining_time"]) if row["remaining_time"] not in [None, "", "None"] else None
            })
    return all_tasks

def get_task_state(start_points,all_tasks,all_end_points):
    task_states = {}
    for task_name,task_dict in all_tasks.items():
        if task_name not in task_states:
            task_states[task_name] = []
        # print(task_name,task_dict)
        for i in range(len(task_dict)):
            pickup_pos = get_pickup_coord(task_name,start_points[task_name])
            unload_point,end_points = get_end_points(task_dict[i]["end_point"],all_end_points)
            if task_dict[i]["priority"] == "Urgent":
                # task_states[task_name]["remaining_time"] = task_dict[i]["remaining_time"]
                task_states[task_name].append({
                "task_id": task_dict[i]["task_id"],
                "pickup_point": pickup_pos,
                "unload_point": unload_point,
                "end_points": end_points,
                "destination": task_dict[i]["end_point"],
                "priority": task_dict[i]["priority"],
                "numbers_before_urgent": 0,
                # "remaining_time": 300,
                "numbers_left": len(all_tasks[task_name])-i-1
            })
                for j in range(i):
                    task_states[task_name][j]['numbers_before_urgent'] = i-j
            else:
                task_states[task_name].append({
                "task_id": task_dict[i]["task_id"],
                "pickup_point": pickup_pos,
                "unload_point": unload_point,
                "end_points": end_points,
                "destination": task_dict[i]["end_point"],
                "priority": task_dict[i]["priority"],
                "numbers_before_urgent": -1,
                # "remaining_time": 300,
                "numbers_left": len(all_tasks[task_name])-i-1,
            })
    
    return task_states

def get_object_position(agv_position_path):
    start_points, end_points, agv_list = {}, {}, []
    with open(agv_position_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t, name = row["type"].strip(), row["name"].strip()
            x, y = int(row["x"]), int(row["y"])
            if t == "start_point":
                start_points[name] = (x, y)
            elif t == "end_point":
                end_points[name] = (x, y)
            elif t == "agv":
                agv_list.append({
                    "id": name,
                    "pose": (x, y),
                    "pitch": int(row["pitch"])
                })
    return start_points, end_points, agv_list


def append_to_csv(steps):

    with open(AGV_TRAJECTORY_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "name", "X", "Y", "pitch", "loaded", "destination", "Emergency", "task-id"])
        for step in steps.values():
            for s in step:
                writer.writerow([
                    s["timestamp"],
                    s["name"],
                    s["X"],
                    s["Y"],
                    s["pitch"],
                    str(s["loaded"]).lower(),
                    s["destination"],
                    str(s["Emergency"]).lower(),
                    s.get("task-id", "")

                ])
              
              
"""检查轨迹文件中是否存在碰撞或对穿情况"""
def check_trajectory_conflicts(trajectory_file):
    # 读取轨迹文件
    agv_positions = {}  # 格式: {timestamp: {agv_name: (x, y)}}
    agv_task_info = {}  # 格式: {timestamp: {agv_name: task_id}}
    
    with open(trajectory_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['timestamp'] == 'timestamp':  # 跳过表头行
                continue
            timestamp = int(row['timestamp'])
            agv_name = row['name']
            x = int(row['X'])
            y = int(row['Y'])
            task_id = row.get('task-id', '')
            
            if timestamp not in agv_positions:
                agv_positions[timestamp] = {}
                agv_task_info[timestamp] = {}
            agv_positions[timestamp][agv_name] = (x, y)
            agv_task_info[timestamp][agv_name] = task_id
    
    has_conflict = False
    
    # 检查每个时间点的碰撞
    for t in sorted(agv_positions.keys()):
        # 检查同一时间点的位置碰撞
        positions = agv_positions[t]
        checked_agvs = set()
        
        for agv1, pos1 in positions.items():
            for agv2, pos2 in positions.items():
                if agv1 != agv2 and (agv1, agv2) not in checked_agvs and (agv2, agv1) not in checked_agvs:
                    checked_agvs.add((agv1, agv2))
                    
                    # 检查位置碰撞
                    if pos1 == pos2:
                        print(f"时间 {t}: {agv1} 和 {agv2} 在位置 {pos1} 发生碰撞")
                        has_conflict = True
                    
                    # 如果有下一个时间点，检查对穿
                    if t + 1 in agv_positions and agv1 in agv_positions[t+1] and agv2 in agv_positions[t+1]:
                        next_pos1 = agv_positions[t+1][agv1]
                        next_pos2 = agv_positions[t+1][agv2]
                    
                        
                        # 检查交叉路径
                        if pos1 == next_pos2 and pos2 == next_pos1:
                            print(f"时间 {t}-{t+1}: {agv1} 和 {agv2} 发生路径交叉")
                            print(f"  {agv1}: {pos1} -> {next_pos1}")
                            print(f"  {agv2}: {pos2} -> {next_pos2}")
                            has_conflict = True
    
    # 检查任务顺序是否正确
    print("\n=== 检查任务顺序 ===")
    check_task_order(agv_task_info)
    
    if not has_conflict:
        print("无对穿或碰撞")
    
    return has_conflict

def check_task_order(agv_task_info):
    """检查AGV是否按正确顺序领取任务"""
    # 按时间戳排序
    timestamps = sorted(agv_task_info.keys())
    
    # 记录每个AGV的任务历史
    agv_task_history = {}  # {agv_name: [task_ids]}
    
    for timestamp in timestamps:
        for agv_name, task_id in agv_task_info[timestamp].items():
            if agv_name not in agv_task_history:
                agv_task_history[agv_name] = []
            
            # 如果AGV在这个时间点有任务ID，且与上一个不同，说明领取了新任务
            if task_id and (not agv_task_history[agv_name] or agv_task_history[agv_name][-1] != task_id):
                agv_task_history[agv_name].append(task_id)
                print(f"时间 {timestamp}: AGV {agv_name} 领取任务 {task_id}")
    
    # 检查每个AGV的任务顺序
    for agv_name, task_list in agv_task_history.items():
        if len(task_list) > 1:
            print(f"AGV {agv_name} 的任务序列: {task_list}")
            
            # 检查是否有重复任务
            if len(task_list) != len(set(task_list)):
                print(f"警告: AGV {agv_name} 有重复任务!")
            
            # 检查任务ID的格式和顺序
            for i, task_id in enumerate(task_list):
                if '-' in task_id:
                    parts = task_id.split('-')
                    if len(parts) == 2:
                        pickup_point, task_num = parts[0], parts[1]
                        try:
                            task_num = int(task_num)
                            if i > 0:
                                prev_task_id = task_list[i-1]
                                if '-' in prev_task_id:
                                    prev_parts = prev_task_id.split('-')
                                    if len(prev_parts) == 2 and prev_parts[0] == pickup_point:
                                        prev_task_num = int(prev_parts[1])
                                        if task_num < prev_task_num:
                                            print(f"错误: AGV {agv_name} 任务顺序错误! {prev_task_id} -> {task_id}")
                        except ValueError:
                            pass
  
  
class Simulation:
    def __init__(self, agv_states, task_states, env, dimension = (21,21)):
        self.agv_states = agv_states
        self.task_states = task_states
        self.dimension = dimension
        self.surface_tasks = {}
        #记录时间、任务
        self.task_queue = {}
        self.tried_tasks = set()  # 改为 set 来记录所有失败的组合
        self.allocate_tasks = {}    #存储分配的任务，方便回溯修正任务错配
        self.time = 0
        self.agvs = []
        self.env = env
        self.grid_size = (21,21)
        
        #初始化12个AGV类
        self.init_agvs()
        #初始化供货点当前任务
        self.init_surface_tasks()
        #初始化任务队列
        self.init_task_queue()

        # self.a_star = A_star()
    
    def init_agvs(self):
        for agv_name,agv_state in self.agv_states.items():
            self.agvs.append(AGV(agv_name,agv_state['state'],None))
        
    def init_surface_tasks(self):
        for name, task_info in self.task_states.items():
            if task_info: 
                first_task = task_info[0] 
                task_id = first_task['task_id'] 
                task_details = {k: v for k, v in first_task.items() if k != 'task_id'}
                self.surface_tasks[task_id] = task_details
                self.surface_tasks[task_id]['pickup_name'] = name
        print(f"初始化 surface_tasks: {list(self.surface_tasks.keys())}")
    
    def init_task_queue(self):
        for name, task_info in self.task_states.items():
            if name not in self.task_queue.keys():
                self.task_queue[name] = []
            for task in task_info:
                self.task_queue[name].append(task['task_id'])
        print(self.task_queue)
        
    #判断是否出现任务错配
    def check_pickup_task(self):
        for agv in self.agvs:
            # 检查索引是否有效
            if len(agv.steps) <= self.time or len(agv.steps) <= self.time - 1:
                continue
                
            #如果当前时刻有agv从供货台拿到任务，则需要检查
            if not agv.steps[self.time-1]['task-id']\
            and agv.steps[self.time]['task-id']:
                pickup_name = agv.steps[self.time]['pickup_name']
                sim_task_id = agv.steps[self.time]['task-id']
                
                first_elements = [value[0] for value in self.task_queue.values()]
                #更新实际表面任务
                if sim_task_id in first_elements:
                    self.task_queue[pickup_name].pop(0)
                    if not self.task_queue[pickup_name]:
                        del self.task_queue[pickup_name]
                
                else:
                    print(f"检测到任务错配: AGV {agv.name} 在时间 {self.time} 获取任务 {sim_task_id}")
                    print(f"但任务队列中第一个任务应该是: {self.task_queue[pickup_name][0]}")
                    earlier_task = self.task_queue[pickup_name][0]
                    later_task = sim_task_id
                    #earlier_task被更早分配出去，但agv更晚到达，需要回溯并交换任务，然后重规划
                    print(f"开始重新规划: 交换任务 {earlier_task} 和 {later_task}")
                    self.replan(earlier_task, later_task)

                    # 重新检查，避免无限递归
                    if agv.task_id in first_elements:
                        self.task_queue[pickup_name].pop(0)
                        return


            
        
        # self.real_surface_task[]
        
        
                
                
        
    def get_cost_matrix_and_allocate_task(self,unassigned_agvs):
        cost_matrix = {}
        assigned_task = []
        min_agv = None
        min_task_id = None
        min_cost = float('inf')
        # print(f"当前 surface_tasks: {list(self.surface_tasks.keys())}")
        # print(f"当前 tried_tasks: {self.tried_tasks}")
        # print(self.surface_tasks)
        for agv in unassigned_agvs:
            cost_matrix[agv.name] = {}
            for task_id,task_info in self.surface_tasks.items():
                # 计算AGV到任务的距离
                cost = self.manhattan_distance(agv.state[:2], task_info['pickup_point']) 
                # 考虑紧急程度权重
                urgent_weight = 1.0
                if task_info['numbers_before_urgent'] >= 0:
                    urgent_weight = 0.7
                cost *= urgent_weight
                
                #如果是之前尝试过的，将其cost标注为inf
                if (agv.name, task_id) in self.tried_tasks:
                    cost = float('inf')
                    print(f"AGV {agv.name} 任务 {task_id} 已尝试过，设为 inf")
                    
                cost_matrix[agv.name][task_id] = cost
                
                if cost < min_cost:
                    min_agv = agv
                    min_task_id = task_id
                    min_cost = cost
                    
        if min_agv and min_task_id:
            # 记录分配结果
            assigned_task = {
                "agv": min_agv.name,
                "task_id": min_task_id,
                "agv_start_point": min_agv.state,
                "pickup_point": self.surface_tasks[min_task_id]['pickup_point'],
                "end_points": self.surface_tasks[min_task_id]['end_points'],
                "destination": self.surface_tasks[min_task_id]['destination'],
                "priority": self.surface_tasks[min_task_id]['priority'],
                "pickup_name": self.surface_tasks[min_task_id]['pickup_name'],
                "time": self.time
            }
            print(f"分配: AGV {min_agv.name} → 任务 {min_task_id}, 代价: {min_cost}")
        else:
            print("没有可用的任务分配")
            
        return assigned_task

    def get_unassigned_agvs(self):
        unassigned_agvs = []
        for agv in self.agvs:
            # print(agv.name)
            if agv.task_id == None:
                unassigned_agvs.append(agv)
        return unassigned_agvs
    
    def path_to_step(self,path,task,loaded):
        steps = []
        if not loaded:
            for i in range(len(path)):
                steps.append({
                    "timestamp": path[i][2],
                    "name": task['agv'],
                    "X": path[i][0],
                    "Y": path[i][1],
                    "pitch": path[i][3],
                    "loaded": "FALSE",
                    "destination": "",
                    "Emergency": "FALSE",
                    "task-id": "",
                    "pickup_name": ""
                })
        else:
            #这里要减1，因为要修改最后一个位置的step
            for i in range(len(path)-1):
                steps.append({
                    "timestamp": path[i][2],
                    "name": task['agv'],
                    "X": path[i][0],
                    "Y": path[i][1],
                    "pitch": path[i][3],
                    "loaded": "TRUE",
                    "destination": task['destination'],
                    "Emergency": "FALSE" if task['priority'] == "Normal" else "TRUE",
                    "task-id": task['task_id'],
                    "pickup_name": task['pickup_name']
                })
            steps.append({
                "timestamp": path[i][2]+1,
                "name": task['agv'],
                "X": path[i][0],
                "Y": path[i][1],
                "pitch": path[i][3],
                "loaded": "FALSE",
                "destination": '',
                "Emergency": 'FALSE',
                "task-id": '',
                "pickup_name": ""
            })
        return steps
    
    def time_forward(self):
        self.time += 1
        #还存在没有领到任务的小车
        while True:
             #代价最小的agv先规划路径
            unassigned_agvs = self.get_unassigned_agvs()
            if not unassigned_agvs:
                break
            if not self.surface_tasks:
                break
            found_valid_path = False
            #当前任务没有找到路径，需重新计算cost_matrix计算min_cost
            while not found_valid_path:
                assigned_task = self.get_cost_matrix_and_allocate_task(unassigned_agvs)
                if not assigned_task:  # 没有可用任务了
                    break  # 退出循环，处理下一个AGV
                # 为当前任务搜索路径
                path, steps = self.a_star(assigned_task)
                print(f"AGV {assigned_task['agv']} 任务找到路径{path}")
                if env.is_valid_path(assigned_task['end_points'],path, self.time):
                    found_valid_path = True
                # update task
                    # update tried_tasks
                    self.update_tried_tasks(assigned_task['agv'])
                    # update surface task
                    self.update_surface_tasks(assigned_task['pickup_name'],assigned_task['task_id'])
                    # update agv path and steps
                    self.update_agvs(assigned_task,path,steps)
                    # update env
                    self.update_env(assigned_task['agv'],path)
                    # 保存分配的任务信息
                    self.update_allocate_tasks(assigned_task)

                else:
                    print(f"AGV {assigned_task['agv']} 无法为{assigned_task['task_id']}任务找到有效路径")
                    self.tried_tasks.add((assigned_task['agv'], assigned_task['task_id']))
                    break
        
        # update agv states
        self.update_agv_state()
        # 检查是否存在任务错位
        self.check_pickup_task()
        
        #assinged_task【start: tuple, pickup_point: tuple, end_points: tuple】 
    #env【obstacles: list, moving_obstacles: dict = {}, grid_size: tuple = (21, 21)】
    def a_star(self, assigned_task):
        def has_temp(pos, timestamp, temp_obstacle_positions):
            has_temp = False
            for agv_name, temp in temp_obstacle_positions.items():
                if isinstance(temp, tuple) and len(temp) >= 2:
                    temp_pos, temp_time = temp
                    if temp_pos == pos and timestamp + 1 >= temp_time:
                        has_temp = True
                        break  # 找到后提前退出循环
            return has_temp
        
        def get_valid_neighbors(state,temp_obstacles):
            possible_moves = [
                (1, 0, 0),    # 右
                (-1, 0, 180), # 左
                (0, 1, 90),   # 上
                (0, -1, 270)  # 下
            ]
            x, y, timestamp, direction = state
            
            '准备障碍物'
            static_obstacles = []
            forward_one_second_obstacles = []
            forward_two_seconds_obstacles = []
            
            static_obstacles = self.env.get_static_obstacles()
            forward_one_second_obstacles = self.env.get_forward_one_second_obstacles(timestamp)
            forward_two_seconds_obstacles = self.env.get_forward_two_seconds_obstacles(timestamp)
            
            # 检查原地等待: 1. 不在障碍物中 2. 不在移动障碍物中 
            if (x,y) not in static_obstacles and (x,y) not in forward_one_second_obstacles:
                if not has_temp((x,y),timestamp,temp_obstacles):
                    yield (x, y, timestamp+1, direction), 1, None
                
            # 检查移动和转向: 1. 不在障碍物中 2. 不在移动障碍物中 3. 不对穿
            for dx, dy, new_direction in possible_moves:
                new_x, new_y = x + dx, y + dy
                if not (1 <= new_x <= self.grid_size[0] and 1 <= new_y <= self.grid_size[1]):
                    continue
                
                # 同方向移动
                if direction == new_direction:
                    if ((new_x, new_y) not in static_obstacles and 
                        (new_x, new_y) not in forward_one_second_obstacles and 
                        not self.env.judge_swapping((x,y,timestamp,direction),(new_x,new_y,timestamp+1,new_direction))):
                        if not has_temp((new_x,new_y),timestamp,temp_obstacles):
                            yield (new_x, new_y, timestamp+1, new_direction), 1, None
                # 需要转向
                else:
                    if ((x, y) not in static_obstacles and (x, y) not in forward_one_second_obstacles) and \
                        ((new_x, new_y) not in static_obstacles and (new_x, new_y) not in forward_two_seconds_obstacles) and \
                        not self.env.judge_swapping((x,y,timestamp+1,direction),(new_x,new_y,timestamp+2,new_direction)):
                            if not has_temp((x,y),timestamp,temp_obstacles) and not has_temp((new_x,new_y),timestamp+1,temp_obstacles):
                                turn_state = (x, y, timestamp+1, new_direction)
                                move_state = (new_x, new_y, timestamp+2, new_direction)
                                yield move_state, 2, turn_state
        
        
        
        pickup_point = assigned_task['pickup_point']
        temp_obstacles = self.env.get_temp_obstacles(assigned_task)
        path = []
        steps = []
        frontier = []
        visited = set()
        loaded = 0
        heapq.heappush(frontier,(self.manhattan_distance(assigned_task['agv_start_point'][:2], pickup_point), 0,\
                        assigned_task['agv_start_point'], [assigned_task['agv_start_point']]))
        
        '从起点到供货台'
        while frontier:
            _,cost,current,path = heapq.heappop(frontier)
            #若超出长度未找到，则直接截断
            if len(path) > 40:
                print('qqqqqqqqqqqqqqqqqqqq')
                return [],[]
            
            if current[:2] == pickup_point:
                pickup_movement = (*pickup_point[:2], current[2]+1, current[3])
                steps.extend(self.path_to_step(path, assigned_task, loaded))
                loaded = 1
                break
            
            # 检查访问状态
            current_state = current[:3]
            if current_state in visited:
                continue
            visited.add(current_state)
                
            # 扩展邻居节点
            for neighbor, move_cost, turn_state in get_valid_neighbors(current,temp_obstacles):
                neighbor_state = neighbor[:3]
                if neighbor_state in visited:
                    continue
                new_cost = cost + move_cost
                predicted_cost = new_cost + manhattan_distance(neighbor[:2], pickup_point)
                # 如果需要转向，把转向状态加入路径
                new_path = path + ([turn_state] if turn_state else []) + [neighbor]
                heapq.heappush(frontier, (predicted_cost, new_cost, neighbor, new_path))
            
        if (path[-1][0], path[-1][1]) != pickup_point:
            return [], []
        
        '从供货台到卸货点'
        end_points = assigned_task['end_points']
        possible_paths = []
        #如果有小车停在终点，则移除该终点
        for temp in temp_obstacles.values():
            if pickup_movement[2]+1 > temp[1] and temp[0] in end_points:
                end_points.remove(temp[0])
        
        for end_point in end_points:
            frontier = []
            possible_path = []
            visited = set()
            start_state = pickup_movement
            heapq.heappush(frontier, (manhattan_distance(start_state[:2], end_point), 0, start_state, [start_state]))
            while frontier:
                _, cost, current, possible_path = heapq.heappop(frontier)
                #超出时间限制，直接结束路线
                if len(possible_path) > 45:
                    possible_path = []
                    break
                
                if current[:2] == end_point:
                    possible_paths.append(possible_path)
                    break
                
                current_state = current[:3]
                if current_state in visited:
                    continue
                visited.add(current_state)
                
                for neighbor, move_cost, turn_state in get_valid_neighbors(current,temp_obstacles):
                    neighbor_state = neighbor[:3]
                    if neighbor_state in visited:
                        continue
                        
                    new_cost = cost + move_cost
                    predicted_cost = new_cost + manhattan_distance(neighbor[:2], end_point)
                    new_path = possible_path + ([turn_state] if turn_state else []) + [neighbor]
                    heapq.heappush(frontier, (predicted_cost, new_cost, neighbor, new_path))
        
        min_cost = float('inf')
        min_path = []
        for p in possible_paths:
            if len(p) < min_cost:
                min_cost = len(p)
                min_path = p
        
        if min_path == []:
            return [],[]
        
        else:
            min_path = min_path + [(min_path[-1][0],min_path[-1][1],min_path[-1][2]+1,min_path[-1][3])]
            steps.extend(self.path_to_step(min_path,assigned_task,loaded))
        path.extend(min_path)
        
        #因为会将上一段路径终点作为起始点考虑，因此需删去，否则会在路径中重复
        if path[0][2] != 0:
            return path[1:],steps[1:]

        return path, steps    
            

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    
    def update_surface_tasks(self, name, task_id):
        del self.surface_tasks[task_id]
        #在task_states中找到task_id后面一个任务
        if task_id == self.task_states[name][-1]:
            return
        for i in range(len(self.task_states[name])-1):
            if task_id == self.task_states[name][i]['task_id']:
                first_task = self.task_states[name][i+1]
                task_id = first_task['task_id'] 
                task_details = {k: v for k, v in first_task.items() if k != 'task_id'}
                self.surface_tasks[task_id] = task_details
                self.surface_tasks[task_id]['pickup_name'] = name
                print(f"更新新任务 {task_id} 到 {name}")
                break
    
    def update_agvs(self, assigned_task, path, steps):
        for agv in self.agvs:
            if agv.name == assigned_task['agv']:
                agv.task_id = assigned_task['task_id']
                agv.priority = assigned_task['priority']
                agv.path += path
                agv.steps += steps
                break
                    
    def update_env(self, agv_name, path):
        if agv_name not in self.env.moving_obstacles.keys():
            self.env.moving_obstacles[agv_name] = []
        self.env.moving_obstacles[agv_name] += path
            
    def update_tried_tasks(self, agv_name):
        # 当任务成功完成时，从 tried_tasks 中移除该 AGV 的所有失败记录
        # 因为该 AGV 现在可以接受新任务了
        self.tried_tasks = {(agv, task) for agv, task in self.tried_tasks if agv != agv_name}
            
    def reset_agv_for_new_task(self, agv):
        """重置AGV状态，使其能够接受新任务"""
        if agv.task_id is not None and self.time >= len(agv.path) - 1:
            print(f"AGV {agv.name} 完成任务 {agv.task_id}，重置状态接受新任务")
            agv.task_id = None
            agv.priority = False
            
    def update_agv_state(self):
        for agv in self.agvs:
            # 确保时间戳不超过路径长度
            if self.time < len(agv.path):
                agv.state = agv.path[self.time]
            else:
                # 如果时间超出路径范围，使用最后一个状态
                if agv.path:  # 确保路径不为空
                    agv_state_temp = (*agv.path[-1][:2], self.time, agv.path[-1][3])
                    agv.state = agv_state_temp
                    agv.path.append(agv_state_temp)
                    # print(f"AGV {agv.name} 超出路径范围，使用最后状态: {agv_state_temp}")
                    agv.steps.append({
                        "timestamp": agv_state_temp[2],
                        "name": agv.name,
                        "X": agv_state_temp[0],
                        "Y": agv_state_temp[1],
                        "pitch": agv_state_temp[3],
                        "loaded": "FALSE",
                        "destination": "",
                        "Emergency": "FALSE",
                        "task-id": "",
                        "pickup_name": ""
                    })
            
            # 检查AGV是否完成当前任务，如果是则重置状态
            self.reset_agv_for_new_task(agv)
                
    def update_allocate_tasks(self, assigned_task):
        self.allocate_tasks[assigned_task['task_id']] = assigned_task
          
    def all_over(self):
        # 检查是否还有未分配的任务
        if self.surface_tasks:
            print(f"还有未分配的任务: {list(self.surface_tasks.keys())}")
            return False
            
        # 检查是否还有AGV正在执行任务
        for agv in self.agvs:
            if agv.task_id is not None:
                print(f"AGV {agv.name} 还在执行任务: {agv.task_id}")
                return False
                
        # 检查是否还有任务队列
        if self.task_queue:
            print(f"还有任务队列: {self.task_queue}")
            return False
            
        return True
    
    def steps_reorganize(self):
        steps_dict = {}
        for agv in self.agvs:
            for step in agv.steps:
                if step['timestamp'] not in steps_dict:
                    steps_dict[step["timestamp"]] = []
                steps_dict[step['timestamp']].append(step)
        return steps_dict

    def replan(self, earlier_task, later_task):
        earlier_task_info = self.allocate_tasks[earlier_task]
        later_task_info = self.allocate_tasks[later_task]
        earlier_time = earlier_task_info['time']
        later_time = later_task_info['time']
        
        print(f"重新规划: AGV {earlier_task_info['agv']} 和 {later_task_info['agv']}")
        print(f"时间点: {earlier_time} 和 {later_time}")
        
        # 清除两个AGV的现有路径和步骤
        for agv in self.agvs:
            if agv.name == earlier_task_info['agv']:
                agv.path = agv.path[:earlier_time]
                agv.steps = agv.steps[:earlier_time]
                agv.task_id = None
                agv.priority = False
                # 确保AGV状态正确
                if agv.path:
                    agv.state = agv.path[-1]
                print(f"AGV {agv.name} 路径长度: {len(agv.path)}")
            if agv.name == later_task_info['agv']:
                agv.path = agv.path[:later_time]
                agv.steps = agv.steps[:later_time]
                agv.task_id = None
                agv.priority = False
                # 确保AGV状态正确
                if agv.path:
                    agv.state = agv.path[-1]
                print(f"AGV {agv.name} 路径长度: {len(agv.path)}")
        
        # 清除环境中的移动障碍物
        for name, path in self.env.moving_obstacles.items():
            if name == earlier_task_info['agv']:
                self.env.moving_obstacles[name] = self.env.moving_obstacles[name][:earlier_time]
            if name == later_task_info['agv']:
                self.env.moving_obstacles[name] = self.env.moving_obstacles[name][:later_time]
          
        # 交换任务信息
        earlier_task, later_task = self.switch_task(earlier_task_info, later_task_info)
        path1, steps1 = self.a_star(earlier_task)
        if path1 and self.env.is_valid_path(earlier_task['end_points'], path1, earlier_time):
            self.env.moving_obstacles[earlier_task['agv']] += path1
            self.update_agvs(earlier_task, path1, steps1)
            self.update_allocate_tasks(earlier_task)
            print(f"AGV {earlier_task['agv']} 重新规划成功，任务: {earlier_task['task_id']}")
        else:
            print(f"AGV {earlier_task['agv']} 重新规划失败")
            
        path2, steps2 = self.a_star(later_task)
        if path2 and self.env.is_valid_path(later_task['end_points'], path2, later_time):
            self.env.moving_obstacles[later_task['agv']] += path2
            self.update_agvs(later_task, path2, steps2)
            self.update_allocate_tasks(later_task)
            print(f"AGV {later_task['agv']} 重新规划成功，任务: {later_task['task_id']}")
        else:
            print(f"AGV {later_task['agv']} 重新规划失败")
        
        # 不改变其他已规划好的路径，只调换当前错误任务

    def switch_task(self, earlier_task, later_task):
        keys = ["task_id","end_points","priority","destination"]
        print(f"交换前 - earlier_task: {earlier_task['task_id']} -> {earlier_task['agv']}")
        print(f"交换前 - later_task: {later_task['task_id']} -> {later_task['agv']}")
        
        for key in keys:
            value1 = earlier_task[key]
            value2 = later_task[key]
            earlier_task[key], later_task[key] = value2, value1
            
        print(f"交换后 - earlier_task: {earlier_task['task_id']} -> {earlier_task['agv']}")
        print(f"交换后 - later_task: {later_task['task_id']} -> {later_task['agv']}")
        
        return earlier_task, later_task
    
        
    def undo_task_assignment(self,agv_name, task_id):
        #agv
        pass
        #env(moving_obstacles)
        
        
                 
#  assigned_tasks.append({
#                 "agv": agv_id_min,
#                 "task": task_id_min,
#                 "agv_start_point": agv_states[agv_id_min]['pos'],
#                 "pickup_point": task_states[task_name_min][0]['pickup_point'],
#                 "end_points": task_states[task_name_min][0]['end_points'],
#                 "destination": task_states[task_name_min][0]['destination'],
#                 "priority": task_states[task_name_min][0]['priority'],
#             })   

class ENV:
    def __init__(self, start_points, destination_points):
        self.start_points = start_points
        self.destination_points = destination_points
        self.moving_obstacles = {}
    
    def get_end_points(self, destination_name):
        possible_pos = [
        (1, 0),    # 右
        (-1, 0), # 左
        (0, 1),   # 上
        (0, -1)  # 下
        ]
        # print(end_point)
        # print(end_point_name)
        destination = self.destination_points[destination_name]
        possible_end_points = []
        for x,y in possible_pos:
            possible_end_points.append((x+destination[0],y+destination[1]))
        # print(possible_end_point)
        return possible_end_points
    
        
    '''
    用于Astar检查
    '''
    def get_static_obstacles(self):
        return self.start_points + self.destination_points
        
    def get_forward_one_second_obstacles(self, time):
        one_second_obstacles = []
        for name, path in self.moving_obstacles.items():
            if time+1 < len(path):
                one_second_obstacles.append(path[time+1][:2])
        # one_second_obstacles = list(set(one_second_obstacles))
        return one_second_obstacles
        
    def get_forward_two_seconds_obstacles(self, time):
        forward_two_seconds_obstacles = []
        for path in self.moving_obstacles.values():
            if time+2 < len(path):
                forward_two_seconds_obstacles.append(path[time+2][:2])
        # forward_two_seconds_obstacles = list(set(forward_two_seconds_obstacles))
        return forward_two_seconds_obstacles
        
    def get_temp_obstacles(self, assigned_task):
        temp_obstacle_positions = {}
        for name, pos in self.moving_obstacles.items():
            if name not in temp_obstacle_positions and assigned_task['agv'] != name:
                temp_obstacle_positions[name] = []
            if assigned_task['agv'] != name:
            # 只有当路径不为空时才添加临时障碍物
                if pos and len(pos) > 0:
                    temp_obstacle_positions[name] = (pos[-1][:2], pos[-1][2])
        return temp_obstacle_positions
        
    
    #判断是否对穿,如果对穿，则返回True，否则返回False
    def judge_swapping(self,current_pos_time_dir,new_pos_time_dir):
        current_x,current_y,current_t,current_d = current_pos_time_dir
        new_x,new_y,new_t,new_d = new_pos_time_dir
        for _,trajectory in self.moving_obstacles.items():
            # for i in range(len(trajectory)-1):
            #     if trajectory[i][2] != i:
            #         print(f"{_}的时间戳有误")
            #         print(current_pos_time_dir)
                #检查对穿
            if new_t < len(trajectory):
                if trajectory[current_t][:2] == (new_x,new_y) and \
                    trajectory[current_t+1][:2] == (current_x,current_y):
                    # print(trajectory[current_t][:2],(new_x,new_y))
                    return True
        return False
            
    
    '''
    用于simulation验证
    '''
    def is_valid_path(self, end_points, new_path, time):
        #路径终点需到达卸货点
        if not new_path:
            return False
        
        if new_path[-1][:2] not in end_points:
            print(new_path)
            print(end_points)
            print('xxxxxx')
            return False
        
        for name,trajectory in self.moving_obstacles.items():
            clip_trajectory = trajectory[time:]
            for i in range(min(len(clip_trajectory),len(new_path))-1):
                #没有碰撞
                if clip_trajectory[i][:2] == new_path[i][:2]:
                    # print(name)
                    # print(trajectory)
                    # print(new_path)
                    print(clip_trajectory[i][:2],i)
                    print('yyyyyy')
                    return False
                #没有对穿
                if new_path[i][:2] == clip_trajectory[i-1][:2] and \
                new_path[i-1][:2] == clip_trajectory[i][:2]:
                    print('zzzzz')
                    return False
        return True
                    
        
        
    
    
class AGV:
    def __init__(self, name, state, task_id, priority=False, path=None, steps=None):
        self.name = name
        self.state = state
        self.task_id = task_id
        self.priority = priority
        self.path = path if path is not None else []
        self.steps = steps if steps is not None else []
    
    def __repr__(self):
        return 'AGV:' + str(self.name) + '= '+ ', state:' + str(self.state) + ', task_id:' +\
        str(self.task_id) + ', priority:' + str(self.priority) + ', path:' + str(self.path) + '\n'
        
        
        
def manhattan_distance(pos1, pos2):
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    
    
    
if __name__ == '__main__':
    
    AGV_POSITION_PATH = 'agv_position.csv'
    AGV_TASK_PATH = 'agv_task.csv'
    AGV_TRAJECTORY_PATH = 'agv_trajectory.csv'
    #获取所有任务
    all_tasks = get_task_list(AGV_TASK_PATH)
    #获取取货点、卸货点和小车初始状态
    start_points, end_points, agv_list = get_object_position(AGV_POSITION_PATH)

    #静态障碍
    env =  ENV(list(start_points.values()),list(end_points.values()))
    
    #补充小车信息
    agv_states = get_agv_state(agv_list)
    #整理任务状态
    task_states = get_task_state(start_points,all_tasks,end_points)
    unassigned_agvs = list(agv_states.keys())
    # print(agv_states)
    # print(task_states)

    sim = Simulation(agv_states,task_states,env,(21,21))
    while not sim.all_over():
        sim.time_forward()
        
    all_steps = sim.steps_reorganize()
    print(all_steps)
    
    append_to_csv(all_steps)

    check_trajectory_conflicts(AGV_TRAJECTORY_PATH)