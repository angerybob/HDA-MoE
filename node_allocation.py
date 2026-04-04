import numpy as np
import gurobipy as gp
from gurobipy import GRB
import ast
import math
from tqdm import tqdm
import random
from heapq import heappush, heappop
from collections import defaultdict
from skopt import gp_minimize
from skopt.space import Real
from scipy.optimize import linear_sum_assignment

class MoE3DPNMOptimizer:
    def __init__(self, routing_trace, E=64, e=6, h=2048,IS=1408, B=128, D=64, BW=25e9, comp=10e12, num_layers=26, mlp_first=True):
        # Initialize parameters
        self.E = E         # Number of experts
        self.e = e
        self.h = h          # Hidden dimension
        self.IS=IS
        self.B = B          # Batch size
        self.D = D          # 3D PNM device count
        self.BW = BW        # Bandwidth (bytes/s)
        self.comp = comp    # Compute (FLOP/s)
        self.layer = num_layers # Number of MoE layers
        self.mlp_first=mlp_first
        self.routing_trace=routing_trace
        self.R_cc = (self.BW * self.IS * self.e) / (2 * self.D * self.comp)
        # Per-expert activation frequency
        self.f = np.zeros((self.layer, self.E))
        for layer_id in range(self.layer):
            for sub_list in routing_trace[str([layer_id,layer_id+1][self.mlp_first])]:
                for num in sub_list:
                    self.f[layer_id][num]+=1
        self.f=self.f/len(routing_trace[str(1)])
        self.fg = self._generate_co_activation(routing_trace)
        self.route_cache = {}
        self.X=8
        self.Y=8
        self.M=np.zeros((self.D,self.X,self.Y))

        
    def _find_optimal_aggregator(self):
        """For simplicity, use the geometric-center device as the aggregation point."""
        # A full implementation would use expert distribution; here we use geometric center.
        x_center = np.mean([d[0] for d in self.M])
        y_center = np.mean([d[1] for d in self.M])
        min_dist = float('inf')
        best_d = 0
        for d, (x, y) in enumerate(self.M):
            dist = abs(x - x_center) + abs(y - y_center)
            if dist < min_dist:
                min_dist = dist
                best_d = d
        return best_d
    
    def _generate_co_activation(self,routing_trace):
        """Build co-activation frequency tables per layer."""
        fg = {}
        fg_pruning={}
        k=self.e
        print("Begin to sampling activation data...")
        '''for k in tqdm(range(2, self.e+1)):
            #print(k)
            if k==2 or k==self.e:'''
        fg[k] = {}
        fg_pruning[k] = {}
        #print((1.5*(k//2)+0.5*math.ceil(k/2))//2)
        #threshold = len(list(combinations(range(self.E), int((1.5*(k//2)+0.5*math.ceil(k/2))//2)))) / len(list(combinations(range(self.E), k)))
        #threshold = (1.5*len(list(combinations(range(self.E), int(k//2))))+0.5*len(list(combinations(range(self.E), math.ceil(k/2)))))/2 / len(list(combinations(range(self.E), k)))
        #threshold = (len(list(combinations(range(self.E), k//2))) / len(list(combinations(range(self.E), k))))*len(routing_trace["1"])
        for layer_id in tqdm(range(self.layer)):
            #print(layer_id)
            fg[k][layer_id]={}
            fg_pruning[k][layer_id]={}
            for sub_list in routing_trace[str([layer_id,layer_id+1][self.mlp_first])]:
                temp_list=list(sorted(list(sub_list)))

                               
                list_key=str(temp_list)
                if list_key in fg[k][layer_id]:
                    fg[k][layer_id][list_key] += 1
                else:
                    fg[k][layer_id][list_key] = 1
            # Sum of all values (optional normalization step)
            #total_sum = sum(fg[k][layer_id].values())
            # pruning
            #if layer_id>6:

            # Normalize
            for key in fg[k][layer_id]:
                fg[k][layer_id][key] /= len(routing_trace[str([layer_id,layer_id+1][self.mlp_first])])
                '''if k==self.e:
                    fg_pruning[k][layer_id][key] = fg[k][layer_id][key] 
                elif fg[k][layer_id][key]>threshold:
                    fg_pruning[k][layer_id][key] = fg[k][layer_id][key]'''        
        return fg

    # ----------------- Performance model -----------------
    def compute_time(self, P):
        """Compute time t_comp."""
        compute_load = np.sum(P * self.f[:,:, None] * self.B * 2 * self.h*self.IS, axis=1)
        return np.max(compute_load / self.comp, axis=1)
    
    def compute_time_dynamic(self, P,comp_map):
        compute_load = np.sum(P * comp_map[None,:,None], axis=1)
        return np.max(compute_load / self.comp, axis=1)

    def comm_time(self, P):
        """Approximate communication time t_comm."""

        single_comm = np.zeros((self.layer,self.D))

        for layer_id, layer_fg in self.fg[self.e].items():
            for list_key, freq in layer_fg.items():
                group = ast.literal_eval(list_key)
                devices = np.sum(P[layer_id][group]>0,axis=0)
                redundant = freq * self.B * self.h * (devices>0)
                single_comm[layer_id] += redundant

        return 4*np.max(single_comm, axis=1) / (self.BW)
    
    def comm_time_dynamic(self, P,random_samples):

        single_comm = np.zeros((self.layer,self.D))
    
        for l in range(self.layer):
            for d in range(self.D):
                expert_gropus=list(np.nonzero(P[l, :, d])[0])
                for sublist in random_samples:
                    if bool(set(sublist).intersection(set(expert_gropus))):
                        single_comm[l,d]+=self.h
        
        return 4*np.max(single_comm, axis=1) / (self.BW)
    
    def _get_xy_path(self, src, dst):
        """XY-routing path with memoization."""
        src=tuple((int(src[0]),int(src[1])))
        dst=tuple((int(dst[0]),int(dst[1])))
        cache_key = (tuple(src), tuple(dst))
        if cache_key not in self.route_cache:
            path = []
            current = src
            while current[0] != dst[0]:
                next_node = (current[0] + (1 if dst[0] > current[0] else -1), current[1])
                path.append((current, next_node))
                current = next_node
            while current[1] != dst[1]:
                next_node = (current[0], current[1] + (1 if dst[1] > current[1] else -1))
                path.append((current, next_node))
                current = next_node
            self.route_cache[cache_key] = path
        return self.route_cache[cache_key]
    
    def _simulate_comm(self,M, layer_data,layer_id,P,chunks=20):
        """Discrete-event simulation for communication time."""
        # Init structures
        link_schedule = defaultdict(list)  # {link: [(start_time, end_time)]}
        event_queue = []
        for sublist in self.fg[self.e][layer_id].keys():
            sublist=ast.literal_eval(sublist)
            non_zero_coords = []
            for i in sublist:
                # Coordinates of non-zero entries in the 2D placement for this expert

                d_id,rows, cols = np.nonzero(M[np.nonzero(P[layer_id][i])])

                
                for x, y in zip(rows, cols):
                    if (x, y) not in non_zero_coords:
                        non_zero_coords.append((x, y))

            d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][random.choice(sublist)])])
            #idex=np.random.randint(0,len(x_s))
            idex=0
            x_center,y_center=x_s[idex],y_s[idex]
            aggregator=tuple((x_center,y_center))
            data_size = self.fg[self.e][layer_id][str(sublist)]*self.B*self.h
            

            for p in non_zero_coords:
                path = self._get_xy_path(p, aggregator)
                for _ in range(chunks):
                    heappush(event_queue, (0.0, data_size/chunks, path))

        # Drain event queue
        max_finish_time = 0
        count=0
        while event_queue:
            current_time, remaining_data, remaining_path = heappop(event_queue)
            
            if not remaining_path:
                
                max_finish_time = max(max_finish_time, current_time)
                #print(max_finish_time)
                continue
                
            current_link = remaining_path[0]
            available_bw = self.BW
            
            # Next free time window on this link
            last_end = 0
            for start, end in sorted(link_schedule[current_link]):
                if last_end <= current_time < start:
                    available_window = start - current_time
                    break
                last_end = end
                current_time=max(current_time,end)
            else:
                available_window = float('inf')
                
            # Bytes transferred this step
            trans_time = remaining_data / available_bw
            actual_trans = min(trans_time, available_window)
            
            # Book link usage
            new_start = current_time
            new_end = current_time + actual_trans
            link_schedule[current_link].append((new_start, new_end))
            
            # Remaining work / next hop
            if actual_trans < trans_time:
                new_remaining = remaining_data - actual_trans * available_bw
                heappush(event_queue, (new_end, new_remaining, remaining_path))
            else:
                heappush(event_queue, (new_end, remaining_data, remaining_path[1:]))
        link_load = defaultdict(float)
        for link, time_windows in link_schedule.items():
            for start, end in time_windows:
                link_load[link] += end - start  
        return max_finish_time,link_load
    
    def comm_time_acc(self, M,P,layer_id,chunks=1):
        """Discrete-event simulation for accumulated communication time."""
        
        # Per-device send volume
        layer_data = np.zeros((self.D))


        
        #print("processing the expert freq...")
        for list_key, freq in self.fg[self.e][layer_id].items():
            group = ast.literal_eval(list_key)
            devices = np.sum(P[layer_id][group]>0,axis=0)
            #redundant = freq * self.B * self.h * np.maximum((devices-1),0)
            redundant = freq * self.B * self.h * (devices>0)
            #single_comm[layer_id] -= redundant
            layer_data += redundant
        #print("finish processing!")
        # Run DES
        comm_time,link = self._simulate_comm(M,layer_data,layer_id,P,chunks)
        #layer_comm_times.append(comm_time)
            
        return comm_time,link 
    
    
    def _simulate_comm_dynamic(self,M, layer_data,layer_id,P,random_samples,chunks=20):
        """Discrete-event simulation (dynamic sampling variant)."""
        link_schedule = defaultdict(list)
        event_queue = []


        for sublist in random_samples:

            non_zero_coords = []
            for i in sublist:
                d_id,rows, cols = np.nonzero(M[np.nonzero(P[layer_id][i])])
                
                for x, y in zip(rows, cols):
                    if (x, y) not in non_zero_coords:
                        non_zero_coords.append((x, y))

            d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][random.choice(sublist)])])
            #idex=np.random.randint(0,len(x_s))
            while len(x_s)==0:
                d_id,x_s,  y_s=np.nonzero(M[np.nonzero(P[layer_id][random.choice(sublist)])])
            idex=0
            x_center,y_center=x_s[idex],y_s[idex]
            aggregator=tuple((x_center,y_center))
            data_size = self.h
            

            for p in non_zero_coords:
                path = self._get_xy_path(p, aggregator)
                for _ in range(chunks):
                    heappush(event_queue, (0.0, data_size/chunks, path))

 
        max_finish_time = 0
        while event_queue:
            #print(len(event_queue))
            current_time, remaining_data, remaining_path = heappop(event_queue)
            
            if not remaining_path:
                #print(f"{len(event_queue)} remaining...")
                max_finish_time = max(max_finish_time, current_time)
                continue
                
            current_link = remaining_path[0]
            available_bw = self.BW
            
            last_end = 0
            for start, end in sorted(link_schedule[current_link]):
                if last_end <= current_time < start:
                    available_window = start - current_time
                    break
                last_end = end
                current_time=max(current_time,end)
            else:
                available_window = float('inf')
                
            trans_time = remaining_data / available_bw
            actual_trans = min(trans_time, available_window)
            
            new_start = current_time
            new_end = current_time + actual_trans
            link_schedule[current_link].append((new_start, new_end))
            
            if actual_trans < trans_time:
                new_remaining = remaining_data - actual_trans * available_bw
                heappush(event_queue, (new_end, new_remaining, remaining_path))
            else:
                heappush(event_queue, (new_end, remaining_data, remaining_path[1:]))
                
        return max_finish_time
    
    def comm_time_acc_dynamic(self, M,P,layer_id,random_samples,chunks=1):
        """Discrete-event simulation with dynamic random samples."""
        
        layer_data = np.zeros((self.D))


        

        for list_key, freq in self.fg[self.e][layer_id].items():
            group = ast.literal_eval(list_key)
            devices = np.sum(P[layer_id][group]>0,axis=0)
            redundant = freq * self.B * self.h * (devices>0)
            layer_data += redundant
            
        comm_time = self._simulate_comm_dynamic(M,layer_data,layer_id,P,random_samples,chunks)

            
        return comm_time



    
    def EP_deployment(self,L, E, D):
        """
        Build expert placement tensor P with shape L x E x D.
        P[l,e,d] is the fraction of expert e on device d (0 <= value <= 1).
        """
        P = np.zeros((L,E, D))
        
        if D >= E:
            # Each expert spans multiple devices with uniform split
            k, r = divmod(D, E)
            devices = np.arange(D)
            np.random.shuffle(devices)
            start = 0
            for e in range(E):
                num_devices = k + 1 if e < r else k
                end = start + num_devices
                assigned_devices = devices[start:end]
                P[:,e, assigned_devices] = 1.0 / num_devices
                start = end
        else:
            # Fewer devices than experts: pack experts evenly, one device per expert slot
            m, r = divmod(E, D)
            experts = np.arange(E)
            np.random.shuffle(experts)
            expert_idx = 0
            for d in range(D):
                num_experts = m + 1 if d < r else m
                assigned_experts = experts[expert_idx : expert_idx + num_experts]
                P[:,assigned_experts, d] = 1.0
                expert_idx += num_experts
        return P
    
    def ilp_solver_gurobi(self, l, gamma=4,time_limit=60):
        """Gurobi MILP for expert placement on this layer."""
        try:
            model = gp.Model("MoE_Expert_Placement")
            
            P = model.addVars(self.E, self.D, lb=0, ub=1, name="P")
            Z = model.addVars(self.E, self.D, vtype=GRB.BINARY, name="Z")  # Z[i,d] == 1 iff P[i,d] > 0

            P_init=self.EP_deployment(self.layer,self.E,self.D)
            Z_init=P_init>0

            for i in range(self.E):
                for d in range(self.D):
                    P[i,d].Start=P_init[l,i,d]
                    Z[i,d].Start=Z_init[l,i,d]

            t_comp = model.addVar(name="t_comp")
            t_comm = model.addVar(name="t_comm")

            
            model.setObjective(t_comp + 2*t_comm, GRB.MINIMIZE)
            
            print("Begin to add constraint for expert placement...")
            
              
            for i in range(self.E):
                model.addConstr(gp.quicksum(P[i,c] for c in range(self.D)) == 1, 
                            f"expert_{i}_placement_in_layer_{l}")
            comp_per_expert=2 * self.h*self.IS * self.B
            max_comp = (1 / self.R_cc + 1) * (self.e) / self.D
            print("Begin to add constraint for computation node-balance...")

            for c in range(self.D):
                comp_load = gp.quicksum(P[i,c] * self.f[l,i] for i in range(self.E))
                model.addConstr(comp_load <= max_comp, f"comp_load_layer_{l}_node_{c}")
                model.addConstr(comp_load >= 0, f"min_comp_load_layer_{l}_node_{c}")
                model.addConstr(t_comp >= comp_load*comp_per_expert/self.comp, f"comp_time_layer_{l}_node_{c}")
            
            print("Begin to add constraint for communication node-balance...")
            comm_per_token=self.B * self.h
            

            for c in range(self.D):
                for i in range(self.E):
                    model.addConstr(P[i,c] <= Z[i,c], f"P_leq_Z_{l}_{i}_{c}")
                single_comm = 0

                for list_key, freq in self.fg[self.e][l].items():
                    Y = model.addVar(vtype=GRB.BINARY, name="Y_"+list_key+f"_placed_on_{c}_in_layer_{l}") 
                    group = ast.literal_eval(list_key)
                    devices = gp.quicksum(Z[g,c] for g in group)
                    model.addConstr(Y >= devices/len(group), "expert_groups_"+list_key+f"_placed_on_{c}_in_layer_{l}")
                    redundant = freq * Y
                    single_comm += redundant
                comm_time = 4*gamma*single_comm*comm_per_token / (self.BW)
                model.addConstr(t_comm >= comm_time, f"comm_time_node_{c}_in_layer_{l}")
            
            model.Params.TimeLimit = time_limit
            model.Params.MIPGap = 0.05
            model.Params.Threads = 8
            model.Params.Heuristics = 0.1
            model.optimize()

            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = np.zeros((self.layer,self.E, self.D))

                for i in range(self.E):
                    for c in range(self.D):
                        solution[l,i,c] = P[i,c].X
                return solution
            else:
                print(f"No solution found. Status: {model.status}")
                return None
                
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return None

        
        
    def ilp_solver_gurobi_comp(self, l, moe_model="ds",gamma=4,time_limit=60):
        """Gurobi ILP: compute-only objective (no comm constraints)."""
        try:
            model = gp.Model("MoE_Expert_Placement_comp")
            
            if moe_model=="mixtral":
                D=2
            else:
                D=8
            Z = model.addVars(self.E, D, vtype=GRB.BINARY, name="Z")

            t_comp = model.addVar(name="t_comp")

            
            model.setObjective(t_comp, GRB.MINIMIZE)
            
            for i in range(self.E):
                model.addConstr(gp.quicksum(Z[i,c] for c in range(D)) == 1, 
                            f"expert_{i}_placement_in_layer_{l}")
            comp_per_expert=2 * self.h*self.IS * self.B



            for c in range(D):
                comp_load = gp.quicksum(Z[i,c] * self.f[l,i] for i in range(self.E))

                model.addConstr(comp_load >= 0, f"min_comp_load_layer_{l}_node_{c}")
                model.addConstr(t_comp >= comp_load*comp_per_expert/self.comp, f"comp_time_layer_{l}_node_{c}")
            
 
            model.Params.TimeLimit = time_limit
            model.Params.MIPGap = 0.05
            model.Params.Threads = 8
            model.Params.Heuristics = 0.1
            model.optimize()
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                solution = np.zeros((self.layer,self.E, D))

                for i in range(self.E):
                    for c in range(D):
                        solution[l,i,c] = Z[i,c].X
                return solution
            else:
                print(f"No solution found. Status: {model.status}")
                return None
                
        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
            return None
        
  
    
    def optimize_placement_sa(self, initial_placement, P, layer_id, 
                            max_iter=1000, initial_temp=1000, cooling_rate=0.99):
        """
        Simulated annealing on device placement to minimize the MST-based comm metric.
        :param initial_placement: Initial placement (D x X x Y)
        :param P: Fixed expert-to-device assignment (E x D) for this layer context
        :param layer_id: Layer index
        :param max_iter: Max iterations
        :param initial_temp: Initial temperature
        :param cooling_rate: Cooling factor per step
        :return: (best_placement, cost_history)
        """
        current_placement = initial_placement.copy()
        current_cost = self.evaluate_placement(current_placement, P, layer_id)
        best_placement = current_placement.copy()
        best_cost = current_cost
        
        temp = initial_temp
        cost_history = [best_cost]
        
        for i in tqdm(range(max_iter)):
            new_placement = self._perturb_placement(current_placement)
            
            new_cost = self.evaluate_placement(new_placement, P, layer_id)
            cost_diff = new_cost - current_cost
            
            if cost_diff < 0 or math.exp(-cost_diff / temp) > random.random():
                current_placement = new_placement.copy()
                current_cost = new_cost
                
                if new_cost < best_cost:
                    best_placement = new_placement.copy()
                    best_cost = new_cost
            cost_history.append(float(best_cost)) 
            temp *= cooling_rate
        
        return best_placement, cost_history

    def _perturb_placement(self, placement):
        """Swap two devices' grid positions at random."""
        new_placement = placement.copy()
        D = new_placement.shape[0]
        
        d1, d2 = np.random.choice(D, 2, replace=False)
        
        pos1 = np.argwhere(new_placement[d1] == 1)[0]
        pos2 = np.argwhere(new_placement[d2] == 1)[0]
        
        new_placement[d1, pos1[0], pos1[1]] = 0
        new_placement[d2, pos2[0], pos2[1]] = 0
        new_placement[d1, pos2[0], pos2[1]] = 1
        new_placement[d2, pos1[0], pos1[1]] = 1
        
        return new_placement


    def evaluate_placement(self, placement, P,layer_id):
        """
        Weighted MST distance over co-activated expert groups.
        :param placement: D x X x Y one-hot device grid
        :param P: E x D expert assignment for the layer
        :return: Average weighted MST length
        """
        X, Y = placement.shape[1], placement.shape[2]
        device_coords = {d: np.argwhere(placement[d] == 1)[0] for d in range(self.D)}
        expert_to_device = [np.where(P[layer_id][e] == 1)[0].tolist() for e in range(self.E)]
        
        total_weight = 0.0
        total_freq = 0.0
        

  
        for group_str, freq in self.fg[self.e][layer_id].items():
            devices = [expert_to_device[e] for e in ast.literal_eval(group_str)]
            devices = list(set(d for sublist in devices for d in sublist))
            coords = [tuple(device_coords[d]) for d in devices]
            
            if len(coords) < 2:
                continue
                
            mst_dist = self._calculate_mst(coords)
            total_weight += freq * mst_dist
            total_freq += freq
                    
        return total_weight / total_freq if total_freq > 0 else 0

    def _calculate_mst(self, coords):
        """Kruskal's algorithm for Manhattan MST total length."""
        edges = []
        n = len(coords)
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(coords[i][0] - coords[j][0])
                dy = abs(coords[i][1] - coords[j][1])
                edges.append((i, j, dx + dy))
        
        edges.sort(key=lambda x: x[2])
        parent = list(range(n))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            parent[find(u)] = find(v)
        
        mst_sum = 0
        for u, v, w in edges:
            if find(u) != find(v):
                union(u, v)
                mst_sum += w
        return mst_sum
    
    
    def optimize_placement_bo(self, initial_placement, P, layer_id, 
                             max_iter=50, random_state=None):
        """
        Bayesian optimization over continuous device coordinates; Hungarian maps to a valid grid.
        :param initial_placement: D x X x Y
        :param P: Expert-device matrix for evaluation
        :param layer_id: Layer index
        :param max_iter: gp_minimize n_calls
        :param random_state: RNG seed
        :return: (best_placement, func_vals)
        """
        D, X, Y = initial_placement.shape
        assert D == X * Y, "Device count must equal mesh size X*Y"
        initial_params = []
        for d in range(D):
            pos = np.argwhere(initial_placement[d] == 1)[0]
            initial_params.extend(pos.tolist())
            
        space = [Real(0, X-1), Real(0, Y-1)] * D
        
        def objective(params):
            device_coords = np.array(params).reshape(D, 2)
            grid_points = np.array([(x, y) for x in range(X) for y in range(Y)])
            cost_matrix = np.zeros((D, X*Y))
            
            for d in range(D):
                for g, (x, y) in enumerate(grid_points):
                    dx = abs(device_coords[d, 0] - x)
                    dy = abs(device_coords[d, 1] - y)
                    cost_matrix[d, g] = dx + dy
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            placement = np.zeros((D, X, Y), dtype=int)
            for d in range(D):
                x, y = grid_points[col_ind[d]]
                placement[d, x, y] = 1
            result,link=self.comm_time_acc(placement, P, layer_id)
            return result
        
        result = gp_minimize(
            objective,
            space,
            n_calls=max_iter,
            x0=initial_params,
            random_state=random_state,
            n_initial_points=min(10, max_iter),
            verbose=True
        )
        
        best_params = result.x
        grid_points = np.array([(x, y) for x in range(X) for y in range(Y)])
        device_coords = np.array(best_params).reshape(D, 2)
        cost_matrix = np.zeros((D, X*Y))
        for d in range(D):
            for g, (x, y) in enumerate(grid_points):
                dx = abs(device_coords[d, 0] - x)
                dy = abs(device_coords[d, 1] - y)
                cost_matrix[d, g] = dx + dy
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        best_placement = np.zeros((D, X, Y), dtype=int)
        for d in range(D):
            x, y = grid_points[col_ind[d]]
            best_placement[d, x, y] = 1
        
        return best_placement, result.func_vals
    # ----------------- Dynamic placement helpers -----------------
    def priority_detection(self, P,layer_id,random_samples):
        """Pick highest-priority expert to move off the most loaded node."""
        priorities = []
        if layer_id > 0:
            comp_map=np.zeros((self.E))
            for sublist in random_samples:
                comp_map[sublist]+=2*self.h*self.IS
            node_load = np.sum(P * comp_map[None,:,None], axis=1)
            congested_node = np.argmax(node_load,axis=1)[layer_id]
            for i in range(self.E):
                prio = (P[layer_id,i, congested_node] * comp_map[i] / self.comp)
                priorities.append((prio, i))
        if not priorities:
            return (0.0, 0)
        return sorted(priorities, reverse=True)[0]


    def optimal_broadcast_chunk(self, alpha=1e-7, k=1):
        """Alpha-beta style optimal broadcast chunk size (latency + bandwidth terms)."""
        beta=1/self.BW
        c =  np.sqrt(2*self.h*self.IS*alpha / (2*beta * k * np.sqrt(self.D)))
        latency = alpha * (2 * np.sqrt(self.D) + 2*self.h*self.IS / c)
        bandwidth = beta * k * (2*self.h*self.IS + 2 * c * np.sqrt(self.D))
        return latency + bandwidth




