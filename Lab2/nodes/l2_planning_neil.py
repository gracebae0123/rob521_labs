#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
import scipy.spatial as sp
from scipy.linalg import block_diag
import os

#os.environ["SDL_VIDEODRIVER"] = "dummy"

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    im_np = im_np[:,:,0] # (49, 159)
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost, traj_arrive=None):
        self.point = point # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        self.traj_arrive = traj_arrive
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist, myhal=False):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape # (49, 159)

        self.map_settings_dict = load_map_yaml(map_setings_filename)
        self.T_wm = self.transform_map_to_world()
        self.T_mw = np.linalg.inv(self.T_wm)
        self.myhal = myhal

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) # [x_min, x_max; y_min, y_max]
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.25 #0.5m/s (Feel free to change!)
        self.rot_vel_max = 1.6 #0.2rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", self.occupancy_map.shape, self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return
    
    @staticmethod
    def transform_2d(x, y, theta):
        T = np.eye(3)
        T[0, 0] = np.cos(theta)
        T[1, 1] = np.cos(theta)
        T[0, 1] = -np.sin(theta)
        T[1, 0] = np.sin(theta)
        T[0, 2] = x
        T[1, 2] = y
        return T
    
    def transform_map_to_world(self):
        x, y, theta = self.map_settings_dict["origin"]
        return self.transform_2d(x, y, theta)
    
    #Functions required for RRT
    def sample_map_space(self, prob_goal = 0.05, myhal = False):
        #Return an [x,y] coordinate to drive the robot towards
        #print("TO DO: Sample point to drive towards")
        # With probability prob_goal to sample around goal point
        goal_d = self.closest_node_dist(self.goal_point)[1]
        if goal_d < 5:
            if np.random.rand() < 0:
                return self.goal_point + 3 * goal_d * np.random.randn(2, 1)
        if np.random.rand() < prob_goal:
            return self.goal_point + 5 * self.stopping_dist * np.random.randn(2, 1)
        if not self.myhal:
            real_bounds = np.array([[-3.5, 43.5],[-49.25, 10.5]])
            return np.random.rand(2, 1) * (real_bounds[:, [1]] - real_bounds[:, [0]])  + real_bounds[:, [0]] 
        return np.random.rand(2, 1) * (self.bounds[:, [1]] - self.bounds[:, [0]])  + self.bounds[:, [0]] 
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        #print("TO DO: Check that nodes are not duplicates")
        i = self.closest_node(point[:2, :])
        return np.all(self.nodes[i].point == point)
    
    def closest_node(self, point, k=1):
        #Returns the index of the closest node
        #print("TO DO: Implement a method to get the closest node to a sapled point")
        kdtree = sp.cKDTree(np.stack([node.point[:-1, :] for node in self.nodes], axis=0).squeeze(-1))
        d, i = kdtree.query(point.T, k = k)
        return i[0]
    
    def closest_node_dist(self, point, k=1):
        #Returns the index of the closest node
        #print("TO DO: Implement a method to get the closest node to a sapled point")
        kdtree = sp.cKDTree(np.stack([node.point[:-1, :] for node in self.nodes], axis=0).squeeze(-1))
        d, i = kdtree.query(point.T, k = k)
        return i[0], d[0]
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        #print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel).squeeze()
        T_wi = self.transform_2d(node_i[0], node_i[1], node_i[2])
        robot_traj[:2, :] = (T_wi @ np.concatenate((robot_traj[:2, :], np.ones((1, robot_traj.shape[1]))), axis=0))[:2, :] #np.concatenate((point.T, np.ones((point.shape[1], 1))), axis=-1)
        robot_traj[2, :] += node_i[2, 0]
        return robot_traj #(3, num_step)
    
    def check_collision(self, traj):
        # traj: (2, N)
        footprints = self.points_to_robot_circle(traj) # (N, point_per_circle, 2)
        if self.myhal:
            return np.any(self.occupancy_map[footprints[..., 0], footprints[..., 1]] == 0, axis = -1)# (N, point_per_circle)
        else:
            return np.any(self.occupancy_map[footprints[..., 1], footprints[..., 0]] == 0, axis = -1)
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        #print("TO DO: Implement a control scheme to drive you towards the sampled point")
        vels = np.linspace(-self.vel_max, self.vel_max, 6) # Change this
        rot_vels = np.linspace(-self.rot_vel_max, self.rot_vel_max, 5)
        search_grid = np.array(np.meshgrid(vels, rot_vels)).T.reshape(-1,2)

        traj = self.trajectory_rollout(search_grid[:, :1], search_grid[:, 1:])
        end_points = traj[:, :, -1] # N * 3
        T_wi = self.transform_2d(node_i[0], node_i[1], node_i[2])
        T_iw = np.linalg.inv(T_wi)
        traj_w = (T_wi @ np.concatenate((traj[:, :2, :], np.ones_like(traj[:, :1, :])), axis=1))[:, :2, :]
        traj_w = np.transpose(traj_w, (0, 2, 1)).reshape((-1, 2)).T
        in_collision = self.check_collision(traj_w).reshape(traj.shape[0], traj.shape[2])
        in_collision = np.any(in_collision, axis=1)
        search_grid = search_grid[~in_collision, :]
        end_points = end_points[~in_collision, :]

        point_s_i_xy = (T_iw @ np.array([[point_s[0, 0]], [point_s[1, 0]], [1.]]))[:2]
        #point_s_i_theta = point_s[2] - node_i[2]
        best_idx = np.argmin(np.linalg.norm(end_points[:, :-1] - point_s_i_xy.squeeze(), axis = -1), axis = 0)
        return search_grid[best_idx][0], search_grid[best_idx][1]
    
    def trajectory_rollout(self, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        N = 1 if type(vel) in (int, float) or vel.shape == () else vel.shape[0]
        traj = np.zeros((N, 3, self.num_substeps))
        t = np.linspace(0, self.timestep, self.num_substeps)
        traj[:, 0, :] = np.where(rot_vel == 0, vel * t, (vel / rot_vel) * np.sin(rot_vel * t))
        traj[:, 1, :] = np.where(rot_vel == 0, np.zeros_like(traj[:, 1, :]), (vel / rot_vel) * (1 - np.cos(rot_vel * t)))
        traj[:, 2, :] = np.where(rot_vel == 0, np.zeros_like(traj[:, 2, :]), rot_vel * t)
        #if rot_vel == 0:
            #traj[0] = vel * t
        #else:
            #traj[0] = (vel / rot_vel) * np.sin(rot_vel * t)
            #traj[1] = (vel / rot_vel) * (1 - np.cos(rot_vel * t))
            #traj[2] = rot_vel * t
        return traj #(N, 3, num_step)
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        #print("TO DO: Implement a method to get the map cell the robot is currently occupying")
        res = self.map_settings_dict["resolution"]
        point_m = self.T_mw @ np.expand_dims(np.concatenate((point.T, np.ones((point.shape[1], 1))), axis=-1), axis=-1)
        point_map_idx = np.zeros_like(point, dtype=int)
        point_map_idx[0, :] = (point_m[:, 0] / res).astype(int).squeeze()
        point_map_idx[1, :] = ((self.map_shape[0] * res - point_m[:, 1]) / res).astype(int).squeeze()
        return point_map_idx # (2, N)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        #print("TO DO: Implement a method to get the pixel locations of the robot path")
        radius_map = self.robot_radius / self.map_settings_dict["resolution"]

        point_map_idx = self.point_to_cell(points)
        footprints = [] # one for each point
        # breakpoint()
        for i in range(point_map_idx.shape[1]):
            # note: rr, cc are the indices of the pixels in the circle, not x y coordinates
            rr, cc = disk((point_map_idx[0, i], point_map_idx[1, i]), radius_map)
            rr = np.clip(rr, 0, self.map_shape[0] - 1)
            cc = np.clip(cc, 0, self.map_shape[1] - 1)
            #print(rr, cc)
            footprints.append(np.stack((rr, cc), axis=-1))
        return np.stack(footprints, axis=0) #(N, point_per_circle, 2)
    
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    @staticmethod
    def define_circle(p1, p2, p3):
        """
        Returns the center and radius of the circle passing the given 3 points.
        In case the 3 points form a line, returns (None, infinity).
        """
        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        if abs(det) < 1.0e-6:
            return (None, np.inf)
        
        # Center of circle
        cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        
        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
        return ((cx, cy), radius)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        #print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        phi = np.arctan2(point_f[1] - node_i[1], point_f[0] - node_i[0]) - node_i[2]
        phi = (phi + np.pi) % (2 * np.pi) - np.pi

        T_wi = self.transform_2d(node_i[0, 0], node_i[1, 0], node_i[2, 0])
        T_iw = np.linalg.inv(T_wi)
        point_f_i = (T_iw @ np.concatenate((point_f, np.ones_like(point_f[:1])), axis=0))[:2]
        point_f_i_m = np.asarray(point_f_i) * np.array([[-1.], [1.]])
        center_i, r = self.define_circle(np.zeros((2, 1)), point_f_i, point_f_i_m)

        if np.isfinite(r):
            if point_f_i[0, 0] > 0:
                vel = np.abs(phi * r)
                if center_i[1] > 0:
                    rot_vel = vel / r
                else:
                    rot_vel = -vel / r
            else:
                vel = -np.abs((np.pi - phi) * r)
                if center_i[1] > 0:
                    rot_vel = -vel / r
                else:
                    rot_vel = vel / r
        else:
            vel = point_f_i[0, 0] - 0.
            rot_vel = 0.
        
        if np.abs(vel) < self.vel_max and np.abs(rot_vel) < self.rot_vel_max:
            traj = self.trajectory_rollout(vel, rot_vel).squeeze(0)
        else:
            return np.ones((3, self.num_substeps)) * np.NaN
        if np.any(self.check_collision(traj[:2, :])):
            return np.ones((3, self.num_substeps)) * np.NaN
        else:
            return traj
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        #print("TO DO: Implement a cost to come metric")
        return np.sum(np.linalg.norm(trajectory_o[1:, :2] - trajectory_o[:-1, :2], axis = -1))
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        #print("TO DO: Update the costs of connected nodes after rewiring.")
        '''q = []
        visited = [node_id]
        children = self.nodes[node_id].children_ids
        #print(node_id)
        #print(f"0: {children}")
        q += [(child, old_cost) for child in children]
        i = 0
        while q != []:
            child, old_cost = q.pop(0)
            if child in visited:
                #print(child)
                #print(visited)
                raise KeyboardInterrupt
            visited.append(child)
            #traj = self.connect_node_to_point(self.nodes[node_id].point, self.nodes[child].point[0:2])
            #assert not np.any(np.isnan(traj))
            child_old_cost = self.nodes[child].cost
            self.nodes[child].cost = self.nodes[node_id].cost + self.nodes[child].cost - old_cost
            #if i <= 3:
                #print(f"{i+1}: {self.nodes[child].children_ids}")
            self.update_children(child, child_old_cost)
            q += [(childchild, child_old_cost) for childchild in self.nodes[child].children_ids]
            i += 1'''
        self.nodes[node_id].cost = self.nodes[self.nodes[node_id].parent_id].cost + self.cost_to_come(self.nodes[node_id].traj_arrive)

        for child in self.nodes[node_id].children_ids:
            self.update_children(child)

        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        i = 0
        #override_point = False
        while True: #Most likely need more iterations than this to complete the map!
            
            #Sample map space
            #if not override_point:
                #point = self.sample_map_space()
            #else:
                #override_point = False
            point = self.sample_map_space()
            #self.window.add_point(point[:, 0].copy(), color = (255, 0, 0)) #delete me

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            if self.check_if_duplicate(trajectory_o[:, [-1]]):
                #print(1)
                continue

            #Check for collisions
            footprints = self.points_to_robot_circle(trajectory_o[:2, :])
            
            if np.any(self.occupancy_map[footprints[..., 1], footprints[..., 0]] == 0):
                #if np.random.rand() < 0.5:
                    #point = self.nodes[closest_node_id].point[:-1, :] + np.random.rand(2, 1) * self.vel_max * 50 - self.vel_max * 25
                    #override_point = True
                continue

            self.nodes.append(Node(point = trajectory_o[:, [-1]], parent_id = closest_node_id, cost = 0))
            #self.window.add_point(trajectory_o[:-1, -1].copy()) #delete me
            #self.window.add_line(self.nodes[closest_node_id].point[:-1, 0].copy(), trajectory_o[:-1, -1].copy(), width = 3, color = (0, 0, 255))
            
            #Check if goal has been reached
            #print("TO DO: Check if at goal point.")
            if np.linalg.norm(trajectory_o[:2, [-1]] - self.goal_point) < self.stopping_dist:
                break
            i += 1
            print(f"iter: {i}")

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot  
        i = 0
        #override_point = False      
        for _ in range(1000): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()
            #if not override_point:
                #point = self.sample_map_space()
            #else:
                #override_point = False
            self.window.add_point(point[:, 0].copy(), color = (255, 0, 0)) #delete me

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)
            if self.check_if_duplicate(trajectory_o[:, [-1]]):
                #print(1)
                continue

            #Check for collisions
            footprints = self.points_to_robot_circle(trajectory_o[:2, :])
            
            if self.myhal:
                if np.any(self.occupancy_map[footprints[..., 0], footprints[..., 1]] == 0):
                    continue
            else:
                if np.any(self.occupancy_map[footprints[..., 1], footprints[..., 0]] == 0):
                    #if np.random.rand() < 0.5:
                        #point = self.nodes[closest_node_id].point[:-1, :] + np.random.rand(2, 1) * self.vel_max * 50 - self.vel_max * 25
                        #override_point = True
                    continue
            
            cost_to_come = self.cost_to_come(trajectory_o) + self.nodes[closest_node_id].cost
            self.nodes.append(Node(point = trajectory_o[:, [-1]], parent_id = closest_node_id, cost = cost_to_come,
                traj_arrive=trajectory_o))
            self.window.add_point(trajectory_o[:-1, -1].copy()) #delete me
            self.window.add_line(self.nodes[closest_node_id].point[:-1, 0].copy(), trajectory_o[:-1, -1].copy(), width = 3, color = (0, 0, 255)) #delete me
            print(self.nodes[closest_node_id].point[:-1, 0].copy(), trajectory_o[:-1, -1].copy())

            self.nodes[closest_node_id].children_ids.append(len(self.nodes) - 1)

            #Last node rewire
            #print("TO DO: Last node rewiring")
            end_point = trajectory_o[:, [-1]]
            kdtree = sp.cKDTree(np.stack([node.point[:-1, :] for node in self.nodes[:-1]], axis=0).squeeze(-1))
            close_idx = kdtree.query_ball_point(end_point[:2,0], r = self.ball_radius())
            best_ctc = np.inf
            best_id = closest_node_id
            for close_node_id in close_idx:
                new_traj = self.connect_node_to_point(self.nodes[close_node_id].point, end_point[:2])
                if np.any(np.isnan(new_traj)):
                    continue
                curr_ctc = self.cost_to_come(new_traj) + self.nodes[close_node_id].cost
                if curr_ctc < best_ctc:
                    best_ctc = curr_ctc
                    best_id = close_node_id

            if best_id != closest_node_id:
                self.nodes[closest_node_id].children_ids.remove(len(self.nodes) - 1)
                self.nodes[best_id].children_ids.append(len(self.nodes) - 1)
                self.nodes[-1].parent_id = best_id
                self.nodes[-1].cost = best_ctc
                self.nodes[-1].traj_arrive = new_traj

            #Close node rewire
            #print("TO DO: Near point rewiring")
            for close_node_id in close_idx:
                new_traj = self.connect_node_to_point(end_point, self.nodes[close_node_id].point[:2])
                if np.isnan(new_traj).any():
                    continue
                new_ctc = self.cost_to_come(new_traj) + self.nodes[-1].cost
                #print(f"ctc: {self.cost_to_come(new_traj)} || ccc: {self.nodes[-1].cost}")
                if new_ctc < self.nodes[close_node_id].cost:
                    self.nodes[-1].children_ids.append(close_node_id)
                    self.nodes[self.nodes[close_node_id].parent_id].children_ids.remove(close_node_id)
                    self.nodes[close_node_id].parent_id = len(self.nodes) - 1
                    self.nodes[close_node_id].cost = new_ctc
                    self.nodes[close_node_id].traj_arrive = new_traj
                    #print(f"rewire: {len(self.nodes) - 1}")
                    self.update_children(close_node_id)
            '''
            

            #Close node rewire
            #print("TO DO: Near point rewiring")
            for close_node_idx in idx:
                new_traj = self.connect_node_to_point(self.nodes[-1].point, self.nodes[close_node_idx].point[:-1])
                if np.any(np.isnan(new_traj)):
                    continue
                ctc = self.cost_to_come(new_traj) + self.nodes[-1].cost
                if ctc < self.nodes[close_node_idx].cost:
                    self.window.add_line(self.nodes[self.nodes[close_node_idx].parent_id].point[:-1, 0].copy(), self.nodes[close_node_idx].point[:-1, 0].copy(), width = 3, color = (255, 255, 255))
                    self.nodes[self.nodes[close_node_idx].parent_id].children_ids.remove(close_node_idx)
                    self.nodes[close_node_idx].parent_id = len(self.nodes)-1
                    old_cost = self.nodes[close_node_idx].cost
                    self.nodes[close_node_idx].cost = ctc
                    self.nodes[-1].children_ids.append(close_node_idx)
                    self.update_children(close_node_idx, old_cost)
                    
                    self.window.add_line(trajectory_o[:-1, -1].copy(), self.nodes[close_node_idx].point[:-1, 0].copy(), width = 3, color = (0, 0, 255))'''

            #Check for early end
            if np.linalg.norm(trajectory_o[:2, [-1]] - self.goal_point) < self.stopping_dist:
                self.goal_idx = i
                #break
            i += 1
            print(f"iter: {i}")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    #map_filename = "willowgarageworld_05res.png"
    map_filename = "myhal.png"
    #map_setings_filename = "willowgarageworld_05res.yaml"
    map_setings_filename = "myhal.yaml"

    #robot information
    #goal_point = np.array([[42], [-44]]) #m
    #goal_point = np.array([[42], [-44]]) #m
    goal_point = np.array([[7], [0]])
    stopping_dist = 0.3 #m
    #stopping_dist = 0.2 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, myhal=True)
    #nodes = path_planner.rrt_planning()
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path(node_id=path_planner.goal_idx))

    last_node = node_path_metric[:, 0]
    for node_idx in range(1, node_path_metric.shape[1]):
        path_planner.window.add_line(last_node[:2].copy(), node_path_metric[:2, node_idx].copy(), width = 3, color = (0, 0, 0))
        last_node = node_path_metric[:2, node_idx]
    save_name = input("Save name:")

    #Leftover test functions
    #np.save(f"shortest_path_RRT_{save_name}.npy", node_path_metric)
    np.save(f"shortest_path_RRT_{save_name}_myhal.npy", node_path_metric)
    #np.save(f"shortest_path_RRT_star_{save_name}.npy", node_path_metric)
    #np.save(f"shortest_path_RRT_star_{save_name}_myhal.npy", node_path_metric)

def draw():
    #Set map information
    #map_filename = "willowgarageworld_05res.png"
    map_filename = "myhal.png"
    #map_setings_filename = "willowgarageworld_05res.yaml"
    map_setings_filename = "myhal.yaml"

    #robot information
    #goal_point = np.array([[42], [-44]]) #m
    goal_point = np.array([[7], [0]]) #m
    stopping_dist = 0.3 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    #nodes = path_planner.rrt_planning()
    #nodes = path_planner.rrt_star_planning()
    #node_path_metric = np.hstack(path_planner.recover_path())

    node_path_metric = np.load("shortest_path_RRT_myhal_myhal.npy")

    last_node = node_path_metric[:, 0]
    for node_idx in range(1, node_path_metric.shape[1]):
        path_planner.window.add_line(last_node[:2].copy(), node_path_metric[:2, node_idx].copy(), width = 3, color = (0, 0, 0))
        last_node = node_path_metric[:2, node_idx]
    input("Press Enter to Exit")
    pygame.image.save(path_planner.window.screen, "RRT_myhal.png")


if __name__ == '__main__':
    main()
    #draw()
