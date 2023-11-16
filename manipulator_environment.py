import numpy as np
from matplotlib import pyplot as plt


class Planar_Environment(object):
    '''
    Environement for planar n-rotary joint robotic arm
    '''
    def __init__(self, action_bound = 0.1, configuration=[('R', 10), ('R', 10)], start_angles = None, threshold = 1e-1, step_cost=1/400):
        self.action_bound = np.array([action_bound])
        self.configuration = configuration
        self.num_joints =len(configuration) # dont actually need this, but I like having it

        # Sets starting angle of the joints
        if start_angles is None:
            self.start_angles = [0.] * self.num_joints
        elif len(start_angles) != self.num_joints:
            raise Exception("provided start angles do not match number of joints")
        else:
            self.start_angles = start_angles

        self.threshold = threshold # minimum distance from end point to goal to be considered done
        self.step_cost = step_cost # incremental cost (-reward) per step to incentivize non zero action when close to goal

        self.joint_end_points = [] # list of tuples representing the end points of each joint in the arm
        self.working_radius = sum(joint[1] for joint in self.configuration) # max reachable radius, only used for plotting
        self.joint_angles = np.array(self.start_angles, dtype=float) # keeping this as an array makes step() easier, maybe everything should just be an ndarray
        
        '''
        State is 1+2*{num_joints}+2+2 vector whose first element is the current step, next 2*{num_joints} elements represent cos's of joint angles then sin's of joint angles, 
        the next two elements represent the x,y of the end effector and last two represent x,y of goal
        '''
        self.state, _ = self.reset() # dont think its a problem that we call .reset() when we init env and then also call reset outside but idk
        
        # redundant here but makes code more clear when calling from outside the class
        self.action_dim = self.num_joints
        self.state_dim = len(self.state)

    def gen_goal(self):
        '''
        Generates a coordinate in manipulators working space
        '''
        rand_angs = np.random.uniform(-np.pi, np.pi, self.num_joints)
        goal_point  = [0, 0]
        cur_angle = 0
        for i, joint in enumerate(self.configuration):
            cur_angle += rand_angs[i]
            goal_point[0] += joint[1]*np.cos(cur_angle)
            goal_point[1] += joint[1]*np.sin(cur_angle)

        return goal_point

    def reset(self):
        '''
        Resets arm to start position, generates a new goal and returns them combined as the state vector
        '''
        new_goal = self.gen_goal() # 2d point (planar goal)

        # self.joint_angles = np.random.uniform(-np.pi, np.pi, self.num_joints)
        self.joint_angles = np.copy(np.array(self.start_angles))
        
        # compute end effector position
        end_point = [0, 0]
        self.joint_end_points = [end_point.copy()]
        cur_angle = 0
        # iterate through joints
        for i, joint in enumerate(self.configuration):
            cur_angle += self.joint_angles[i] # current frame angle += angle of joint i
            end_point[0] += joint[1]*np.cos(cur_angle) # get end point of joint i wrt world frame
            end_point[1] += joint[1]*np.sin(cur_angle)
            self.joint_end_points.append(end_point.copy())

        self.state = np.concatenate(([0], np.cos(self.joint_angles), np.sin(self.joint_angles), self.joint_end_points[-1], new_goal))
        
        return np.copy(self.state), {} # returns the dict to match returns from gym .reset()
 
    def step(self, action, step):
        '''
        Steps the manipulator by the action provided. Action is expected to be delta_joint_angle in radians
        '''
        if len(action) != self.num_joints:
            print("Error: action sent to environment does not match the number of joints")
            return None # in this condition should maybe exit instead 
        
        self.joint_angles += action # add commanded delta angles to joint angles

        # compute end effector position
        end_point = [0, 0]
        self.joint_end_points = [end_point.copy()]
        cur_angle = 0
        # iterate through joints
        for i, joint in enumerate(self.configuration):
            cur_angle += self.joint_angles[i] # current frame angle += angle of joint i
            end_point[0] += joint[1]*np.cos(cur_angle) # get end point of joint i wrt world frame
            end_point[1] += joint[1]*np.sin(cur_angle)
            self.joint_end_points.append(end_point.copy())

        self.state = np.concatenate(([step], np.cos(self.joint_angles), np.sin(self.joint_angles), self.joint_end_points[-1], self.state[-2:]))

        distance_to_goal = np.linalg.norm(np.array(self.joint_end_points[-1]) - self.state[-2:])
        reward = -distance_to_goal -self.step_cost*step # reward is the negative distance from end point to goal with an additional cost for every step

        done = distance_to_goal <= self.threshold # episode is done if distance to goal is less than threshold

        return np.copy(self.state), reward, done, False, {} # last two returns are just to match gym .step()
    

    def viz_arm(self, axs=None):
        '''
        Plots the joints with the workspace outline
        '''
        show = True if axs is None else False
        x_coors, y_coors = zip(*self.joint_end_points)

        if axs is None:
            axs = plt.gca()
        # fig, axs = plt.subplots()
        circle = plt.Circle((0, 0), self.working_radius, fill = False)
        axs.set_xlim(-1.25*self.working_radius, 1.25*self.working_radius)
        axs.set_ylim(-1.25*self.working_radius, 1.25*self.working_radius)
        axs.set_aspect(1)
        axs.add_artist(circle)

        axs.plot(x_coors, y_coors, marker='o', linestyle='-')
        axs.scatter(self.state[-2], self.state[-1], marker = 'o', color='green')
        axs.grid(True)

        if not show:
            return axs
        plt.show() 
