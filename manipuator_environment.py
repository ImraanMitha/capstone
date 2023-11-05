# this class needs to take in an action and return the state. it should eventually be flexible to arms of different shapes and joints

# the state should just be the end effector position and the goal position right?
# maybe at some later point I will change actions to be torques instead of angles to go to, in which case I would need to add velocity information to state
# TODO: should implement a render function to demo the movement control
import numpy as np

class Planar_Environment(object):
    
    def __init__(self, num_joints=2,configuration=[('R', 10), ('R', 10)], start_pos = [1, 1], threshold = 1e-3):
        self.num_joints = num_joints # dont actually need this, can just use len(configuration)
        self.configuration = configuration

        self.start_pos = start_pos
        self.threshold = threshold # minimum distance from end point to goal to be considered done
        self.working_radius = sum(joint[1] for joint in self.configuration) # only works for planar r manipulators
        self.state = self.reset() # state is 1x4 vector whose first two elements represent end point of manipulator and last two represent pos of goal

    # generates a coordinate in manipulators working space
    # right now this only works for planar RR manipulator 
    def gen_goal(self):
        rand_rad = np.random.uniform(0, self.working_radius)
        rand_ang = np.random.uniform(-np.pi, np.pi)
        return [rand_rad * np.cos(rand_ang), rand_rad * np.sin(rand_ang)]

    # resets arm to start position, generates a new goal and returns them combined as the state vector
    def reset(self):
        new_goal = self.gen_goal()
        self.state = np.array(self.start_pos + new_goal)
        return np.copy(self.state) # TODO: need to figure this out, if should be returning copy or reference
        # return self.state
    
    def step(self, action):
        if len(action) != self.num_joints:
            print("Error: action sent to environment does not match the number of joints")
            return None # in this condition should maybe exit instead 
        
        #### calculate end point, ie fwd kinematics, this implementation is only for planar RR joints ####
        # assuming action is desired angle in radian in [-pi, pi]
        end_point = [0, 0]
        cur_angle = 0
        for i, joint in enumerate(self.configuration):

            angle = action[i] # joint angle command to joint i

            if joint[0] == 'R':
                cur_angle += angle
                end_point[0] += joint[1]*np.cos(cur_angle)
                end_point[1] += joint[1]*np.sin(cur_angle)

            elif joint[0] == 'P':
                print("havent implemented prismatic yet")
                return None # in this condition should maybe exit instead 
            
            else:
                print('messed up configuration to planar_env object, got a joint type thats not R or P')
                return None # in this condition should maybe exit instead 

        reward = -np.linalg.norm(np.array(end_point) - self.state[2:]) # reward is the negative distance from end point to goal

        self.state[:2] = end_point # sets first two elements of state to be the new end point of arm 

        done = -reward <= self.threshold # episode is done if distance from end point to goal is less than threshold

        return np.copy(self.state), reward, done # TODO: need to figure this out, if should be returning copy or reference
        # return self.state, reward, done
        

