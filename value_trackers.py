from collections import deque
import numpy as np

# tracks changes in a variable w/ fixed size buffer as debounce
class StateTracker:
    def __init__(self, var: str, size: int, default=None):
        self.var = var
        self.size = size
        self.prev_state = None
        self.curr_state = self.default = default
        self.state_changed = False
        self.buffer = deque([default] * size, maxlen=size) # TODO: init w/ empty deque
        
    def update(self, new_value):
        self.buffer.append(new_value)
        # print(f"Updating {self.var} tracker with {new_value}: {self.buffer}, {self.curr_state=}, {self.prev_state=}")
        
        last = self.buffer[-1]
        if all(val == last for val in self.buffer) and last != self.curr_state:
            self.prev_state = self.curr_state
            self.curr_state = last
            self.state_changed = True
        else:
            self.state_changed = False
            
    def clear(self):
        self.curr_state = self.default
        self.prev_state = None
        self.state_changed = False
        self.buffer.clear()
        
    def __str__(self):
        return f"{self.var} tracker: {str(self.curr_state)} | {str(self.buffer)}"

# detects sudden changes in movement using stdev of distances
class MovementTracker:
    def __init__(self, size=5, thresh_stdev=10):#, default_timeout=5):
        self.coords = deque(maxlen=size)
        self.change_detected = False
        self.thresh_stdev = thresh_stdev
        
        # self.curr_timeout = 0
        # self.default_timeout = default_timeout
    
    def calculate_stdev(self):
        coords = np.array(self.coords)
        distances = np.linalg.norm(coords - np.mean(coords, axis=0), axis=1)
        stdev = np.std(distances)
        return stdev
        
    def update(self, new_coords: np.ndarray):
        # add new position
        self.coords.append(new_coords)

        if len(self.coords) == self.coords.maxlen:
            stdev = self.calculate_stdev()

            # check if stdev is below the threshold
            if stdev >= self.thresh_stdev:# and self.curr_timeout == 0: # TODO: decide between >= and <= threshold
                # print("Sudden change detected!")
                print(f"Current stdev: {stdev:.2f}")
                self.change_detected = True
                # self.curr_timeout = self.default_timeout
            else:
                self.change_detected = False
                # self.curr_timeout = max(0, self.curr_timeout - 1)
        
    def clear(self):
        self.coords.clear()
        self.change_detected = False
        # self.curr_timeout = 0

# detects sudden changes in movement using acceleration (derived from coords)
class AccelerationTracker:
    def __init__(self, size=5, thresh_accel=10.0):#, default_timeout=5):
        self.positions = deque(maxlen=size)
        # self.distances = deque(maxlen=size)
        self.velocities = deque(maxlen=size)
        self.accelerations = deque(maxlen=size)
        self.jerks = deque(maxlen=size)

        self.thresh_accel = thresh_accel        
        self.change_detected = False
        
        # self.curr_timeout = 0
        # self.default_timeout = default_timeout
    
    def calculate_stdev(self):
        accels = np.array(self.accelerations)
        stdev = np.std(accels)
        return stdev
    
    def update(self, new_coords: np.ndarray):
        # window_size = 5
        # add new position
        self.positions.append(new_coords)
        
        if len(self.positions) > 1:
            # calculate velocity
            vel = np.mean(np.linalg.norm(np.diff(self.positions, axis=0), axis=1))
            self.velocities.append(vel)
            
            # calculate acceleration
            if len(self.velocities) > 1:
                accel = np.mean(self.velocities)
                self.accelerations.append(accel)
                
                if len(self.accelerations) > 1:
                    stdev = self.calculate_stdev()
                    print(accel, stdev)
                    
                    if stdev >= self.thresh_accel:
                        print(f"Current acceleration stdev: {stdev:.2f}")
                        self.change_detected = True
                    else:
                        self.change_detected = False
                    # accel_strength = np.linalg.norm(self.accelerations[-1])
                    # if accel_strength > self.thresh_accel:
                    #     print(f"Current acceleration magnitude: {accel_strength:.2f}")
                    #     self.change_detected = True
                    # else:
                    #     self.change_detected = False

                # if len(self.accelerations) > window_size:
                #     # calculate jerk
                #     jerk = self.accelerations[-window_size:].mean()
                #     self.jerks.append(jerk)
            
                #     if len(self.jerks) >= 1:
                #         jerk_strength = np.linalg.norm(self.jerks[-1])
                #         if jerk_strength > self.thresh_accel:
                #             print(f"Current jerk magnitude: {jerk_strength:.2f}")
                #             self.change_detected = True
                #         else:
                #             self.change_detected = False

        # if len(self.velocities) > 1:
        #     acceleration = self.velocities[-1] - self.velocities[-2]
        #     self.accelerations.append(acceleration)
          
        # if len(self.accelerations) > 1:
        #     jerk = self.accelerations[-1] - self.accelerations[-2]
        #     self.jerks.append(jerk)
            
        # if len(self.jerks) > 1:
        #     jerk_strength = np.linalg.norm(self.jerks[-1])
        #     if jerk_strength > self.thresh_accel:
        #         print(f"Current jerk magnitude: {jerk_strength:.2f}")
        #         self.change_detected = True
        #     else:
        #         self.change_detected = False
            
        # if len(self.accelerations) >= 1:
        #     current_accel_strength = np.linalg.norm(self.accelerations[-1])
        #     if current_accel_strength > self.thresh_accel:
        #         print(f"Current acceleration magnitude: {current_accel_strength:.2f}")
        #         self.change_detected = True
        #     else:
        #         self.change_detected = False

        # # calculate velocity if there are at least two positions
        # if len(self.positions) > 1:
        #     vel = np.linalg.norm(self.positions[-1] - self.positions[-2])
        #     self.velocities.append(vel)

        # # calculate acceleration if there are at least two velocities
        # if len(self.velocities) > 1:
        #     accel = np.linalg.norm(self.velocities[-1] - self.velocities[-2])
        #     self.accelerations.append(accel)

        # # check for sudden acceleration changes
        # if len(self.accelerations) >= 1:
        #     current_accel_strength = self.accelerations[-1]
        #     if current_accel_strength > self.thresh_accel:
        #         print(f"Current acceleration magnitude: {current_accel_strength:.2f}")
        #         self.change_detected = True
                    
        #         # if self.curr_timeout == 0:
        #         #     # print("Sudden change detected!")
        #         #     print(f"Current acceleration magnitude: {current_accel_strength:.2f}")
        #         #     self.change_detected = True
        #         #     self.curr_timeout = self.default_timeout
        #         # else:
        #         #     print(f"still waiting on timeout for {self.curr_timeout} frames")
        #         #     self.change_detected = False
        #         #     self.curr_timeout = max(0, self.curr_timeout - 1)
        #     else:
        #         self.change_detected = False
        #         # self.curr_timeout = max(0, self.curr_timeout - 1) # nesting this under the if -> threshold for # of detected accelerations, not # of frames
        
        # print(self.accelerations)
        
    def clear(self):
        for buffer in (self.positions, self.velocities, self.accelerations):
            buffer.clear()
        self.change_detected = False
        # self.curr_timeout = 0