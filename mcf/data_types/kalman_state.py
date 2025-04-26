import numpy as np
from mcf.data_types.point import Point

class KalmanState:
    def __init__(self, time_step: int=1):
        self.predicted_position = Point(0,0)
        self.predicted_velocity = Point(0,0)
        self.measured_position = Point(0,0)
        self.measured_velocity = Point(0,0)
        self.position = Point(0,0)
        self.velocity = Point(0,0)

        self.process_covariance = 0
        self.measurement_covariance = 0
        
        self.covariance = np.array([[]]) # covariance matrix
        self.observation = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]]) # observation matrix
        
        self.dt = time_step
        
        self.state_transition = np.array([[1, 0, self.dt, 0],
                                          [0, 1, 0, self.dt],
                                          [0, 0, 1, 0 ],
                                          [0, 0, 0, 1 ]])

        # TODO: unused?
        self.input_transition = np.array([[0],
                                          [0],
                                          [0],
                                          [0]])

    def getCurrent(self):
        return np.array([[self.position.x, self.position.y, self.velocity.x, self.velocity.y]]).T

    def getPrediction(self):
        return np.array([[self.predicted_position.x, self.predicted_position.y, self.predicted_velocity.x, self.predicted_velocity.y]]).T
    
    def getMeasurement(self):
        return np.array([[self.measured_position.x, self.measured_position.y, self.measured_velocity.x, self.measured_velocity.y]]).T
    