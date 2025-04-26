import numpy as np
from mcf.data_types.kalman_state import KalmanState
from mcf.data_types import Point

def run_kalman(state: KalmanState):
        _predict_state(state)
        _predict_covariance(state)
        _kalman_filter(state)

def _predict_state(state: KalmanState) -> np.array:
    ps = state.state_transition @ state.getCurrent()
    state.predicted_position = Point(ps[0][0], ps[1][0])
    state.predicted_velocity = Point(ps[2][0], ps[3][0])

def _predict_covariance(state: KalmanState) -> np.array:
    if state.covariance.size == 0:
        state.covariance = state.process_covariance
    state.covariance = state.state_transition @ state.covariance @ state.state_transition.T + state.process_covariance

def _kalman_filter(state: KalmanState):
    # kalman gain
    K = state.covariance @ state.observation.T @ np.linalg.inv(state.observation @ state.covariance @ state.observation.T + state.measurement_covariance)

    # kalman update
    s_t = state.getPrediction() + K @ (state.observation @ state.getMeasurement() - state.observation @ state.getPrediction())
    state.position = Point(s_t[0][0], s_t[1][0])
    state.velocity = Point(s_t[2][0], s_t[3][0])
    state.covariance = (np.eye(4) - K @ state.observation) @ state.covariance
    