import numpy as np
from pinocchio import RobotWrapper


def pin_gen_coords_to_plot(robot: RobotWrapper, q: np.ndarray):
    """
    Generate the coordinates for plotting of pincochio robot states.
     - The function makes continuous joints described by cos(ø) and sin(ø) to be described by ø for plotting.
    Args:
        robot (RobotWrapper): pinocchio robot wrapper
        q (np.ndarray): pinocchio joint configuration

    Returns:
        np.ndarray: coordinates to plot the robot
    """

