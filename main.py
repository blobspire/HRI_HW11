import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# vehicle dynamics
def dynamics(state, action):
    delta = 1.0 * np.array([np.cos(action), np.sin(action)])
    new_state = state + delta
    return new_state

# trajectory rollout
def rollout(initial_state, actions):
    state = np.copy(initial_state)
    xi = [state.tolist()]
    for a in actions:
        state = dynamics(state, a)
        xi.append(state.tolist())
    return np.array(xi)

# human cost function
def human_cost(human_actions, initial_human_state, initial_robot_state, robot_actions):
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    cost = 0.
    for idx, x in enumerate(zip(human_trajectory, robot_trajectory)):
        human_state = x[0]
        robot_state = x[1]
        # design your state cost (for the human) here!
        cost += 1/np.linalg.norm(human_state - robot_state)
        cost += -human_state[0]
    return cost

# robot cost function
def robot_cost(robot_actions, initial_human_state, initial_robot_state, human_actions):
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    cost = 0.
    for idx, x in enumerate(zip(human_trajectory, robot_trajectory)):
        human_state = x[0]
        robot_state = x[1]
        # design your state cost (for the robot) here!
        cost += -human_state[0]
    return cost

# plot the trajectories
def plot_trajectory(initial_human_state, initial_robot_state, human_actions, robot_actions):
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    plt.plot(human_trajectory[:,0], human_trajectory[:,1], 'bo-')
    plt.plot(robot_trajectory[:,0], robot_trajectory[:,1], 'ro-')
    plt.savefig("result.png")

# initialize the vehicles
initial_human_state = np.array([-7.0, 0.])
initial_robot_state = np.array([-6.0, 0.1])
robot_actions = np.array([0., 0., 0., 0., 0.])
human_actions = np.zeros_like(robot_actions)

# optimize the human's actions
result = minimize(
            fun=human_cost, 
            x0=human_actions, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi) for _ in range(len(human_actions))]
            )
human_actions = result.x

# plot the results (saves as a png file)
plot_trajectory(initial_human_state, initial_robot_state, human_actions, robot_actions)
