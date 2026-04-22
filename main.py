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
        cost += 1/np.linalg.norm(human_state - robot_state) # Stay away from the robot (cost bigger as closer)
        # cost += -human_state[0] # Maximize distance along first (x) axis (left to right)

        # If collide, add big cost
        if np.linalg.norm(human_state - robot_state) < 0.01:
            cost += 1000
            break
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

        # If collide, add big cost
        if np.linalg.norm(human_state - robot_state) < 0.01:
            cost += 1000
            break
    return cost

def nested_cost(human_actions_local, robot_actions, initial_human_state, initial_robot_state):
    # Optimize for the human actions
    # fun human cost, optimize human actions
    # human actions = results
    result = minimize(
            fun=human_cost, 
            x0=human_actions_local, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='L-BFGS-B',
            bounds=[(-np.pi/8, np.pi/8) for _ in range(len(human_actions_local))] # bounds: how human can turn (default: -pi to pi)
            )
    global human_actions
    human_actions = result.x
    human_actions_local = result.x
    return robot_cost(robot_actions, initial_human_state, initial_robot_state, human_actions_local)

# plot the trajectories
def plot_trajectory(initial_human_state, initial_robot_state, human_actions, robot_actions):
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    plt.plot(human_trajectory[:,0], human_trajectory[:,1], 'bo-')
    plt.plot(robot_trajectory[:,0], robot_trajectory[:,1], 'ro-')
    plt.axis('equal')
    # Add human and robot labels
    plt.text(initial_human_state[0], initial_human_state[1], 'Human', fontsize=12, color='blue')
    plt.text(initial_robot_state[0], initial_robot_state[1], 'Robot', fontsize=12, color='red')
    # Add time labels # TODO could also use alpha to show (lighter = past, darker = current)
    for i in range(len(human_trajectory)):
        plt.text(human_trajectory[i,0], human_trajectory[i,1], f't={i}', fontsize=8, color='blue')
        plt.text(robot_trajectory[i,0], robot_trajectory[i,1], f't={i}', fontsize=8, color='red')
    plt.savefig("result.png")

# initialize the vehicles
initial_human_state = np.array([-7.0, 0.]) # Same y would cause local minima
initial_robot_state = np.array([-6.0, 0.1])
robot_actions = np.array([0., 0., 0., 0., 0.]) # actions are slopes (turning angle)
human_actions = np.zeros_like(robot_actions)

# optimize the human's actions
result = minimize(
            fun=nested_cost, 
            x0=robot_actions, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='L-BFGS-B',
            bounds=[(-np.pi/8, np.pi/8) for _ in range(len(human_actions))] # bounds: how human can turn (default: -pi to pi)
            )
robot_actions = result.x

# plot the results (saves as a png file)
plot_trajectory(initial_human_state, initial_robot_state, human_actions, robot_actions)
