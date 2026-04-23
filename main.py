import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# vehicle dynamics
def dynamics(state, action): # state is (x, y)
    theta, speed = action
    delta = speed * np.array([np.cos(theta), np.sin(theta)])
    new_state = state + delta
    return new_state

# trajectory rollout
def rollout(initial_state, actions): # n + 1 length trajectory for n actions, bc initial state
    state = np.copy(initial_state)
    xi = [state.tolist()]
    for a in actions:
        state = dynamics(state, a)
        xi.append(state.tolist())
    return np.array(xi)

# Helper function to reshape the actions from the optimization (which is a flat array) to the shape we need for rollout (n, 2)
def unpack_actions(actions):
    return actions.reshape(-1, 2) # reshape to (n, 2) where n is number of time steps and 2 is (theta, speed)

# human cost function
def human_cost(human_actions, initial_human_state, initial_robot_state, robot_actions):
    human_actions = unpack_actions(human_actions)
    robot_actions = unpack_actions(robot_actions)
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    cost = 0.
    for idx, x in enumerate(zip(human_trajectory, robot_trajectory)):
        human_state = x[0]
        robot_state = x[1]
        # design your state cost (for the human) here!
        cost += 1/(np.linalg.norm(human_state - robot_state) + 1e-6) # Stay away from the robot (cost bigger as closer)
        cost += -human_state[0] # Maximize distance along first (x) axis (left to right)

        # # If collide, add big cost
        # if np.linalg.norm(human_state - robot_state) < 0.1:
        #     cost += 1000
        #     break
    return cost

# robot cost function
def robot_cost(robot_actions, initial_human_state, initial_robot_state, human_actions):
    human_actions = unpack_actions(human_actions)
    robot_actions = unpack_actions(robot_actions)
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    cost = 0.
    for idx, x in enumerate(zip(human_trajectory, robot_trajectory)):
        human_state = x[0]
        robot_state = x[1]
        
        # cost -= 1/(np.linalg.norm(human_state - robot_state) + 1e-6) # Robot wants to be close to human
        cost += human_state[0] # For robot to slow human, remove negation (cost = human's progress)

        # If collide, add big cost
        # if np.linalg.norm(human_state - robot_state) < 0.01:
        #     cost += 1000
        #     break
    return cost

# Determine the robot's cost for a given human plan
def nested_cost(robot_actions, initial_human_state, initial_robot_state, human_actions):
    # For each candidate robot plan, solve the human's best response
    result = minimize(
            fun=human_cost, 
            x0=human_actions, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi) for _ in range(len(human_actions))] # bounds: how human can turn (default: -pi to pi)
            )

    human_actions = result.x
    return robot_cost(robot_actions, initial_human_state, initial_robot_state, human_actions)

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
initial_robot_state = np.array([-6.0, 0])

time_steps = 10
robot_actions = np.tile([0.0, 0.5], time_steps) # actions are slopes (turning angle)
human_actions = np.tile([0.0, 0.5], time_steps)

bounds = []
for _ in range(time_steps):
    bounds.append((-np.pi, np.pi)) # theta bounds: how agent can turn (default: -pi to pi)
    bounds.append((0, 1)) # speed bounds: how fast agent can go (default: 0 to 1)

# Optimize the robot's actions given human's best response
result = minimize(
            fun=nested_cost, 
            x0=robot_actions, 
            args=(initial_human_state, initial_robot_state, human_actions),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi) for _ in range(len(robot_actions))] # bounds: how human can turn (default: -pi to pi)
            )
robot_actions = result.x

# Solve for the human's best response to the final robot plan. This best response was determined within the nested optimization so we can use the robot's plan to re-solve for it 
# We could also use a global variable to extract the human's best response from within the nested optimization but this is cleaner
result = minimize(
            fun=human_cost, 
            x0=human_actions, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi) for _ in range(len(human_actions))] # bounds: how human can turn (default: -pi to pi)
            )
human_actions = result.x

# Remove the png to ensure we're viewing an updated result
if os.path.exists("result.png"):
    os.remove("result.png")
# plot the results (saves as a png file)
plot_trajectory(initial_human_state, initial_robot_state, human_actions, robot_actions)
