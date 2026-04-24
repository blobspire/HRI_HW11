import os
import atexit
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize


# Configuration
AGENT_RADIUS = 0.5
RESULT_PATH = "result.png"
INITIAL_HUMAN_STATE = np.array([-7.0, 0.0])  # Same y would cause local minima
INITIAL_ROBOT_STATE = np.array([-5.0, 0.1])
TIME_STEPS = 10
INITIAL_ACTION = np.array([0.0, 0.0])  # (theta, speed)
ANGLE_BOUNDS = (-np.pi, np.pi)
SPEED_BOUNDS = (0.0, 1.0)


def notify_completion():
    try:
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("\a", end="", flush=True)

atexit.register(notify_completion) # Play sound when exit to notify developer

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


def build_bounds(time_steps):
    bounds = []
    for _ in range(time_steps):
        bounds.append(ANGLE_BOUNDS) # theta bounds: how agent can turn (default: -pi to pi)
        bounds.append(SPEED_BOUNDS) # speed bounds: how fast agent can go (default: 0 to 1)
    return bounds

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

        # If collide, add big cost
        if np.linalg.norm(human_state - robot_state) < AGENT_RADIUS * 2: # If distance less than sum of radii, they collide
            cost += 1000
            break
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

        # If collide, add big reward
        # if np.linalg.norm(human_state - robot_state) < AGENT_RADIUS * 2:
        #     cost -= 1000
        #     break
    return cost

def solve_human_best_response(initial_human_state, initial_robot_state, robot_actions, human_actions_guess):
    return minimize(
            fun=human_cost, 
            x0=human_actions_guess, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='Powell',
            bounds=bounds
            )

def solve_robot_best_response(initial_human_state, initial_robot_state, human_actions, robot_actions_guess):
    return minimize(
            fun=robot_cost,
            x0=robot_actions_guess,
            args=(initial_human_state, initial_robot_state, human_actions),
            method='Powell',
            bounds=bounds
            )

# Determine the human's cost for a given robot best response
def nested_cost(human_actions, initial_human_state, initial_robot_state, robot_actions):
    # For each candidate human plan, solve the robot's best response
    result = solve_robot_best_response(initial_human_state, initial_robot_state, human_actions, robot_actions)
    robot_actions = result.x
    return human_cost(human_actions, initial_human_state, initial_robot_state, robot_actions)

# plot the trajectories
def plot_trajectory(initial_human_state, initial_robot_state, human_actions, robot_actions):
    human_trajectory = rollout(initial_human_state, human_actions)
    robot_trajectory = rollout(initial_robot_state, robot_actions)
    fig, ax = plt.subplots()
    ax.plot(human_trajectory[:,0], human_trajectory[:,1], 'b-', alpha=0.6, linewidth=2)
    ax.plot(robot_trajectory[:,0], robot_trajectory[:,1], 'r-', alpha=0.6, linewidth=2)

    num_steps = len(human_trajectory)
    for i, (human_state, robot_state) in enumerate(zip(human_trajectory, robot_trajectory)):
        alpha = 0.12 + 0.55 * (i + 1) / num_steps
        ax.add_patch(Circle(human_state, AGENT_RADIUS, facecolor='blue', edgecolor='blue', alpha=alpha, linewidth=1.5))
        ax.add_patch(Circle(robot_state, AGENT_RADIUS, facecolor='red', edgecolor='red', alpha=alpha, linewidth=1.5))
        ax.scatter(human_state[0], human_state[1], color='blue', s=18, alpha=min(alpha + 0.15, 1.0), zorder=3)
        ax.scatter(robot_state[0], robot_state[1], color='red', s=18, alpha=min(alpha + 0.15, 1.0), zorder=3)
    ax.axis('equal')
    # Add human and robot labels
    ax.text(initial_human_state[0], initial_human_state[1] + AGENT_RADIUS + 0.1, 'Human', fontsize=12, color='blue')
    ax.text(initial_robot_state[0], initial_robot_state[1] + AGENT_RADIUS + 0.1, 'Robot', fontsize=12, color='red')
    # Add time labels
    for i in range(len(human_trajectory)):
        ax.text(human_trajectory[i,0], human_trajectory[i,1] - AGENT_RADIUS - 0.1, f't={i}', fontsize=8, color='black')
        ax.text(robot_trajectory[i,0], robot_trajectory[i,1] - AGENT_RADIUS - 0.1, f't={i}', fontsize=8, color='black')
    fig.savefig(RESULT_PATH)
    plt.close(fig)

# initialize the vehicles
initial_human_state = INITIAL_HUMAN_STATE.copy()
initial_robot_state = INITIAL_ROBOT_STATE.copy()

robot_actions = np.tile(INITIAL_ACTION, TIME_STEPS) # actions are slopes (turning angle)
human_actions = np.tile(INITIAL_ACTION, TIME_STEPS)

bounds = build_bounds(TIME_STEPS)

# Optimize the human's actions given the robot's best response
result = minimize(
            fun=nested_cost, 
            x0=human_actions, 
            args=(initial_human_state, initial_robot_state, robot_actions),
            method='Powell', # Switch to Powell optimizer because it can handle non-smooth functions better
            bounds=bounds
            )
human_actions = result.x

# Solve for the robot's best response to the final human plan.
result = solve_robot_best_response(initial_human_state, initial_robot_state, human_actions, robot_actions)
robot_actions = result.x

# Remove the png to ensure we're viewing an updated result
if os.path.exists(RESULT_PATH):
    os.remove(RESULT_PATH)
# plot the results (saves as a png file)
plot_trajectory(initial_human_state, initial_robot_state, unpack_actions(human_actions), unpack_actions(robot_actions))