# Lunar Lander Q-Learning with TensorFlow

This project demonstrates the application of Q-learning to solve the **LunarLander-v2** environment using deep Q-networks (DQNs) implemented with **TensorFlow**. The goal is to train an agent to land a lunar lander safely by maximizing rewards through reinforcement learning.

## Installation

To run this project, you'll need the following Python libraries:

- `gym`: OpenAI's toolkit for developing and comparing reinforcement learning algorithms.
- `numpy`: For numerical operations.
- `PIL` (Python Imaging Library): For image rendering.
- `tensorflow`: For building and training the deep Q-network.
- `pyvirtualdisplay`: To run the environment in a headless mode (useful when working with environments that require graphical output).
- `utils`: Custom utility functions used in this project.

You can install the required dependencies using `pip`:

```bash
pip install gym numpy tensorflow pillow pyvirtualdisplay
```

## Project Overview

This project implements a deep Q-learning agent that interacts with the **LunarLander-v2** environment. The agent uses a **Q-network** (a neural network model) to estimate the action-value function, which is updated during training based on experience replay and a target network.

### Key Components:
1. **Q-Network**: The neural network used to predict Q-values for state-action pairs.
2. **Experience Replay**: A memory buffer that stores experiences of the agent, which are later used to train the Q-network.
3. **Target Network**: A copy of the Q-network used to compute the target values for the loss function.
4. **Epsilon-Greedy Policy**: A policy where the agent explores with probability `epsilon` and exploits with probability `1 - epsilon`.

### Hyperparameters:
- **MEMORY_SIZE**: Size of the experience replay buffer.
- **GAMMA**: Discount factor used in the Q-learning update.
- **ALPHA**: Learning rate for the optimizer.
- **NUM_STEPS_FOR_UPDATE**: Number of steps to wait before updating the Q-network.

### Environment:
- **LunarLander-v2**: The environment where the agent learns to control a lander to land on a flat surface.

### Training Process:
- **Episodes**: The agent interacts with the environment for a defined number of episodes.
- **Action Selection**: In each step of an episode, the agent chooses an action based on the current state using the epsilon-greedy policy.
- **Network Update**: The Q-network is updated every `NUM_STEPS_FOR_UPDATE` steps based on the experiences stored in the memory buffer.
- **Target Network**: The target network is updated periodically with the Q-network weights.
  
## Running the Code

### Setup the environment:

The following code sets up the **LunarLander-v2** environment:

```python
import gym
env = gym.make('LunarLander-v2')
env.reset()
```

### Initialize Q-network and Target Q-network:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])
```

### Training loop:
The agent interacts with the environment, collects experiences, and periodically updates the Q-network.

```python
for i in range(num_episodes):
    state = env.reset()
    total_points = 0
    for t in range(max_num_timesteps):
        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        action = utils.get_action(q_values, epsilon)
        
        next_state, reward, done, _ = env.step(action)
        memory_buffer.append(experience(state, action, reward, next_state, done))

        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
        if update:
            experiences = utils.get_experiences(memory_buffer)
            agent_learn(experiences, GAMMA)

        state = next_state
        total_points += reward

        if done:
            break
```

### Saving the Model:
Once the agent has learned to perform well in the environment, the trained model is saved:

```python
if av_latest_points >= 200.0:
    print(f"\n\nEnvironment solved in {i+1} episodes!")
    q_network.save('lunar_lander_model.h5')
    break
```

## Features

- **Experience Replay**: Stores and samples experiences to break correlation and stabilize training.
- **Target Network**: Reduces the variance in the training process.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation.
- **Model Saving**: Saves the trained model once the environment is solved.
- **Video Generation**: After training, a video of the agent's performance is generated and embedded.

## Utilities

- **`utils.py`**: Contains utility functions for updating the target network, selecting actions based on the epsilon-greedy policy, and visualizing the training progress.
  
## Evaluation

To evaluate the performance of the agent, the code generates a video of the agent's actions in the environment:

```python
filename = "./videos/lunar_lander.mp4"
utils.create_video(filename, env, q_network)
utils.embed_mp4(filename)
```

## Conclusion

This project demonstrates a complete Q-learning setup using TensorFlow for the **LunarLander-v2** environment. By using a deep neural network for Q-value approximation, experience replay, and a target network, the agent is able to learn to land the lunar lander safely and achieve a high score.

