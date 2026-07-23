# Deep Q-Network (DQN) Sample

This sample demonstrates how to implement and train a Deep Q-Network (DQN) agent using AiDotNet's reinforcement learning APIs. DQN was a breakthrough algorithm from DeepMind (2015) that combined Q-learning with deep neural networks, enabling RL to scale to high-dimensional state spaces.

## Overview

The sample trains a DQN agent to navigate a GridWorld environment, learning to find the optimal path from a start position to a goal position.

### Key DQN Components Demonstrated

1. **Experience Replay Buffer**: Stores past experiences and samples random batches for training
   - Breaks temporal correlations in training data
   - Improves sample efficiency
   - Stabilizes learning

2. **Target Network**: A slowly-updating copy of the Q-network
   - Provides stable Q-value targets
   - Reduces oscillations during training
   - Updated every N steps (hard update) or continuously (soft update)

3. **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
   - Starts with high exploration (epsilon = 1.0)
   - Gradually decreases to favor exploitation (epsilon -> 0.01)
   - Ensures the agent discovers good strategies

4. **Q-Value Learning**: Neural network learns to estimate action values
   - Q(s, a) = expected cumulative reward from state s taking action a
   - Uses Bellman equation for temporal difference learning

## Environment: GridWorld

A simple 5x5 grid navigation task:

```
+---------------+
| S |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   |   |
+---+---+---+---+---+
|   |   |   |   | G |
+---------------+
```

- **S**: Start position (0, 0)
- **G**: Goal position (4, 4)
- **Actions**: Up, Down, Left, Right
- **State**: One-hot encoded position (25 dimensions)
- **Rewards**: +10 for reaching goal, -0.1 per step, -1 for timeout

## Running the Sample

```bash
cd samples/reinforcement-learning/DeepQLearning
dotnet run
```

## Expected Output

```
=== AiDotNet Deep Q-Network (DQN) Sample ===
Training a DQN agent on a GridWorld environment

Environment: GridWorld (5x5)
  State space: 25 (one-hot encoded position)
  Action space: 4 (up, down, left, right)
  Goal: Navigate from start to goal position

Creating DQN agent with experience replay and target network...
  Hidden layers: [64, 64]
  Learning rate: 0.001
  Discount factor (gamma): 0.99
  Epsilon: 1.0 -> 0.01 (decay: 0.995)
  Replay buffer size: 10000
  Batch size: 32
  Target update frequency: 100 steps

Training DQN Agent...

Episode | Steps | Reward | Epsilon | Avg Q-Value | Success Rate (100)
---------------------------------------------------------------------
      0 |   100 |  -11.0 |   1.000 |       0.000 |              0.0%
     50 |    87 |   -8.6 |   0.778 |       0.123 |             12.0%
    100 |    42 |    5.9 |   0.605 |       0.456 |             35.0%
    150 |    23 |    7.8 |   0.471 |       1.234 |             62.0%
    200 |    14 |    8.6 |   0.367 |       2.567 |             78.0%
    250 |    11 |    9.0 |   0.285 |       3.891 |             88.0%
    300 |     9 |    9.2 |   0.222 |       5.234 |             94.0%

Environment solved at episode 312! (Success rate: 91.0%)

--- Q-Value Learning Curve ---
  Avg Q-Value (smoothed over 10 episodes):
  Max: 5.678, Min: 0.000
  |                                   ##########|
  |                              ###############|
  |                         ####################|
  |                    #########################|
  |               ##############################|
  |          ###################################|
  |     ########################################|
  |#############################################|
  +---------------------------------------------+

--- Testing Trained Agent ---
Running 10 test episodes with deterministic policy...

  Test  1:   9 steps, Reward:    9.2 [SUCCESS]
  Test  2:  10 steps, Reward:    9.1 [SUCCESS]
  Test  3:   9 steps, Reward:    9.2 [SUCCESS]
  Test  4:   8 steps, Reward:    9.3 [SUCCESS]
  Test  5:   9 steps, Reward:    9.2 [SUCCESS]
  Test  6:   9 steps, Reward:    9.2 [SUCCESS]
  Test  7:  11 steps, Reward:    9.0 [SUCCESS]
  Test  8:   8 steps, Reward:    9.3 [SUCCESS]
  Test  9:   9 steps, Reward:    9.2 [SUCCESS]
  Test 10:   9 steps, Reward:    9.2 [SUCCESS]

--- Summary Statistics ---
  Test success rate: 100%
  Average steps to goal: 9.1
  Average reward: 9.19

--- Learned Policy Visualization ---

  Grid (5x5):
  ----------------
  | S | > | > | v | v |
  | v | > | v | v | v |
  | v | > | > | v | v |
  | v | > | > | > | v |
  | > | > | > | > | G |
  ----------------
  S = Start, G = Goal, ^v<> = Learned Policy Direction

=== Sample Complete ===
```

## Code Structure

### Main Components

1. **DQNOptions**: Configuration for the DQN agent
   - State/action sizes
   - Network architecture (hidden layers)
   - Learning hyperparameters
   - Exploration parameters

2. **DQNAgent**: The learning agent
   - Neural network Q-function
   - Experience replay buffer
   - Target network for stability
   - Epsilon-greedy action selection

3. **GridWorldEnvironment**: The training environment
   - State representation
   - Reward function
   - Episode termination

### Training Loop

```csharp
// For each episode
var state = env.Reset();

while (!done)
{
    // 1. Select action (epsilon-greedy)
    var action = agent.SelectAction(state, training: true);

    // 2. Take action in environment
    var (nextState, reward, done) = env.Step(action);

    // 3. Store experience in replay buffer
    agent.StoreExperience(state, action, reward, nextState, done);

    // 4. Train on batch from replay buffer
    agent.Train();

    state = nextState;
}
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.001 | Step size for gradient descent |
| Discount Factor | 0.99 | Future reward weighting |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Final exploration rate |
| Epsilon Decay | 0.995 | Exploration decay rate |
| Batch Size | 32 | Experiences per training step |
| Replay Buffer | 10000 | Maximum stored experiences |
| Target Update | 100 | Steps between target network updates |
| Warmup Steps | 500 | Steps before training begins |

## Algorithm: Deep Q-Learning

The DQN algorithm learns the optimal action-value function Q*(s, a):

1. **Initialize** Q-network and target network with random weights
2. **For each step**:
   - Select action using epsilon-greedy policy
   - Execute action and observe reward and next state
   - Store transition in replay buffer
   - Sample random batch from replay buffer
   - Compute target: `y = r + gamma * max(Q_target(s', a'))`
   - Update Q-network by minimizing `(y - Q(s, a))^2`
3. **Periodically** update target network weights

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
2. van Hasselt, H., et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
3. Schaul, T., et al. (2015). "Prioritized Experience Replay." ICLR.

## Next Steps

After understanding this basic DQN implementation, explore:

- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separates state value and action advantage
- **Prioritized Experience Replay**: Samples important transitions more often
- **Rainbow DQN**: Combines multiple DQN improvements

See the `AiDotNet.ReinforcementLearning.Agents` namespace for advanced implementations.
