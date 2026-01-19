# CartPole - Reinforcement Learning with PPO

This sample demonstrates training a PPO (Proximal Policy Optimization) agent to balance a pole on a cart.

## What You'll Learn

- How to configure reinforcement learning with `PredictionModelBuilder`
- How to implement the CartPole environment
- How to train a PPO agent
- How to evaluate agent performance

## The CartPole Problem

The agent controls a cart that can move left or right. A pole is attached to the cart by a joint. The goal is to prevent the pole from falling over by moving the cart.

**State Space** (4 dimensions):
- Cart position
- Cart velocity
- Pole angle
- Pole angular velocity

**Action Space** (2 actions):
- Move left (0)
- Move right (1)

**Reward**:
- +1 for each timestep the pole remains upright
- Episode ends when pole angle > 12° or cart position > 2.4

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet CartPole RL ===
Training PPO agent to balance a pole

Environment: CartPole-v1
  State space: 4 dimensions
  Action space: 2 actions

Training PPO agent...
  Episode 10: Reward = 23.4, Avg = 18.2
  Episode 20: Reward = 45.6, Avg = 31.5
  Episode 50: Reward = 89.2, Avg = 67.3
  Episode 100: Reward = 195.0, Avg = 156.8
  Episode 200: Reward = 500.0, Avg = 423.5 (Solved!)

Training complete!
  Best reward: 500.0
  Solved in 142 episodes

Testing trained agent...
  Test 1: 487 steps
  Test 2: 500 steps (max)
  Test 3: 492 steps
  Average: 493 steps
```

## Code Highlights

```csharp
var agent = new PPOAgent<double>(
    stateSize: 4,
    actionSize: 2,
    hiddenSize: 64,
    learningRate: 3e-4,
    gamma: 0.99,
    epsilon: 0.2);

// Training loop
for (int episode = 0; episode < maxEpisodes; episode++)
{
    var state = env.Reset();
    double totalReward = 0;

    while (!env.IsDone)
    {
        var action = agent.SelectAction(state);
        var (nextState, reward, done) = env.Step(action);
        agent.Store(state, action, reward, nextState, done);
        state = nextState;
        totalReward += reward;
    }

    agent.Train();
}
```

## PPO Algorithm Overview

PPO is a policy gradient method that uses:
- **Clipped objective**: Prevents large policy updates
- **Value function**: Estimates expected returns
- **Advantage estimation**: Reduces variance in gradient estimates

```
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 3e-4 | Adam optimizer step size |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Epsilon (ε) | 0.2 | Clipping parameter |
| Hidden size | 64 | Neural network hidden layer size |
| Batch size | 64 | Mini-batch size for updates |

## Next Steps

- [DeepQLearning](../DeepQLearning/) - Value-based RL with DQN
- Explore other agents: SAC, DDPG, A3C
