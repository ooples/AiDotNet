# Issue #401: Reinforcement Learning (DQN, PPO, A3C) - Junior Developer Implementation Guide

## Overview

Reinforcement Learning (RL) teaches agents to make decisions through trial and error, learning from rewards and penalties. Unlike supervised learning, RL has no labeled data - agents learn optimal behavior by interacting with an environment.

This guide covers three powerful RL algorithms:
- **DQN (Deep Q-Network)**: Value-based method using experience replay
- **PPO (Proximal Policy Optimization)**: Policy gradient with stability constraints
- **A3C (Asynchronous Advantage Actor-Critic)**: Distributed actor-critic learning

**Learning Value**: Understanding sequential decision-making, exploration vs exploitation, credit assignment, and policy optimization.

**Estimated Complexity**: Advanced (25-35 hours)

**Prerequisites**:
- Neural networks and backpropagation
- Understanding of Markov Decision Processes (MDPs)
- Basic probability and expected value
- Gradient descent and optimization

---

## Educational Objectives

By implementing RL algorithms, you will learn:

1. **Markov Decision Processes**: States, actions, rewards, transitions
2. **Value Functions**: Q-values, state values, advantage functions
3. **Policy Optimization**: Learning action selection strategies
4. **Exploration Strategies**: Epsilon-greedy, entropy regularization
5. **Experience Replay**: Breaking correlation in sequential data
6. **Target Networks**: Stabilizing Q-learning
7. **Actor-Critic Methods**: Combining value and policy learning
8. **Distributed Training**: Parallel experience collection

---

## Reinforcement Learning Background

### The RL Problem

An agent interacts with an environment:
1. Observes **state** `s_t`
2. Takes **action** `a_t`
3. Receives **reward** `r_t` and next state `s_{t+1}`
4. Goal: Maximize cumulative reward `Σ γ^t r_t`

Where γ (gamma) is a discount factor (0 < γ < 1).

### Key Concepts

**Q-Function**: Expected cumulative reward from state-action pair
```
Q(s, a) = E[r_t + γ r_{t+1} + γ² r_{t+2} + ... | s_t=s, a_t=a]
```

**Policy**: Mapping from states to actions
```
π(a|s) = probability of taking action a in state s
```

**Bellman Equation**: Recursive relationship
```
Q(s, a) = r + γ max_a' Q(s', a')
```

---

## Architecture Design

### Core Interfaces

```csharp
namespace AiDotNet.ReinforcementLearning
{
    /// <summary>
    /// Represents an environment that an agent can interact with.
    /// </summary>
    /// <typeparam name="TState">State representation type</typeparam>
    /// <typeparam name="TAction">Action type</typeparam>
    public interface IEnvironment<TState, TAction>
    {
        /// <summary>
        /// Resets environment to initial state.
        /// </summary>
        TState Reset();

        /// <summary>
        /// Executes action and returns (next_state, reward, done).
        /// </summary>
        (TState nextState, double reward, bool done) Step(TAction action);

        /// <summary>
        /// Gets available actions in current state.
        /// </summary>
        List<TAction> GetLegalActions(TState state);

        /// <summary>
        /// Gets state dimension (for neural network input).
        /// </summary>
        int StateDimension { get; }

        /// <summary>
        /// Gets number of possible actions.
        /// </summary>
        int ActionCount { get; }
    }

    /// <summary>
    /// Base interface for RL agents.
    /// </summary>
    /// <typeparam name="TState">State type</typeparam>
    /// <typeparam name="TAction">Action type</typeparam>
    public interface IRLAgent<TState, TAction>
    {
        /// <summary>
        /// Selects action given current state.
        /// </summary>
        TAction SelectAction(TState state, bool training = true);

        /// <summary>
        /// Updates agent from experience.
        /// </summary>
        void Update(Experience<TState, TAction> experience);

        /// <summary>
        /// Trains agent on a batch of experiences.
        /// </summary>
        double Train(List<Experience<TState, TAction>> batch);
    }

    /// <summary>
    /// Represents a single experience tuple (s, a, r, s', done).
    /// </summary>
    public class Experience<TState, TAction>
    {
        public TState State { get; set; }
        public TAction Action { get; set; }
        public double Reward { get; set; }
        public TState NextState { get; set; }
        public bool Done { get; set; }
    }
}
```

### Experience Replay Buffer

```csharp
namespace AiDotNet.ReinforcementLearning.Memory
{
    /// <summary>
    /// Stores and samples experiences for training.
    /// Breaks correlation between sequential experiences.
    /// </summary>
    public interface IReplayBuffer<TState, TAction>
    {
        /// <summary>
        /// Adds experience to buffer.
        /// </summary>
        void Add(Experience<TState, TAction> experience);

        /// <summary>
        /// Samples random batch for training.
        /// </summary>
        List<Experience<TState, TAction>> Sample(int batchSize);

        /// <summary>
        /// Number of experiences stored.
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Maximum buffer capacity.
        /// </summary>
        int Capacity { get; }
    }

    public class ReplayBuffer<TState, TAction> : IReplayBuffer<TState, TAction>
    {
        private readonly Queue<Experience<TState, TAction>> _buffer;
        private readonly int _capacity;
        private readonly Random _random;

        public ReplayBuffer(int capacity)
        {
            _capacity = capacity;
            _buffer = new Queue<Experience<TState, TAction>>(capacity);
            _random = new Random();
        }

        public void Add(Experience<TState, TAction> experience)
        {
            if (_buffer.Count >= _capacity)
            {
                _buffer.Dequeue(); // Remove oldest
            }
            _buffer.Enqueue(experience);
        }

        public List<Experience<TState, TAction>> Sample(int batchSize)
        {
            if (_buffer.Count < batchSize)
            {
                throw new InvalidOperationException(
                    $"Buffer has {_buffer.Count} experiences, need {batchSize}");
            }

            // Random sampling without replacement
            var indices = Enumerable.Range(0, _buffer.Count)
                .OrderBy(_ => _random.Next())
                .Take(batchSize)
                .ToList();

            var bufferArray = _buffer.ToArray();
            return indices.Select(i => bufferArray[i]).ToList();
        }

        public int Size => _buffer.Count;
        public int Capacity => _capacity;
    }
}
```

---

## Algorithm 1: Deep Q-Network (DQN)

### Theory

DQN approximates the Q-function using a neural network:
```
Q(s, a; θ) ≈ Q*(s, a)
```

**Loss Function** (Bellman error):
```
L(θ) = E[(r + γ max_a' Q(s', a'; θ^-) - Q(s, a; θ))²]
```

Where θ^- are "target network" parameters (updated periodically).

### Key Innovations

1. **Experience Replay**: Store experiences, sample randomly for training
2. **Target Network**: Separate network for computing targets, updated slowly
3. **Epsilon-Greedy**: Explore with probability ε, exploit otherwise

### Implementation

**File**: `src/ReinforcementLearning/Algorithms/DQN/DQNAgent.cs`

```csharp
public class DQNAgent<TState, TAction> : IRLAgent<TState, TAction>
    where TState : class
{
    private readonly IQNetwork<TState> _qNetwork;       // Main Q-network
    private readonly IQNetwork<TState> _targetNetwork;  // Target network
    private readonly IReplayBuffer<TState, TAction> _replayBuffer;
    private readonly IOptimizer<double> _optimizer;

    // Hyperparameters
    private readonly double _gamma;              // Discount factor
    private readonly double _epsilonStart;       // Initial exploration
    private readonly double _epsilonEnd;         // Final exploration
    private readonly double _epsilonDecay;       // Decay rate
    private readonly int _targetUpdateFreq;      // Target network update frequency
    private readonly int _batchSize;

    private double _epsilon;
    private int _stepCount;
    private readonly Random _random;

    public DQNAgent(
        IQNetwork<TState> qNetwork,
        double gamma = 0.99,
        double epsilonStart = 1.0,
        double epsilonEnd = 0.01,
        double epsilonDecay = 0.995,
        int targetUpdateFreq = 1000,
        int batchSize = 64,
        int bufferCapacity = 100000)
    {
        _qNetwork = qNetwork;
        _targetNetwork = qNetwork.Clone(); // Copy architecture and initial weights
        _replayBuffer = new ReplayBuffer<TState, TAction>(bufferCapacity);
        _optimizer = new AdamOptimizer<double>(learningRate: 0.0001);

        _gamma = gamma;
        _epsilonStart = epsilonStart;
        _epsilonEnd = epsilonEnd;
        _epsilonDecay = epsilonDecay;
        _targetUpdateFreq = targetUpdateFreq;
        _batchSize = batchSize;

        _epsilon = _epsilonStart;
        _stepCount = 0;
        _random = new Random();
    }

    public TAction SelectAction(TState state, bool training = true)
    {
        // Epsilon-greedy exploration
        if (training && _random.NextDouble() < _epsilon)
        {
            // Explore: random action
            return GetRandomAction();
        }
        else
        {
            // Exploit: best action according to Q-network
            var qValues = _qNetwork.PredictQValues(state);
            return GetActionFromIndex(qValues.ArgMax());
        }
    }

    public void Update(Experience<TState, TAction> experience)
    {
        // Add to replay buffer
        _replayBuffer.Add(experience);

        // Train if enough experiences
        if (_replayBuffer.Size >= _batchSize)
        {
            var batch = _replayBuffer.Sample(_batchSize);
            Train(batch);
        }

        _stepCount++;

        // Update target network periodically
        if (_stepCount % _targetUpdateFreq == 0)
        {
            UpdateTargetNetwork();
        }

        // Decay epsilon
        _epsilon = Math.Max(_epsilonEnd, _epsilon * _epsilonDecay);
    }

    public double Train(List<Experience<TState, TAction>> batch)
    {
        // Prepare batch data
        var states = batch.Select(e => e.State).ToList();
        var actions = batch.Select(e => GetActionIndex(e.Action)).ToList();
        var rewards = batch.Select(e => e.Reward).ToArray();
        var nextStates = batch.Select(e => e.NextState).ToList();
        var dones = batch.Select(e => e.Done).ToArray();

        // Compute current Q-values: Q(s, a)
        var currentQs = _qNetwork.PredictBatch(states);

        // Compute target Q-values using target network
        var nextQs = _targetNetwork.PredictBatch(nextStates);

        var targets = new Matrix<double>(batch.Count, _qNetwork.ActionCount);

        for (int i = 0; i < batch.Count; i++)
        {
            // Copy current Q-values
            for (int a = 0; a < _qNetwork.ActionCount; a++)
            {
                targets[i, a] = currentQs[i, a];
            }

            // Update Q-value for taken action
            var actionIdx = actions[i];
            if (dones[i])
            {
                // Terminal state: Q(s, a) = r
                targets[i, actionIdx] = rewards[i];
            }
            else
            {
                // Bellman update: Q(s, a) = r + γ max_a' Q(s', a')
                var maxNextQ = nextQs.Row(i).Max();
                targets[i, actionIdx] = rewards[i] + _gamma * maxNextQ;
            }
        }

        // Train Q-network to match targets
        var loss = _qNetwork.Train(states, targets, _optimizer);

        return loss;
    }

    private void UpdateTargetNetwork()
    {
        // Copy weights from main network to target network
        _targetNetwork.CopyWeightsFrom(_qNetwork);
    }

    private TAction GetRandomAction()
    {
        var actionIdx = _random.Next(_qNetwork.ActionCount);
        return GetActionFromIndex(actionIdx);
    }

    private int GetActionIndex(TAction action)
    {
        // Convert action to integer index (implementation depends on TAction)
        if (action is int idx) return idx;
        throw new NotImplementedException("Action conversion needed");
    }

    private TAction GetActionFromIndex(int index)
    {
        // Convert integer index to action (implementation depends on TAction)
        if (typeof(TAction) == typeof(int)) return (TAction)(object)index;
        throw new NotImplementedException("Action conversion needed");
    }
}
```

### Q-Network Architecture

**File**: `src/ReinforcementLearning/Networks/QNetwork.cs`

```csharp
public interface IQNetwork<TState>
{
    /// <summary>
    /// Predicts Q-values for all actions given a state.
    /// </summary>
    Vector<double> PredictQValues(TState state);

    /// <summary>
    /// Predicts Q-values for batch of states.
    /// </summary>
    Matrix<double> PredictBatch(List<TState> states);

    /// <summary>
    /// Trains network on batch.
    /// </summary>
    double Train(List<TState> states, Matrix<double> targets, IOptimizer<double> optimizer);

    /// <summary>
    /// Creates copy of network with same architecture.
    /// </summary>
    IQNetwork<TState> Clone();

    /// <summary>
    /// Copies weights from another network.
    /// </summary>
    void CopyWeightsFrom(IQNetwork<TState> source);

    int ActionCount { get; }
}

public class DenseQNetwork : IQNetwork<Vector<double>>
{
    private readonly List<ILayer<double>> _layers;
    private readonly int _stateDim;
    private readonly int _actionCount;

    public DenseQNetwork(int stateDim, int actionCount, List<int> hiddenSizes)
    {
        _stateDim = stateDim;
        _actionCount = actionCount;
        _layers = new List<ILayer<double>>();

        // Build network: state -> hidden layers -> Q-values for each action
        var sizes = new List<int> { stateDim };
        sizes.AddRange(hiddenSizes);
        sizes.Add(actionCount);

        for (int i = 0; i < sizes.Count - 1; i++)
        {
            _layers.Add(new DenseLayer<double>(sizes[i], sizes[i + 1]));
            if (i < sizes.Count - 2)
            {
                _layers.Add(new ReLUActivation<double>());
            }
        }
    }

    public Vector<double> PredictQValues(Vector<double> state)
    {
        var batch = new Matrix<double>(1, _stateDim);
        for (int i = 0; i < _stateDim; i++)
        {
            batch[0, i] = state[i];
        }

        var output = Forward(batch);
        return output.Row(0);
    }

    public Matrix<double> PredictBatch(List<Vector<double>> states)
    {
        var batch = new Matrix<double>(states.Count, _stateDim);
        for (int i = 0; i < states.Count; i++)
        {
            for (int j = 0; j < _stateDim; j++)
            {
                batch[i, j] = states[i][j];
            }
        }

        return Forward(batch);
    }

    private Matrix<double> Forward(Matrix<double> input)
    {
        var output = input;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    public double Train(List<Vector<double>> states, Matrix<double> targets, IOptimizer<double> optimizer)
    {
        var batch = PredictBatch(states);

        // Compute MSE loss
        var loss = 0.0;
        for (int i = 0; i < batch.Rows; i++)
        {
            for (int j = 0; j < batch.Columns; j++)
            {
                var error = targets[i, j] - batch[i, j];
                loss += error * error;
            }
        }
        loss /= (batch.Rows * batch.Columns);

        // Backpropagation
        var gradient = ComputeGradient(batch, targets);
        optimizer.Step(_layers, gradient);

        return loss;
    }

    public int ActionCount => _actionCount;

    // Clone and copy methods omitted for brevity
}
```

### Training Loop

**File**: `src/ReinforcementLearning/Training/DQNTrainer.cs`

```csharp
public class DQNTrainer<TState, TAction>
{
    private readonly DQNAgent<TState, TAction> _agent;
    private readonly IEnvironment<TState, TAction> _environment;

    public DQNTrainer(DQNAgent<TState, TAction> agent, IEnvironment<TState, TAction> environment)
    {
        _agent = agent;
        _environment = environment;
    }

    public TrainingResults Train(int numEpisodes, int maxStepsPerEpisode = 1000)
    {
        var episodeRewards = new List<double>();

        for (int episode = 0; episode < numEpisodes; episode++)
        {
            var state = _environment.Reset();
            var totalReward = 0.0;

            for (int step = 0; step < maxStepsPerEpisode; step++)
            {
                // Select and execute action
                var action = _agent.SelectAction(state, training: true);
                var (nextState, reward, done) = _environment.Step(action);

                // Store experience
                var experience = new Experience<TState, TAction>
                {
                    State = state,
                    Action = action,
                    Reward = reward,
                    NextState = nextState,
                    Done = done
                };

                _agent.Update(experience);

                totalReward += reward;
                state = nextState;

                if (done) break;
            }

            episodeRewards.Add(totalReward);

            if (episode % 10 == 0)
            {
                var avgReward = episodeRewards.Skip(Math.Max(0, episode - 100)).Average();
                Console.WriteLine($"Episode {episode}: Reward = {totalReward:F2}, Avg100 = {avgReward:F2}");
            }
        }

        return new TrainingResults { EpisodeRewards = episodeRewards };
    }
}
```

---

## Algorithm 2: Proximal Policy Optimization (PPO)

### Theory

PPO is a policy gradient method that directly learns a policy π(a|s).

**Objective**: Maximize expected return
```
J(θ) = E[Σ r_t]
```

**Policy Gradient**:
```
∇J(θ) = E[∇ log π(a|s) * A(s,a)]
```

Where A(s,a) is the advantage function: `A(s,a) = Q(s,a) - V(s)`

**PPO Clipped Objective**:
```
L(θ) = E[min(r(θ) A, clip(r(θ), 1-ε, 1+ε) A)]
```

Where `r(θ) = π(a|s; θ) / π(a|s; θ_old)` is the probability ratio.

**Key Idea**: Limit policy updates to prevent catastrophic performance drops.

### Implementation

**File**: `src/ReinforcementLearning/Algorithms/PPO/PPOAgent.cs`

```csharp
public class PPOAgent<TState> : IRLAgent<TState, int>
{
    private readonly IPolicyNetwork<TState> _policyNetwork;  // π(a|s)
    private readonly IValueNetwork<TState> _valueNetwork;    // V(s)
    private readonly IOptimizer<double> _policyOptimizer;
    private readonly IOptimizer<double> _valueOptimizer;

    private readonly double _gamma;           // Discount factor
    private readonly double _lambda;          // GAE parameter
    private readonly double _clipEpsilon;     // PPO clipping parameter
    private readonly double _entropyCoef;     // Entropy bonus coefficient
    private readonly int _updateEpochs;       // Number of optimization epochs per update

    private List<Experience<TState, int>> _trajectoryBuffer;

    public PPOAgent(
        IPolicyNetwork<TState> policyNetwork,
        IValueNetwork<TState> valueNetwork,
        double gamma = 0.99,
        double lambda = 0.95,
        double clipEpsilon = 0.2,
        double entropyCoef = 0.01,
        int updateEpochs = 10)
    {
        _policyNetwork = policyNetwork;
        _valueNetwork = valueNetwork;
        _policyOptimizer = new AdamOptimizer<double>(learningRate: 0.0003);
        _valueOptimizer = new AdamOptimizer<double>(learningRate: 0.001);

        _gamma = gamma;
        _lambda = lambda;
        _clipEpsilon = clipEpsilon;
        _entropyCoef = entropyCoef;
        _updateEpochs = updateEpochs;

        _trajectoryBuffer = new List<Experience<TState, int>>();
    }

    public int SelectAction(TState state, bool training = true)
    {
        // Sample from policy distribution
        var actionProbs = _policyNetwork.PredictActionProbabilities(state);

        if (training)
        {
            // Sample stochastically during training
            return SampleAction(actionProbs);
        }
        else
        {
            // Use most likely action during evaluation
            return actionProbs.ArgMax();
        }
    }

    public void Update(Experience<TState, int> experience)
    {
        // Collect experiences into trajectory buffer
        _trajectoryBuffer.Add(experience);

        // Update when trajectory is complete (episode ends)
        if (experience.Done)
        {
            var loss = Train(_trajectoryBuffer);
            _trajectoryBuffer.Clear();
        }
    }

    public double Train(List<Experience<TState, int>> trajectory)
    {
        // 1. Compute advantages using Generalized Advantage Estimation (GAE)
        var advantages = ComputeGAE(trajectory);
        var returns = ComputeReturns(trajectory);

        // 2. Get old policy probabilities (before update)
        var oldLogProbs = trajectory.Select(e =>
            Math.Log(_policyNetwork.PredictActionProbabilities(e.State)[e.Action])
        ).ToArray();

        var totalLoss = 0.0;

        // 3. Multiple epochs of optimization
        for (int epoch = 0; epoch < _updateEpochs; epoch++)
        {
            for (int i = 0; i < trajectory.Count; i++)
            {
                var state = trajectory[i].State;
                var action = trajectory[i].Action;
                var advantage = advantages[i];
                var returnValue = returns[i];
                var oldLogProb = oldLogProbs[i];

                // Policy loss (PPO clipped objective)
                var actionProbs = _policyNetwork.PredictActionProbabilities(state);
                var newLogProb = Math.Log(actionProbs[action]);
                var ratio = Math.Exp(newLogProb - oldLogProb);

                var clippedRatio = Clamp(ratio, 1.0 - _clipEpsilon, 1.0 + _clipEpsilon);
                var policyLoss = -Math.Min(ratio * advantage, clippedRatio * advantage);

                // Entropy bonus (encourage exploration)
                var entropy = -actionProbs.Select(p => p * Math.Log(p + 1e-8)).Sum();
                policyLoss -= _entropyCoef * entropy;

                // Value loss (MSE)
                var predictedValue = _valueNetwork.PredictValue(state);
                var valueLoss = Math.Pow(returnValue - predictedValue, 2);

                // Update networks
                _policyOptimizer.Step(_policyNetwork, policyLoss);
                _valueOptimizer.Step(_valueNetwork, valueLoss);

                totalLoss += policyLoss + valueLoss;
            }
        }

        return totalLoss / (_updateEpochs * trajectory.Count);
    }

    private double[] ComputeGAE(List<Experience<TState, int>> trajectory)
    {
        // Generalized Advantage Estimation
        var advantages = new double[trajectory.Count];
        var gae = 0.0;

        for (int t = trajectory.Count - 1; t >= 0; t--)
        {
            var reward = trajectory[t].Reward;
            var value = _valueNetwork.PredictValue(trajectory[t].State);
            var nextValue = trajectory[t].Done ? 0.0 :
                _valueNetwork.PredictValue(trajectory[t].NextState);

            // TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            var delta = reward + _gamma * nextValue - value;

            // GAE: A_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ...
            gae = delta + _gamma * _lambda * gae;
            advantages[t] = gae;
        }

        // Normalize advantages
        var mean = advantages.Average();
        var std = Math.Sqrt(advantages.Select(a => Math.Pow(a - mean, 2)).Average());
        for (int i = 0; i < advantages.Length; i++)
        {
            advantages[i] = (advantages[i] - mean) / (std + 1e-8);
        }

        return advantages;
    }

    private double[] ComputeReturns(List<Experience<TState, int>> trajectory)
    {
        var returns = new double[trajectory.Count];
        var runningReturn = 0.0;

        for (int t = trajectory.Count - 1; t >= 0; t--)
        {
            runningReturn = trajectory[t].Reward + _gamma * runningReturn;
            returns[t] = runningReturn;
        }

        return returns;
    }

    private int SampleAction(Vector<double> probabilities)
    {
        var random = new Random().NextDouble();
        var cumulative = 0.0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (random < cumulative)
                return i;
        }

        return probabilities.Length - 1;
    }

    private double Clamp(double value, double min, double max)
    {
        return Math.Max(min, Math.Min(max, value));
    }
}
```

### Policy and Value Networks

**File**: `src/ReinforcementLearning/Networks/PolicyNetwork.cs`

```csharp
public interface IPolicyNetwork<TState>
{
    /// <summary>
    /// Predicts action probability distribution.
    /// </summary>
    Vector<double> PredictActionProbabilities(TState state);

    int ActionCount { get; }
}

public interface IValueNetwork<TState>
{
    /// <summary>
    /// Predicts state value V(s).
    /// </summary>
    double PredictValue(TState state);
}

public class DensePolicyNetwork : IPolicyNetwork<Vector<double>>
{
    private readonly List<ILayer<double>> _layers;
    private readonly SoftmaxLayer<double> _softmax;

    public DensePolicyNetwork(int stateDim, int actionCount, List<int> hiddenSizes)
    {
        _layers = BuildNetwork(stateDim, actionCount, hiddenSizes);
        _softmax = new SoftmaxLayer<double>();
        ActionCount = actionCount;
    }

    public Vector<double> PredictActionProbabilities(Vector<double> state)
    {
        var input = VectorToMatrix(state);
        var output = Forward(input);

        // Apply softmax to get probabilities
        var probs = _softmax.Forward(output);
        return probs.Row(0);
    }

    public int ActionCount { get; }

    private Matrix<double> Forward(Matrix<double> input)
    {
        var output = input;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    // Helper methods omitted for brevity
}

public class DenseValueNetwork : IValueNetwork<Vector<double>>
{
    private readonly List<ILayer<double>> _layers;

    public DenseValueNetwork(int stateDim, List<int> hiddenSizes)
    {
        _layers = BuildNetwork(stateDim, 1, hiddenSizes); // Output: single value
    }

    public double PredictValue(Vector<double> state)
    {
        var input = VectorToMatrix(state);
        var output = Forward(input);
        return output[0, 0];
    }

    private Matrix<double> Forward(Matrix<double> input)
    {
        var output = input;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }
}
```

---

## Algorithm 3: Asynchronous Advantage Actor-Critic (A3C)

### Theory

A3C runs multiple agents in parallel, each with its own environment copy:
- **Actor**: Policy network π(a|s)
- **Critic**: Value network V(s)
- **Asynchronous**: Multiple workers collect experience independently
- **Advantage**: A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)

**Key Innovation**: Parallel experience collection breaks correlations without replay buffer.

### Implementation

**File**: `src/ReinforcementLearning/Algorithms/A3C/A3CAgent.cs`

```csharp
public class A3CAgent<TState>
{
    private readonly IPolicyNetwork<TState> _globalPolicyNetwork;
    private readonly IValueNetwork<TState> _globalValueNetwork;
    private readonly IEnvironment<TState, int> _environmentFactory;
    private readonly int _numWorkers;
    private readonly double _gamma;

    public A3CAgent(
        IPolicyNetwork<TState> globalPolicy,
        IValueNetwork<TState> globalValue,
        IEnvironment<TState, int> environmentFactory,
        int numWorkers = 4,
        double gamma = 0.99)
    {
        _globalPolicyNetwork = globalPolicy;
        _globalValueNetwork = globalValue;
        _environmentFactory = environmentFactory;
        _numWorkers = numWorkers;
        _gamma = gamma;
    }

    public void Train(int totalSteps)
    {
        // Launch worker threads
        var workers = new List<Task>();

        for (int i = 0; i < _numWorkers; i++)
        {
            int workerId = i;
            var task = Task.Run(() => WorkerThread(workerId, totalSteps / _numWorkers));
            workers.Add(task);
        }

        // Wait for all workers to complete
        Task.WaitAll(workers.ToArray());
    }

    private void WorkerThread(int workerId, int steps)
    {
        // Each worker has local copies of networks
        var localPolicy = _globalPolicyNetwork.Clone();
        var localValue = _globalValueNetwork.Clone();
        var environment = _environmentFactory.Clone();

        var optimizer = new RMSPropOptimizer<double>(learningRate: 0.0007);
        var trajectoryLength = 20; // Update every N steps

        for (int step = 0; step < steps; step++)
        {
            // 1. Sync local networks with global
            localPolicy.SyncFrom(_globalPolicyNetwork);
            localValue.SyncFrom(_globalValueNetwork);

            // 2. Collect trajectory
            var trajectory = CollectTrajectory(environment, localPolicy, trajectoryLength);

            // 3. Compute advantages
            var advantages = ComputeAdvantages(trajectory, localValue);

            // 4. Compute gradients
            var policyGradients = ComputePolicyGradients(localPolicy, trajectory, advantages);
            var valueGradients = ComputeValueGradients(localValue, trajectory, advantages);

            // 5. Apply gradients to global networks (with lock)
            lock (_globalPolicyNetwork)
            {
                optimizer.ApplyGradients(_globalPolicyNetwork, policyGradients);
            }

            lock (_globalValueNetwork)
            {
                optimizer.ApplyGradients(_globalValueNetwork, valueGradients);
            }

            if (step % 100 == 0)
            {
                Console.WriteLine($"Worker {workerId}: Step {step}");
            }
        }
    }

    private List<Experience<TState, int>> CollectTrajectory(
        IEnvironment<TState, int> env,
        IPolicyNetwork<TState> policy,
        int length)
    {
        var trajectory = new List<Experience<TState, int>>();
        var state = env.Reset();

        for (int i = 0; i < length; i++)
        {
            var actionProbs = policy.PredictActionProbabilities(state);
            var action = SampleAction(actionProbs);

            var (nextState, reward, done) = env.Step(action);

            trajectory.Add(new Experience<TState, int>
            {
                State = state,
                Action = action,
                Reward = reward,
                NextState = nextState,
                Done = done
            });

            state = nextState;

            if (done)
            {
                state = env.Reset();
            }
        }

        return trajectory;
    }

    private double[] ComputeAdvantages(
        List<Experience<TState, int>> trajectory,
        IValueNetwork<TState> valueNetwork)
    {
        var advantages = new double[trajectory.Count];

        for (int t = 0; t < trajectory.Count; t++)
        {
            var value = valueNetwork.PredictValue(trajectory[t].State);
            var nextValue = trajectory[t].Done ? 0.0 :
                valueNetwork.PredictValue(trajectory[t].NextState);

            // Advantage: A(s,a) = r + γV(s') - V(s)
            advantages[t] = trajectory[t].Reward + _gamma * nextValue - value;
        }

        return advantages;
    }

    // Gradient computation methods omitted for brevity
}
```

### Shared Global Networks

**Key Points**:
- Global networks are shared across all workers
- Workers periodically sync their local networks with global
- Gradients are accumulated asynchronously (with locks to prevent race conditions)
- No experience replay needed - parallelism provides diversity

---

## Reward Shaping

### Theory

Well-designed rewards are crucial for RL success. Poor rewards lead to:
- Slow learning
- Local optima
- Undesired behaviors

**Reward Shaping Principles**:
1. **Sparse vs Dense**: Dense rewards provide more frequent feedback
2. **Intermediate Rewards**: Reward progress toward goal, not just final success
3. **Negative Penalties**: Discourage bad actions
4. **Discount Factor**: Controls long-term vs short-term focus

### Implementation

**File**: `src/ReinforcementLearning/Rewards/RewardShaper.cs`

```csharp
public interface IRewardShaper<TState, TAction>
{
    double ShapeReward(
        TState state,
        TAction action,
        double rawReward,
        TState nextState,
        bool done);
}

public class PotentialBasedShaper<TState> : IRewardShaper<TState, int>
{
    private readonly Func<TState, double> _potentialFunction;
    private readonly double _gamma;

    public PotentialBasedShaper(Func<TState, double> potentialFunction, double gamma)
    {
        _potentialFunction = potentialFunction;
        _gamma = gamma;
    }

    public double ShapeReward(
        TState state,
        int action,
        double rawReward,
        TState nextState,
        bool done)
    {
        // Potential-based shaping: F(s,s') = γΦ(s') - Φ(s)
        // Proven to not change optimal policy!
        var phi_s = _potentialFunction(state);
        var phi_next = done ? 0.0 : _potentialFunction(nextState);

        var shapingReward = _gamma * phi_next - phi_s;

        return rawReward + shapingReward;
    }
}

// Example: CartPole reward shaping
public class CartPoleRewardShaper : IRewardShaper<CartPoleState, int>
{
    public double ShapeReward(
        CartPoleState state,
        int action,
        double rawReward,
        CartPoleState nextState,
        bool done)
    {
        if (done)
        {
            return -100.0; // Large penalty for falling
        }

        // Reward staying upright and centered
        var angleReward = 1.0 - Math.Abs(nextState.PoleAngle) / 0.2095; // Normalize
        var positionReward = 1.0 - Math.Abs(nextState.CartPosition) / 2.4;

        return angleReward * 0.5 + positionReward * 0.5;
    }
}
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/ReinforcementLearning/DQNTests.cs`

```csharp
[TestClass]
public class DQNTests
{
    [TestMethod]
    public void TestDQN_SimpleMaze()
    {
        // Simple 5x5 grid maze
        var environment = new GridMaze(5, 5);
        var qNetwork = new DenseQNetwork(stateDim: 2, actionCount: 4, new List<int> { 64, 64 });
        var agent = new DQNAgent<Vector<double>, int>(qNetwork);
        var trainer = new DQNTrainer<Vector<double>, int>(agent, environment);

        var results = trainer.Train(numEpisodes: 500);

        // Verify learning: average reward should increase
        var firstHundred = results.EpisodeRewards.Take(100).Average();
        var lastHundred = results.EpisodeRewards.Skip(400).Average();

        Assert.IsTrue(lastHundred > firstHundred,
            $"No learning detected: {firstHundred} -> {lastHundred}");

        // Verify convergence: agent should solve maze
        Assert.IsTrue(lastHundred > 0.8,
            $"Agent failed to solve maze: {lastHundred}");
    }

    [TestMethod]
    public void TestReplayBuffer_Sampling()
    {
        var buffer = new ReplayBuffer<Vector<double>, int>(capacity: 100);

        // Add experiences
        for (int i = 0; i < 150; i++)
        {
            buffer.Add(new Experience<Vector<double>, int>
            {
                State = new Vector<double>(new[] { (double)i }),
                Action = i % 4,
                Reward = i,
                NextState = new Vector<double>(new[] { (double)(i + 1) }),
                Done = false
            });
        }

        // Verify capacity respected
        Assert.AreEqual(100, buffer.Size);

        // Verify sampling is random
        var sample1 = buffer.Sample(32);
        var sample2 = buffer.Sample(32);

        var differentCount = sample1.Zip(sample2, (a, b) => a.Reward != b.Reward).Count(x => x);
        Assert.IsTrue(differentCount > 20, "Samples should be different");
    }

    [TestMethod]
    public void TestTargetNetwork_UpdateFrequency()
    {
        var qNetwork = new DenseQNetwork(stateDim: 4, actionCount: 2, new List<int> { 32 });
        var agent = new DQNAgent<Vector<double>, int>(
            qNetwork,
            targetUpdateFreq: 100);

        // Train for 250 steps
        var environment = new CartPoleEnvironment();
        for (int i = 0; i < 250; i++)
        {
            var state = environment.Reset();
            var action = agent.SelectAction(state);
            var (nextState, reward, done) = environment.Step(action);

            agent.Update(new Experience<Vector<double>, int>
            {
                State = state,
                Action = action,
                Reward = reward,
                NextState = nextState,
                Done = done
            });
        }

        // Verify target network was updated 2 times (at steps 100 and 200)
        // (Implementation detail verification)
    }
}
```

### Integration Tests

Test complete RL workflow on benchmark environments:
1. CartPole (classic control)
2. MountainCar (sparse reward)
3. LunarLander (continuous control)

---

## Common Pitfalls

### 1. Unstable Q-Learning

**Problem**: Q-values diverge, loss explodes

**Solutions**:
- Use target network with slow updates
- Clip gradients to prevent explosions
- Reduce learning rate
- Use Huber loss instead of MSE

### 2. Overestimation Bias

**Problem**: DQN overestimates Q-values due to max operator

**Solution**: Use Double DQN
```csharp
// Instead of: target = r + γ max_a Q_target(s', a)
// Use: target = r + γ Q_target(s', argmax_a Q_main(s', a))
var bestAction = currentQs.Row(i).ArgMax();
var targetQ = nextQs[i, bestAction];
```

### 3. Catastrophic Forgetting

**Problem**: Agent forgets how to solve old states when learning new ones

**Solutions**:
- Larger replay buffer
- Prioritized experience replay (sample important transitions more)
- Regularization techniques

### 4. Exploration-Exploitation Trade-off

**Problem**: Too much exploration wastes time, too little gets stuck

**Solutions**:
- Decay epsilon gradually (1.0 → 0.01 over training)
- Use entropy regularization in policy gradients
- Try curiosity-driven exploration

### 5. Reward Hacking

**Problem**: Agent finds unintended ways to maximize reward

**Solutions**:
- Careful reward design
- Constrain action space
- Human oversight and reward shaping

---

## Advanced Topics

### 1. Prioritized Experience Replay

Sample important transitions more frequently:
```csharp
priority_i ∝ |TD_error_i|^α
```

### 2. Dueling DQN

Separate value and advantage streams:
```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
```

### 3. Multi-Step Returns

Use n-step returns instead of 1-step:
```
G_t = r_t + γ r_{t+1} + ... + γ^n V(s_{t+n})
```

### 4. Distributional RL

Learn distribution of returns instead of expected value (C51, QR-DQN).

### 5. Model-Based RL

Learn environment model, use for planning (World Models, MuZero).

---

## Performance Optimization

### 1. Vectorized Environments

Run multiple environments in parallel:
```csharp
var states = new Matrix<double>(numEnvs, stateDim);
// Execute actions in all environments simultaneously
```

### 2. GPU Acceleration

Use GPU for network forward/backward passes.

### 3. Efficient Data Structures

Use circular buffers for replay memory to avoid reallocations.

### 4. Jit Compilation

Use AOT or JIT compilation for hot paths.

---

## Validation and Verification

### Checklist

- [ ] Agent learns to solve CartPole (avg reward > 195 over 100 episodes)
- [ ] Replay buffer samples uniformly
- [ ] Target network updates at specified frequency
- [ ] Epsilon decays from 1.0 to 0.01
- [ ] Loss decreases over training
- [ ] Policy improves monotonically (for PPO)

### Benchmark Environments

1. **CartPole-v1**: Balance pole on cart (simple, 2D state)
2. **MountainCar-v0**: Drive car up hill (sparse reward challenge)
3. **LunarLander-v2**: Land spacecraft (complex dynamics)
4. **Atari Games**: Visual observations (end-to-end learning)

---

## Resources

### Papers
- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (DQN, 2013)
- Schulman et al., "Proximal Policy Optimization Algorithms" (PPO, 2017)
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (A3C, 2016)

### Books
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd edition)
- Graesser & Keng, "Foundations of Deep Reinforcement Learning"

### Libraries
- OpenAI Gym (environment benchmarks)
- Stable-Baselines3 (reference implementations)

---

## Success Metrics

### Functionality
- [ ] DQN solves CartPole in < 500 episodes
- [ ] PPO achieves stable learning with clipped objective
- [ ] A3C parallelizes training across multiple workers

### Code Quality
- [ ] Modular architecture (agent, network, environment separate)
- [ ] Comprehensive unit tests
- [ ] Benchmark results documented

### Performance
- [ ] Training completes in reasonable time (< 10 min for CartPole)
- [ ] Memory efficient (replay buffer doesn't exceed limits)

---

## Next Steps

After mastering these RL algorithms:
1. Implement **DDPG/TD3** for continuous action spaces
2. Explore **SAC (Soft Actor-Critic)** for maximum entropy RL
3. Study **Hindsight Experience Replay** for sparse rewards
4. Apply RL to real problems (robotics, game playing, resource allocation)

**Congratulations!** You've learned three foundational RL algorithms that power modern AI systems.
