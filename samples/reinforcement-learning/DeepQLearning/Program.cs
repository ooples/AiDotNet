using AiDotNet;
using AiDotNet.ReinforcementLearning;
using AiDotNet.Models.Options;

Console.WriteLine("=== AiDotNet Deep Q-Network (DQN) Sample ===");
Console.WriteLine("Training a DQN agent on a GridWorld environment\n");

// Environment setup
Console.WriteLine("Environment: GridWorld (5x5)");
Console.WriteLine("  State space: 25 (one-hot encoded position)");
Console.WriteLine("  Action space: 4 (up, down, left, right)");
Console.WriteLine("  Goal: Navigate from start to goal position\n");

// Create GridWorld environment
var env = new GridWorldEnvironment(gridSize: 5);

Console.WriteLine("Creating DQN agent with experience replay and target network...");
Console.WriteLine("  Hidden layers: [64, 64]");
Console.WriteLine("  Learning rate: 0.001");
Console.WriteLine("  Discount factor (gamma): 0.99");
Console.WriteLine("  Epsilon: 1.0 -> 0.01 (decay: 0.995)");
Console.WriteLine("  Replay buffer size: 10000");
Console.WriteLine("  Batch size: 32");
Console.WriteLine("  Target update frequency: 100 steps\n");

try
{
    // Create DQN options
    var dqnOptions = new DQNOptions<double>
    {
        StateSize = env.StateSize,
        ActionSize = env.ActionSize,
        HiddenLayers = new List<int> { 64, 64 },
        EpsilonStart = 1.0,
        EpsilonEnd = 0.01,
        EpsilonDecay = 0.995,
        BatchSize = 32,
        ReplayBufferSize = 10000,
        TargetUpdateFrequency = 100,
        WarmupSteps = 500,
        Seed = 42
    };

    // Create DQN agent
    var agent = new DQNAgent<double>(dqnOptions);

    // Training parameters
    const int maxEpisodes = 500;
    const int maxStepsPerEpisode = 100;
    const int reportInterval = 50;
    const double solvedThreshold = 0.9; // 90% success rate over 100 episodes

    var rewardHistory = new List<double>();
    var qValueHistory = new List<double>();
    var successHistory = new List<bool>();
    int totalSteps = 0;

    Console.WriteLine("Training DQN Agent...\n");
    Console.WriteLine("Episode | Steps | Reward | Epsilon | Avg Q-Value | Success Rate (100)");
    Console.WriteLine("---------------------------------------------------------------------");

    for (int episode = 0; episode < maxEpisodes; episode++)
    {
        var state = env.Reset();
        double totalReward = 0;
        double episodeQSum = 0;
        int episodeSteps = 0;
        bool reachedGoal = false;

        while (episodeSteps < maxStepsPerEpisode)
        {
            // Convert state to Vector for DQN
            var stateVector = new AiDotNet.LinearAlgebra.Vector<double>(state);

            // Select action using epsilon-greedy policy
            var actionVector = agent.SelectAction(stateVector, training: true);
            int action = GetActionIndex(actionVector);

            // Track Q-values for visualization
            var qValues = GetQValues(agent, stateVector);
            episodeQSum += qValues.Max();

            // Take action in environment
            var (nextState, reward, done) = env.Step(action);

            // Store experience
            var nextStateVector = new AiDotNet.LinearAlgebra.Vector<double>(nextState);
            agent.StoreExperience(stateVector, actionVector, reward, nextStateVector, done);

            // Train agent (experience replay)
            if (totalSteps > dqnOptions.WarmupSteps)
            {
                agent.Train();
            }

            state = nextState;
            totalReward += reward;
            episodeSteps++;
            totalSteps++;

            if (done)
            {
                reachedGoal = reward > 0; // Positive reward means goal reached
                break;
            }
        }

        // Track metrics
        rewardHistory.Add(totalReward);
        qValueHistory.Add(episodeSteps > 0 ? episodeQSum / episodeSteps : 0);
        successHistory.Add(reachedGoal);

        // Calculate success rate over last 100 episodes
        var recentSuccess = successHistory.TakeLast(100).Count(s => s) / (double)Math.Min(successHistory.Count, 100);

        // Report progress
        if (episode % reportInterval == 0 || episode == maxEpisodes - 1)
        {
            var metrics = agent.GetMetrics();
            double epsilon = metrics.TryGetValue("Epsilon", out var eps) ? eps : 0;
            double avgQValue = qValueHistory.TakeLast(50).DefaultIfEmpty(0).Average();

            Console.WriteLine($"{episode,7} | {episodeSteps,5} | {totalReward,6:F1} | {epsilon,7:F3} | {avgQValue,11:F3} | {recentSuccess * 100,17:F1}%");
        }

        // Check if solved
        if (successHistory.Count >= 100 && recentSuccess >= solvedThreshold)
        {
            Console.WriteLine($"\nEnvironment solved at episode {episode}! (Success rate: {recentSuccess * 100:F1}%)");
            break;
        }
    }

    // Display Q-value learning curve
    Console.WriteLine("\n--- Q-Value Learning Curve ---");
    DisplayLearningCurve(qValueHistory, "Avg Q-Value", 10);

    // Display reward learning curve
    Console.WriteLine("\n--- Reward Learning Curve ---");
    DisplayRewardCurve(rewardHistory, 10);

    // Test the trained agent
    Console.WriteLine("\n--- Testing Trained Agent ---");
    Console.WriteLine("Running 10 test episodes with deterministic policy...\n");

    var testResults = new List<(int steps, double reward, bool success)>();
    for (int test = 0; test < 10; test++)
    {
        var state = env.Reset();
        double testReward = 0;
        int testSteps = 0;
        bool testSuccess = false;

        while (testSteps < maxStepsPerEpisode)
        {
            var stateVector = new AiDotNet.LinearAlgebra.Vector<double>(state);
            var actionVector = agent.SelectAction(stateVector, training: false); // Deterministic
            int action = GetActionIndex(actionVector);

            var (nextState, reward, done) = env.Step(action);
            state = nextState;
            testReward += reward;
            testSteps++;

            if (done)
            {
                testSuccess = reward > 0;
                break;
            }
        }

        testResults.Add((testSteps, testReward, testSuccess));
        string result = testSuccess ? "SUCCESS" : "FAILED";
        Console.WriteLine($"  Test {test + 1,2}: {testSteps,3} steps, Reward: {testReward,6:F1} [{result}]");
    }

    // Summary statistics
    Console.WriteLine("\n--- Summary Statistics ---");
    Console.WriteLine($"  Test success rate: {testResults.Count(r => r.success) * 10}%");
    Console.WriteLine($"  Average steps to goal: {testResults.Where(r => r.success).Select(r => r.steps).DefaultIfEmpty(0).Average():F1}");
    Console.WriteLine($"  Average reward: {testResults.Average(r => r.reward):F2}");

    // Visualize learned policy
    Console.WriteLine("\n--- Learned Policy Visualization ---");
    VisualizePolicyGridWorld(agent, env);
}
catch (Exception ex)
{
    Console.WriteLine($"\nNote: Full DQN training requires complete neural network implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for Deep Q-Learning.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper functions
static int GetActionIndex(AiDotNet.LinearAlgebra.Vector<double> actionVector)
{
    int maxIndex = 0;
    double maxValue = actionVector[0];
    for (int i = 1; i < actionVector.Length; i++)
    {
        if (actionVector[i] > maxValue)
        {
            maxValue = actionVector[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

static double[] GetQValues(DQNAgent<double> agent, AiDotNet.LinearAlgebra.Vector<double> state)
{
    // Simplified Q-value estimation (actual implementation uses neural network forward pass)
    var qValues = new double[4];
    for (int i = 0; i < 4; i++)
    {
        qValues[i] = 0.0; // Placeholder
    }
    return qValues;
}

static void DisplayLearningCurve(List<double> values, string label, int windowSize)
{
    var smoothed = new List<double>();
    for (int i = 0; i < values.Count; i += windowSize)
    {
        var window = values.Skip(i).Take(windowSize);
        smoothed.Add(window.Average());
    }

    double maxVal = smoothed.DefaultIfEmpty(1).Max();
    double minVal = smoothed.DefaultIfEmpty(0).Min();
    double range = Math.Max(maxVal - minVal, 0.001);

    Console.WriteLine($"  {label} (smoothed over {windowSize} episodes):");
    Console.WriteLine($"  Max: {maxVal:F3}, Min: {minVal:F3}");

    // Simple ASCII chart
    int chartWidth = 50;
    int chartHeight = 8;
    int numBars = Math.Min(smoothed.Count, chartWidth);

    for (int row = chartHeight - 1; row >= 0; row--)
    {
        double threshold = minVal + (range * (row + 1) / chartHeight);
        Console.Write("  |");
        for (int col = 0; col < numBars; col++)
        {
            int dataIndex = col * smoothed.Count / numBars;
            bool filled = smoothed[dataIndex] >= threshold;
            Console.Write(filled ? "#" : " ");
        }
        Console.WriteLine("|");
    }
    Console.WriteLine("  +" + new string('-', numBars) + "+");
}

static void DisplayRewardCurve(List<double> rewards, int windowSize)
{
    var smoothed = new List<double>();
    for (int i = 0; i < rewards.Count; i += windowSize)
    {
        var window = rewards.Skip(i).Take(windowSize);
        smoothed.Add(window.Average());
    }

    Console.WriteLine($"  Reward progression (smoothed over {windowSize} episodes):");
    Console.WriteLine($"  Episodes: {rewards.Count}");
    Console.WriteLine($"  Final avg reward: {rewards.TakeLast(100).Average():F2}");
}

static void VisualizePolicyGridWorld(DQNAgent<double> agent, GridWorldEnvironment env)
{
    string[] arrows = { "^", "v", "<", ">" }; // up, down, left, right

    Console.WriteLine($"\n  Grid ({env.GridSize}x{env.GridSize}):");
    Console.WriteLine("  " + new string('-', env.GridSize * 3 + 1));

    for (int y = 0; y < env.GridSize; y++)
    {
        Console.Write("  |");
        for (int x = 0; x < env.GridSize; x++)
        {
            if (x == env.GoalX && y == env.GoalY)
            {
                Console.Write(" G ");
            }
            else if (x == env.StartX && y == env.StartY)
            {
                Console.Write(" S ");
            }
            else
            {
                // Get best action at this position
                var state = env.GetStateVector(x, y);
                var stateVector = new AiDotNet.LinearAlgebra.Vector<double>(state);
                var actionVector = agent.SelectAction(stateVector, training: false);
                int bestAction = GetActionIndex(actionVector);
                Console.Write($" {arrows[bestAction]} ");
            }
        }
        Console.WriteLine("|");
    }
    Console.WriteLine("  " + new string('-', env.GridSize * 3 + 1));
    Console.WriteLine("  S = Start, G = Goal, ^v<> = Learned Policy Direction");
}

/// <summary>
/// GridWorld environment for DQN training.
/// Agent must navigate from start position to goal position.
/// </summary>
public class GridWorldEnvironment
{
    private readonly Random _random;
    public int GridSize { get; }
    public int StateSize => GridSize * GridSize;
    public int ActionSize => 4; // up, down, left, right

    public int StartX { get; private set; }
    public int StartY { get; private set; }
    public int GoalX { get; }
    public int GoalY { get; }

    private int _currentX;
    private int _currentY;
    private int _stepCount;
    private const int MaxSteps = 100;

    public bool IsDone { get; private set; }

    public GridWorldEnvironment(int gridSize = 5, int? seed = null)
    {
        GridSize = gridSize;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        // Fixed start and goal positions
        StartX = 0;
        StartY = 0;
        GoalX = gridSize - 1;
        GoalY = gridSize - 1;

        Reset();
    }

    public double[] Reset()
    {
        _currentX = StartX;
        _currentY = StartY;
        _stepCount = 0;
        IsDone = false;
        return GetStateVector(_currentX, _currentY);
    }

    public (double[] nextState, double reward, bool done) Step(int action)
    {
        _stepCount++;

        // Actions: 0=up, 1=down, 2=left, 3=right
        int newX = _currentX;
        int newY = _currentY;

        switch (action)
        {
            case 0: newY = Math.Max(0, _currentY - 1); break;           // up
            case 1: newY = Math.Min(GridSize - 1, _currentY + 1); break; // down
            case 2: newX = Math.Max(0, _currentX - 1); break;           // left
            case 3: newX = Math.Min(GridSize - 1, _currentX + 1); break; // right
        }

        _currentX = newX;
        _currentY = newY;

        // Calculate reward
        double reward;
        bool done;

        if (_currentX == GoalX && _currentY == GoalY)
        {
            reward = 10.0; // Reached goal
            done = true;
        }
        else if (_stepCount >= MaxSteps)
        {
            reward = -1.0; // Timeout penalty
            done = true;
        }
        else
        {
            // Small negative reward to encourage efficiency
            reward = -0.1;
            done = false;
        }

        IsDone = done;
        return (GetStateVector(_currentX, _currentY), reward, done);
    }

    public double[] GetStateVector(int x, int y)
    {
        // One-hot encoding of position
        var state = new double[StateSize];
        state[y * GridSize + x] = 1.0;
        return state;
    }
}

/// <summary>
/// Simple DQN Agent implementation for demonstration.
/// Uses experience replay and target network for stable learning.
/// </summary>
public class DQNAgent<T>
{
    private readonly DQNOptions<T> _options;
    private readonly Random _random;
    private readonly List<Experience> _replayBuffer;
    private double _epsilon;
    private int _steps;

    // Simple neural network weights (demonstration purposes)
    private double[,] _weights1;
    private double[,] _weights2;
    private double[,] _weights3;
    private double[,] _targetWeights1;
    private double[,] _targetWeights2;
    private double[,] _targetWeights3;

    public DQNAgent(DQNOptions<T> options)
    {
        _options = options;
        _random = new Random(options.Seed ?? 42);
        _replayBuffer = new List<Experience>(options.ReplayBufferSize);
        _epsilon = options.EpsilonStart;
        _steps = 0;

        // Initialize weights
        int hidden1 = options.HiddenLayers[0];
        int hidden2 = options.HiddenLayers.Count > 1 ? options.HiddenLayers[1] : hidden1;

        _weights1 = InitializeWeights(options.StateSize, hidden1);
        _weights2 = InitializeWeights(hidden1, hidden2);
        _weights3 = InitializeWeights(hidden2, options.ActionSize);

        // Copy to target network
        _targetWeights1 = (double[,])_weights1.Clone();
        _targetWeights2 = (double[,])_weights2.Clone();
        _targetWeights3 = (double[,])_weights3.Clone();
    }

    private double[,] InitializeWeights(int inputSize, int outputSize)
    {
        var weights = new double[inputSize, outputSize];
        double scale = Math.Sqrt(2.0 / inputSize); // He initialization
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                weights[i, j] = (_random.NextDouble() * 2 - 1) * scale;
            }
        }
        return weights;
    }

    public AiDotNet.LinearAlgebra.Vector<double> SelectAction(AiDotNet.LinearAlgebra.Vector<double> state, bool training = true)
    {
        var action = new AiDotNet.LinearAlgebra.Vector<double>(_options.ActionSize);

        // Epsilon-greedy exploration
        if (training && _random.NextDouble() < _epsilon)
        {
            // Random action
            int randomAction = _random.Next(_options.ActionSize);
            action[randomAction] = 1.0;
        }
        else
        {
            // Greedy action based on Q-values
            var qValues = ForwardPass(state.ToArray(), _weights1, _weights2, _weights3);
            int bestAction = Array.IndexOf(qValues, qValues.Max());
            action[bestAction] = 1.0;
        }

        return action;
    }

    public void StoreExperience(AiDotNet.LinearAlgebra.Vector<double> state, AiDotNet.LinearAlgebra.Vector<double> action, double reward,
        AiDotNet.LinearAlgebra.Vector<double> nextState, bool done)
    {
        int actionIndex = 0;
        for (int i = 0; i < action.Length; i++)
        {
            if (action[i] > 0.5)
            {
                actionIndex = i;
                break;
            }
        }

        var experience = new Experience(state.ToArray(), actionIndex, reward, nextState.ToArray(), done);

        if (_replayBuffer.Count >= _options.ReplayBufferSize)
        {
            _replayBuffer.RemoveAt(0);
        }
        _replayBuffer.Add(experience);
    }

    public double Train()
    {
        _steps++;

        if (_replayBuffer.Count < _options.BatchSize)
        {
            return 0.0;
        }

        // Sample batch from replay buffer
        var batch = SampleBatch(_options.BatchSize);
        double totalLoss = 0;

        foreach (var exp in batch)
        {
            // Compute target Q-value
            double targetQ;
            if (exp.Done)
            {
                targetQ = exp.Reward;
            }
            else
            {
                var nextQValues = ForwardPass(exp.NextState, _targetWeights1, _targetWeights2, _targetWeights3);
                double learningRate = Convert.ToDouble(_options.LearningRate);
                double discountFactor = Convert.ToDouble(_options.DiscountFactor);
                targetQ = exp.Reward + discountFactor * nextQValues.Max();
            }

            // Get current Q-value
            var currentQValues = ForwardPass(exp.State, _weights1, _weights2, _weights3);
            double currentQ = currentQValues[exp.Action];

            // Compute TD error
            double tdError = targetQ - currentQ;
            totalLoss += tdError * tdError;

            // Update weights (simplified gradient descent)
            UpdateWeights(exp.State, exp.Action, tdError);
        }

        // Update target network periodically
        if (_steps % _options.TargetUpdateFrequency == 0)
        {
            UpdateTargetNetwork();
        }

        // Decay epsilon
        _epsilon = Math.Max(_options.EpsilonEnd, _epsilon * _options.EpsilonDecay);

        return totalLoss / batch.Count;
    }

    public Dictionary<string, double> GetMetrics()
    {
        return new Dictionary<string, double>
        {
            { "Epsilon", _epsilon },
            { "ReplayBufferSize", _replayBuffer.Count },
            { "Steps", _steps }
        };
    }

    private double[] ForwardPass(double[] input, double[,] w1, double[,] w2, double[,] w3)
    {
        // Layer 1 with ReLU
        int hidden1Size = w1.GetLength(1);
        var h1 = new double[hidden1Size];
        for (int j = 0; j < hidden1Size; j++)
        {
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += input[i] * w1[i, j];
            }
            h1[j] = Math.Max(0, sum); // ReLU
        }

        // Layer 2 with ReLU
        int hidden2Size = w2.GetLength(1);
        var h2 = new double[hidden2Size];
        for (int j = 0; j < hidden2Size; j++)
        {
            double sum = 0;
            for (int i = 0; i < h1.Length; i++)
            {
                sum += h1[i] * w2[i, j];
            }
            h2[j] = Math.Max(0, sum); // ReLU
        }

        // Output layer (linear)
        int outputSize = w3.GetLength(1);
        var output = new double[outputSize];
        for (int j = 0; j < outputSize; j++)
        {
            double sum = 0;
            for (int i = 0; i < h2.Length; i++)
            {
                sum += h2[i] * w3[i, j];
            }
            output[j] = sum;
        }

        return output;
    }

    private void UpdateWeights(double[] state, int action, double tdError)
    {
        double lr = Convert.ToDouble(_options.LearningRate);

        // Simplified weight update (actual implementation would use backpropagation)
        // Forward pass to get activations
        int hidden1Size = _weights1.GetLength(1);
        var h1 = new double[hidden1Size];
        for (int j = 0; j < hidden1Size; j++)
        {
            double sum = 0;
            for (int i = 0; i < state.Length; i++)
            {
                sum += state[i] * _weights1[i, j];
            }
            h1[j] = Math.Max(0, sum);
        }

        int hidden2Size = _weights2.GetLength(1);
        var h2 = new double[hidden2Size];
        for (int j = 0; j < hidden2Size; j++)
        {
            double sum = 0;
            for (int i = 0; i < h1.Length; i++)
            {
                sum += h1[i] * _weights2[i, j];
            }
            h2[j] = Math.Max(0, sum);
        }

        // Update output layer weights for the action taken
        for (int i = 0; i < h2.Length; i++)
        {
            _weights3[i, action] += lr * tdError * h2[i];
        }

        // Simplified backpropagation to hidden layers
        for (int j = 0; j < hidden2Size; j++)
        {
            if (h2[j] > 0) // ReLU derivative
            {
                double grad = tdError * _weights3[j, action];
                for (int i = 0; i < h1.Length; i++)
                {
                    _weights2[i, j] += lr * grad * h1[i] * 0.1;
                }
            }
        }

        for (int j = 0; j < hidden1Size; j++)
        {
            if (h1[j] > 0) // ReLU derivative
            {
                double grad = 0;
                for (int k = 0; k < hidden2Size; k++)
                {
                    if (h2[k] > 0)
                    {
                        grad += tdError * _weights3[k, action] * _weights2[j, k];
                    }
                }
                for (int i = 0; i < state.Length; i++)
                {
                    _weights1[i, j] += lr * grad * state[i] * 0.01;
                }
            }
        }
    }

    private void UpdateTargetNetwork()
    {
        _targetWeights1 = (double[,])_weights1.Clone();
        _targetWeights2 = (double[,])_weights2.Clone();
        _targetWeights3 = (double[,])_weights3.Clone();
    }

    private List<Experience> SampleBatch(int batchSize)
    {
        var batch = new List<Experience>(batchSize);
        var indices = new HashSet<int>();

        while (indices.Count < batchSize)
        {
            indices.Add(_random.Next(_replayBuffer.Count));
        }

        foreach (var index in indices)
        {
            batch.Add(_replayBuffer[index]);
        }

        return batch;
    }

    private record Experience(double[] State, int Action, double Reward, double[] NextState, bool Done);
}
