using AiDotNet;
using AiDotNet.ReinforcementLearning;

Console.WriteLine("=== AiDotNet CartPole RL ===");
Console.WriteLine("Training a PPO agent to balance a pole on a cart\n");

// Environment setup
Console.WriteLine("Environment: CartPole");
Console.WriteLine("  State space: 4 dimensions (position, velocity, angle, angular velocity)");
Console.WriteLine("  Action space: 2 actions (left, right)");
Console.WriteLine("  Goal: Keep pole balanced for 500 steps\n");

// Create CartPole environment
var env = new CartPoleEnvironment();

// Create PPO agent
Console.WriteLine("Creating PPO agent...");
Console.WriteLine("  Hidden size: 64");
Console.WriteLine("  Learning rate: 3e-4");
Console.WriteLine("  Gamma: 0.99");
Console.WriteLine("  Epsilon: 0.2\n");

try
{
    var agent = new PPOAgent<double>(
        stateSize: env.StateSize,
        actionSize: env.ActionSize,
        hiddenSize: 64,
        learningRate: 3e-4,
        gamma: 0.99,
        epsilon: 0.2);

    // Training parameters
    const int maxEpisodes = 500;
    const int solvedThreshold = 195;  // Average reward over 100 episodes
    const int reportInterval = 10;

    var rewardHistory = new List<double>();
    double bestReward = 0;
    int? solvedEpisode = null;

    Console.WriteLine("Training...\n");

    for (int episode = 0; episode < maxEpisodes; episode++)
    {
        var state = env.Reset();
        double totalReward = 0;
        int steps = 0;

        while (!env.IsDone && steps < 500)
        {
            // Select action using policy
            var action = agent.SelectAction(state);

            // Take action in environment
            var (nextState, reward, done) = env.Step(action);

            // Store experience
            agent.Store(state, action, reward, nextState, done);

            state = nextState;
            totalReward += reward;
            steps++;
        }

        // Update policy
        agent.Train();

        // Track progress
        rewardHistory.Add(totalReward);
        bestReward = Math.Max(bestReward, totalReward);

        // Calculate moving average
        var recentRewards = rewardHistory.TakeLast(100).ToList();
        double avgReward = recentRewards.Average();

        // Check if solved
        if (recentRewards.Count >= 100 && avgReward >= solvedThreshold && !solvedEpisode.HasValue)
        {
            solvedEpisode = episode;
        }

        // Report progress
        if (episode % reportInterval == 0 || episode == maxEpisodes - 1)
        {
            string status = avgReward >= solvedThreshold ? " (Solved!)" : "";
            Console.WriteLine($"  Episode {episode,3}: Reward = {totalReward,6:F1}, Avg(100) = {avgReward,6:F1}{status}");
        }

        // Early stopping if solved
        if (solvedEpisode.HasValue && episode >= solvedEpisode.Value + 100)
        {
            Console.WriteLine($"\n  Stopping early - solved and stable for 100 episodes");
            break;
        }
    }

    Console.WriteLine("\nTraining complete!");
    Console.WriteLine("─────────────────────────────────────");
    Console.WriteLine($"  Best reward: {bestReward:F0}");
    Console.WriteLine($"  Final avg reward: {rewardHistory.TakeLast(100).Average():F1}");
    if (solvedEpisode.HasValue)
        Console.WriteLine($"  Solved at episode: {solvedEpisode}");

    // Test the trained agent
    Console.WriteLine("\nTesting trained agent...");
    Console.WriteLine("─────────────────────────────────────");

    var testRewards = new List<double>();
    for (int test = 0; test < 5; test++)
    {
        var state = env.Reset();
        double testReward = 0;
        int testSteps = 0;

        while (!env.IsDone && testSteps < 500)
        {
            var action = agent.SelectAction(state, deterministic: true);
            var (nextState, reward, _) = env.Step(action);
            state = nextState;
            testReward += reward;
            testSteps++;
        }

        testRewards.Add(testReward);
        string maxNote = testSteps >= 500 ? " (max)" : "";
        Console.WriteLine($"  Test {test + 1}: {testSteps} steps{maxNote}");
    }

    Console.WriteLine($"\n  Average test reward: {testRewards.Average():F1}");
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full RL training requires complete agent implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for reinforcement learning.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

/// <summary>
/// Simple CartPole environment implementation
/// </summary>
public class CartPoleEnvironment
{
    private readonly Random _random = new(42);

    // Physical constants
    private const double Gravity = 9.8;
    private const double CartMass = 1.0;
    private const double PoleMass = 0.1;
    private const double TotalMass = CartMass + PoleMass;
    private const double PoleLength = 0.5;
    private const double PoleMassLength = PoleMass * PoleLength;
    private const double ForceMag = 10.0;
    private const double Tau = 0.02;  // Time step

    // Thresholds for episode termination
    private const double ThetaThreshold = 12 * Math.PI / 180;  // 12 degrees
    private const double XThreshold = 2.4;

    // State: [x, x_dot, theta, theta_dot]
    private double[] _state = new double[4];

    public int StateSize => 4;
    public int ActionSize => 2;
    public bool IsDone { get; private set; }

    public double[] Reset()
    {
        // Initialize with small random values
        _state = new double[]
        {
            (_random.NextDouble() - 0.5) * 0.1,  // x
            (_random.NextDouble() - 0.5) * 0.1,  // x_dot
            (_random.NextDouble() - 0.5) * 0.1,  // theta
            (_random.NextDouble() - 0.5) * 0.1   // theta_dot
        };
        IsDone = false;
        return _state.ToArray();
    }

    public (double[] nextState, double reward, bool done) Step(int action)
    {
        double x = _state[0];
        double xDot = _state[1];
        double theta = _state[2];
        double thetaDot = _state[3];

        // Apply force based on action
        double force = action == 1 ? ForceMag : -ForceMag;

        // Physics calculations
        double cosTheta = Math.Cos(theta);
        double sinTheta = Math.Sin(theta);

        double temp = (force + PoleMassLength * thetaDot * thetaDot * sinTheta) / TotalMass;
        double thetaAcc = (Gravity * sinTheta - cosTheta * temp) /
            (PoleLength * (4.0 / 3.0 - PoleMass * cosTheta * cosTheta / TotalMass));
        double xAcc = temp - PoleMassLength * thetaAcc * cosTheta / TotalMass;

        // Euler integration
        x += Tau * xDot;
        xDot += Tau * xAcc;
        theta += Tau * thetaDot;
        thetaDot += Tau * thetaAcc;

        _state = new[] { x, xDot, theta, thetaDot };

        // Check termination
        IsDone = Math.Abs(x) > XThreshold || Math.Abs(theta) > ThetaThreshold;

        // Reward is 1 for each step the pole is balanced
        double reward = IsDone ? 0 : 1;

        return (_state.ToArray(), reward, IsDone);
    }
}

/// <summary>
/// Simplified PPO Agent for demonstration
/// </summary>
public class PPOAgent<T>
{
    private readonly int _stateSize;
    private readonly int _actionSize;
    private readonly double _learningRate;
    private readonly double _gamma;
    private readonly double _epsilon;
    private readonly Random _random = new(42);

    // Experience buffer
    private readonly List<(double[] state, int action, double reward, double[] nextState, bool done)> _buffer = new();

    // Simple policy network weights (in real impl, this would be a neural network)
    private double[,] _policyWeights;

    public PPOAgent(int stateSize, int actionSize, int hiddenSize, double learningRate, double gamma, double epsilon)
    {
        _stateSize = stateSize;
        _actionSize = actionSize;
        _learningRate = learningRate;
        _gamma = gamma;
        _epsilon = epsilon;

        // Initialize simple linear policy (real impl would use neural networks)
        _policyWeights = new double[stateSize, actionSize];
        for (int i = 0; i < stateSize; i++)
            for (int j = 0; j < actionSize; j++)
                _policyWeights[i, j] = (_random.NextDouble() - 0.5) * 0.1;
    }

    public int SelectAction(double[] state, bool deterministic = false)
    {
        // Compute action probabilities (softmax over linear combination)
        var logits = new double[_actionSize];
        for (int a = 0; a < _actionSize; a++)
        {
            for (int s = 0; s < _stateSize; s++)
                logits[a] += state[s] * _policyWeights[s, a];
        }

        // Softmax
        double maxLogit = logits.Max();
        var expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray();
        double sumExp = expLogits.Sum();
        var probs = expLogits.Select(e => e / sumExp).ToArray();

        if (deterministic)
        {
            return Array.IndexOf(probs, probs.Max());
        }

        // Sample from distribution
        double r = _random.NextDouble();
        double cumProb = 0;
        for (int a = 0; a < _actionSize; a++)
        {
            cumProb += probs[a];
            if (r < cumProb) return a;
        }
        return _actionSize - 1;
    }

    public void Store(double[] state, int action, double reward, double[] nextState, bool done)
    {
        _buffer.Add((state.ToArray(), action, reward, nextState.ToArray(), done));
    }

    public void Train()
    {
        if (_buffer.Count == 0) return;

        // Compute returns with GAE
        var returns = ComputeReturns();

        // Simple policy gradient update (real PPO would use clipping and multiple epochs)
        for (int i = 0; i < _buffer.Count; i++)
        {
            var (state, action, _, _, _) = _buffer[i];
            double advantage = returns[i];

            // Update policy weights towards the taken action if advantage is positive
            for (int s = 0; s < _stateSize; s++)
            {
                _policyWeights[s, action] += _learningRate * advantage * state[s] * 0.01;
            }
        }

        _buffer.Clear();
    }

    private double[] ComputeReturns()
    {
        var returns = new double[_buffer.Count];
        double runningReturn = 0;

        for (int i = _buffer.Count - 1; i >= 0; i--)
        {
            var (_, _, reward, _, done) = _buffer[i];
            if (done)
                runningReturn = 0;
            runningReturn = reward + _gamma * runningReturn;
            returns[i] = runningReturn;
        }

        // Normalize returns
        double mean = returns.Average();
        double std = Math.Sqrt(returns.Select(r => Math.Pow(r - mean, 2)).Average()) + 1e-8;
        for (int i = 0; i < returns.Length; i++)
            returns[i] = (returns[i] - mean) / std;

        return returns;
    }
}
