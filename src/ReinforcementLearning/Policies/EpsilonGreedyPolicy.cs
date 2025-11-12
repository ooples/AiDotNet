using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Policies;

/// <summary>
/// Implements an epsilon-greedy policy that balances exploration and exploitation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The epsilon-greedy policy is one of the most popular exploration strategies in reinforcement learning.
/// With probability epsilon, it selects a random action (exploration). With probability 1-epsilon,
/// it selects the action with the highest Q-value (exploitation). Epsilon is typically high at the
/// start of training (to encourage exploration) and decays over time (to shift toward exploitation).
/// </para>
/// <para><b>For Beginners:</b> This policy decides between trying new things and using what works.
///
/// How it works:
/// - Generate a random number between 0 and 1
/// - If the number is less than epsilon: Pick a random action (explore)
/// - Otherwise: Pick the best known action (exploit)
///
/// Example with epsilon = 0.1:
/// - 10% of the time: Try a random action to discover new strategies
/// - 90% of the time: Use the action that seems best based on what we know
///
/// Epsilon decay over time:
/// - Start high (e.g., 1.0): Explore a lot when you know nothing
/// - Gradually decrease (e.g., to 0.01): Explore less as you learn more
/// - End low: Mostly use what you learned, rarely explore
///
/// Think of it like learning to navigate a new city:
/// - At first, try many different routes to learn the layout (high epsilon)
/// - As you learn, mostly use good routes but occasionally try alternatives (medium epsilon)
/// - Eventually, almost always use the best route (low epsilon)
/// </para>
/// </remarks>
public class EpsilonGreedyPolicy<T> : IPolicy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private readonly int _actionSpaceSize;
    private double _epsilon;
    private readonly double _epsilonMin;
    private readonly double _epsilonDecay;

    /// <summary>
    /// Gets the current exploration rate (epsilon).
    /// </summary>
    /// <value>The current value of epsilon (between 0 and 1).</value>
    /// <remarks>
    /// <para>
    /// The epsilon value represents the probability of taking a random action. It typically starts
    /// high (e.g., 1.0) and decays toward a minimum value (e.g., 0.01) as training progresses.
    /// </para>
    /// <para><b>For Beginners:</b> This is the current "curiosity level" of the policy.
    ///
    /// What the value means:
    /// - 1.0 (100%): Always explore randomly
    /// - 0.5 (50%): Half random, half best action
    /// - 0.1 (10%): Rarely explore, mostly use best action
    /// - 0.0 (0%): Never explore, always use best action
    ///
    /// You can check this value to see how much exploration is happening during training.
    /// </para>
    /// </remarks>
    public double Epsilon => _epsilon;

    /// <summary>
    /// Initializes a new instance of the <see cref="EpsilonGreedyPolicy{T}"/> class.
    /// </summary>
    /// <param name="actionSpaceSize">The number of possible actions.</param>
    /// <param name="epsilonStart">The starting exploration rate. Default is 1.0.</param>
    /// <param name="epsilonMin">The minimum exploration rate. Default is 0.01.</param>
    /// <param name="epsilonDecay">The decay factor applied to epsilon after each update. Default is 0.995.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// Creates a new epsilon-greedy policy with the specified parameters. The epsilon value will
    /// start at epsilonStart and decay by multiplying by epsilonDecay after each Update() call,
    /// until it reaches epsilonMin.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new exploration strategy.
    ///
    /// Parameters explained:
    /// - actionSpaceSize: How many actions the agent can choose from
    /// - epsilonStart: Initial curiosity level (1.0 = 100% random at start)
    /// - epsilonMin: Minimum curiosity level (0.01 = 1% random eventually)
    /// - epsilonDecay: How fast to reduce curiosity (0.995 = reduce by 0.5% each update)
    /// - seed: Optional number for predictable randomness (testing/debugging)
    ///
    /// Common settings:
    /// - Fast decay: Start at 1.0, decay 0.99, min 0.01 (reaches min in ~460 updates)
    /// - Slow decay: Start at 1.0, decay 0.995, min 0.01 (reaches min in ~900 updates)
    /// - Constant: Start at 0.1, decay 1.0, min 0.1 (always 10% random)
    ///
    /// The decay formula: epsilon = max(epsilon * decay, epsilonMin)
    /// </para>
    /// </remarks>
    public EpsilonGreedyPolicy(
        int actionSpaceSize,
        double epsilonStart = 1.0,
        double epsilonMin = 0.01,
        double epsilonDecay = 0.995,
        int? seed = null)
    {
        if (actionSpaceSize <= 0)
        {
            throw new ArgumentException("Action space size must be positive", nameof(actionSpaceSize));
        }

        if (epsilonStart < 0 || epsilonStart > 1)
        {
            throw new ArgumentException("Epsilon start must be between 0 and 1", nameof(epsilonStart));
        }

        if (epsilonMin < 0 || epsilonMin > 1)
        {
            throw new ArgumentException("Epsilon min must be between 0 and 1", nameof(epsilonMin));
        }

        if (epsilonDecay < 0 || epsilonDecay > 1)
        {
            throw new ArgumentException("Epsilon decay must be between 0 and 1", nameof(epsilonDecay));
        }

        _actionSpaceSize = actionSpaceSize;
        _epsilon = epsilonStart;
        _epsilonMin = epsilonMin;
        _epsilonDecay = epsilonDecay;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _numOps = NumericOperations<T>.Instance;
    }

    /// <inheritdoc/>
    public int SelectAction(Tensor<T> state, Tensor<T>? qValues = null)
    {
        // Explore: select random action
        if (_random.NextDouble() < _epsilon)
        {
            return _random.Next(_actionSpaceSize);
        }

        // Exploit: select best action based on Q-values
        if (qValues == null)
        {
            throw new ArgumentNullException(nameof(qValues),
                "Q-values must be provided for epsilon-greedy exploitation");
        }

        // Find action with highest Q-value
        int bestAction = 0;
        T bestValue = qValues[0];

        for (int i = 1; i < qValues.Length; i++)
        {
            if (_numOps.GreaterThan(qValues[i], bestValue))
            {
                bestValue = qValues[i];
                bestAction = i;
            }
        }

        return bestAction;
    }

    /// <inheritdoc/>
    public Tensor<T> GetActionProbabilities(Tensor<T> state, Tensor<T>? qValues = null)
    {
        var probabilities = new T[_actionSpaceSize];
        double epsilonPerAction = _epsilon / _actionSpaceSize;

        if (qValues == null)
        {
            // If no Q-values, uniform random distribution
            for (int i = 0; i < _actionSpaceSize; i++)
            {
                probabilities[i] = _numOps.FromDouble(1.0 / _actionSpaceSize);
            }
            return new Tensor<T>(probabilities, [_actionSpaceSize]);
        }

        // Find best action
        int bestAction = 0;
        T bestValue = qValues[0];
        for (int i = 1; i < qValues.Length; i++)
        {
            if (_numOps.GreaterThan(qValues[i], bestValue))
            {
                bestValue = qValues[i];
                bestAction = i;
            }
        }

        // Calculate probabilities: epsilon/|A| for all actions, plus (1-epsilon) for best action
        for (int i = 0; i < _actionSpaceSize; i++)
        {
            if (i == bestAction)
            {
                probabilities[i] = _numOps.FromDouble(epsilonPerAction + (1.0 - _epsilon));
            }
            else
            {
                probabilities[i] = _numOps.FromDouble(epsilonPerAction);
            }
        }

        return new Tensor<T>(probabilities, [_actionSpaceSize]);
    }

    /// <inheritdoc/>
    public void Update()
    {
        // Decay epsilon
        _epsilon = Math.Max(_epsilon * _epsilonDecay, _epsilonMin);
    }

    /// <summary>
    /// Sets the epsilon value directly (useful for testing or manual control).
    /// </summary>
    /// <param name="epsilon">The new epsilon value (must be between 0 and 1).</param>
    /// <remarks>
    /// <para>
    /// This method allows manual control of the exploration rate, bypassing the automatic decay.
    /// This can be useful for testing, or for implementing custom exploration schedules.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you manually set the curiosity level.
    ///
    /// Use cases:
    /// - Testing: Set epsilon to 0.0 to see purely greedy behavior
    /// - Custom schedules: Implement your own decay pattern
    /// - Debugging: Temporarily increase exploration to see what happens
    /// - Adaptation: Change exploration based on performance
    ///
    /// The value must be between 0.0 (never explore) and 1.0 (always explore).
    /// </para>
    /// </remarks>
    public void SetEpsilon(double epsilon)
    {
        if (epsilon < 0 || epsilon > 1)
        {
            throw new ArgumentException("Epsilon must be between 0 and 1", nameof(epsilon));
        }

        _epsilon = epsilon;
    }
}
