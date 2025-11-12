using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Policies;

/// <summary>
/// Implements a softmax (Boltzmann) policy that selects actions probabilistically based on Q-values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The softmax policy converts Q-values into a probability distribution using the softmax function
/// (also known as Boltzmann exploration). Actions with higher Q-values have higher selection probabilities,
/// but all actions have a non-zero probability of being selected. The temperature parameter controls
/// the randomness: higher temperature leads to more uniform probabilities (more exploration), while
/// lower temperature makes the policy more deterministic (more exploitation).
/// </para>
/// <para><b>For Beginners:</b> This policy favors better actions but still tries all options sometimes.
///
/// How it differs from epsilon-greedy:
/// - Epsilon-greedy: Either pick best action OR pick randomly
/// - Softmax: Pick actions randomly but favor better ones
///
/// How it works:
/// - Convert Q-values to probabilities using the softmax formula
/// - Higher Q-values get higher probabilities (but not 100%)
/// - Lower Q-values get lower probabilities (but not 0%)
/// - The "temperature" controls how strongly we favor high Q-values
///
/// Temperature effects:
/// - High temperature (e.g., 10.0): Nearly uniform random (lots of exploration)
/// - Medium temperature (e.g., 1.0): Balanced between best and alternatives
/// - Low temperature (e.g., 0.1): Strongly favor best action (little exploration)
/// - Temperature → 0: Becomes pure greedy (always pick best)
///
/// Think of it like choosing from a menu:
/// - High temperature: Might order anything, even if ratings are different
/// - Medium temperature: Likely to order highly-rated dishes, but sometimes try others
/// - Low temperature: Almost always order the top-rated dish
///
/// This is often more "natural" than epsilon-greedy because it never picks terrible actions
/// when good ones are known. The exploration intensity smoothly depends on how much better
/// one action is compared to others.
/// </para>
/// </remarks>
public class SoftmaxPolicy<T> : IPolicy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Random _random;
    private readonly int _actionSpaceSize;
    private double _temperature;
    private readonly double _temperatureMin;
    private readonly double _temperatureDecay;

    /// <summary>
    /// Gets the current temperature parameter.
    /// </summary>
    /// <value>The current temperature value (positive number).</value>
    /// <remarks>
    /// <para>
    /// The temperature parameter controls the randomness of action selection. Higher values
    /// lead to more uniform probabilities (exploration), while lower values make the policy
    /// more deterministic (exploitation). As temperature approaches 0, the policy becomes
    /// greedy (always selecting the best action).
    /// </para>
    /// <para><b>For Beginners:</b> This is the current "randomness level" of the policy.
    ///
    /// What the value means:
    /// - High (e.g., 10.0): Very random, almost ignores Q-values
    /// - Medium (e.g., 1.0): Balanced, favors good actions moderately
    /// - Low (e.g., 0.1): Mostly picks best action, rarely tries others
    /// - Very low (near 0): Almost always picks best action
    ///
    /// You can monitor this value to understand exploration behavior during training.
    /// </para>
    /// </remarks>
    public double Temperature => _temperature;

    /// <summary>
    /// Initializes a new instance of the <see cref="SoftmaxPolicy{T}"/> class.
    /// </summary>
    /// <param name="actionSpaceSize">The number of possible actions.</param>
    /// <param name="temperatureStart">The starting temperature. Default is 1.0.</param>
    /// <param name="temperatureMin">The minimum temperature. Default is 0.1.</param>
    /// <param name="temperatureDecay">The decay factor applied to temperature after each update. Default is 0.995.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// Creates a new softmax policy with the specified parameters. The temperature will start at
    /// temperatureStart and decay by multiplying by temperatureDecay after each Update() call,
    /// until it reaches temperatureMin. This implements an annealing schedule that gradually
    /// shifts from exploration to exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new softmax exploration strategy.
    ///
    /// Parameters explained:
    /// - actionSpaceSize: How many actions the agent can choose from
    /// - temperatureStart: Initial randomness level (1.0 is a good default)
    /// - temperatureMin: Minimum randomness level (0.1 means still some randomness)
    /// - temperatureDecay: How fast to reduce randomness (0.995 = reduce by 0.5% each update)
    /// - seed: Optional number for predictable randomness (testing/debugging)
    ///
    /// Common settings:
    /// - Standard: Start 1.0, decay 0.995, min 0.1 (gradual shift to exploitation)
    /// - Fast cooling: Start 1.0, decay 0.99, min 0.1 (faster shift to exploitation)
    /// - Constant: Start 1.0, decay 1.0, min 1.0 (always same exploration level)
    ///
    /// The decay formula: temperature = max(temperature * decay, temperatureMin)
    ///
    /// Softmax vs Epsilon-Greedy:
    /// - Use softmax when you want smooth, probability-based exploration
    /// - Use epsilon-greedy when you want simple on/off exploration
    /// - Both can work well - often a matter of preference and problem type
    /// </para>
    /// </remarks>
    public SoftmaxPolicy(
        int actionSpaceSize,
        double temperatureStart = 1.0,
        double temperatureMin = 0.1,
        double temperatureDecay = 0.995,
        int? seed = null)
    {
        if (actionSpaceSize <= 0)
        {
            throw new ArgumentException("Action space size must be positive", nameof(actionSpaceSize));
        }

        if (temperatureStart <= 0)
        {
            throw new ArgumentException("Temperature start must be positive", nameof(temperatureStart));
        }

        if (temperatureMin <= 0)
        {
            throw new ArgumentException("Temperature min must be positive", nameof(temperatureMin));
        }

        if (temperatureDecay <= 0 || temperatureDecay > 1)
        {
            throw new ArgumentException("Temperature decay must be between 0 and 1", nameof(temperatureDecay));
        }

        _actionSpaceSize = actionSpaceSize;
        _temperature = temperatureStart;
        _temperatureMin = temperatureMin;
        _temperatureDecay = temperatureDecay;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _numOps = NumericOperations<T>.Instance;
    }

    /// <inheritdoc/>
    public int SelectAction(Tensor<T> state, Tensor<T>? qValues = null)
    {
        if (qValues == null)
        {
            // If no Q-values provided, select uniformly at random
            return _random.Next(_actionSpaceSize);
        }

        // Get action probabilities from softmax
        var probabilities = GetActionProbabilities(state, qValues);

        // Sample from the probability distribution
        double rand = _random.NextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < _actionSpaceSize; i++)
        {
            cumulative += _numOps.ToDouble(probabilities[i]);
            if (rand <= cumulative)
            {
                return i;
            }
        }

        // Fallback (should rarely happen due to numerical precision)
        return _actionSpaceSize - 1;
    }

    /// <inheritdoc/>
    public Tensor<T> GetActionProbabilities(Tensor<T> state, Tensor<T>? qValues = null)
    {
        if (qValues == null)
        {
            // Uniform distribution if no Q-values
            var uniformProbs = new T[_actionSpaceSize];
            for (int i = 0; i < _actionSpaceSize; i++)
            {
                uniformProbs[i] = _numOps.FromDouble(1.0 / _actionSpaceSize);
            }
            return new Tensor<T>(uniformProbs, [_actionSpaceSize]);
        }

        // Apply softmax: exp(Q[i]/T) / sum(exp(Q[j]/T))
        var probabilities = new double[_actionSpaceSize];

        // For numerical stability, subtract max Q-value before exp
        double maxQ = _numOps.ToDouble(qValues[0]);
        for (int i = 1; i < qValues.Length; i++)
        {
            double qVal = _numOps.ToDouble(qValues[i]);
            if (qVal > maxQ)
            {
                maxQ = qVal;
            }
        }

        // Compute exp((Q[i] - maxQ) / T) and sum
        double sum = 0.0;
        for (int i = 0; i < _actionSpaceSize; i++)
        {
            double scaledQ = (_numOps.ToDouble(qValues[i]) - maxQ) / _temperature;
            probabilities[i] = Math.Exp(scaledQ);
            sum += probabilities[i];
        }

        // Normalize to get probabilities
        var result = new T[_actionSpaceSize];
        for (int i = 0; i < _actionSpaceSize; i++)
        {
            result[i] = _numOps.FromDouble(probabilities[i] / sum);
        }

        return new Tensor<T>(result, [_actionSpaceSize]);
    }

    /// <inheritdoc/>
    public void Update()
    {
        // Decay temperature
        _temperature = Math.Max(_temperature * _temperatureDecay, _temperatureMin);
    }

    /// <summary>
    /// Sets the temperature value directly (useful for testing or manual control).
    /// </summary>
    /// <param name="temperature">The new temperature value (must be positive).</param>
    /// <remarks>
    /// <para>
    /// This method allows manual control of the temperature parameter, bypassing the automatic decay.
    /// This can be useful for testing, or for implementing custom temperature schedules.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you manually set the randomness level.
    ///
    /// Use cases:
    /// - Testing: Set very low temperature to see nearly greedy behavior
    /// - Custom schedules: Implement your own decay pattern
    /// - Debugging: Temporarily increase temperature to see more exploration
    /// - Adaptation: Change temperature based on performance
    ///
    /// The value must be positive. Typical range is 0.1 to 10.0.
    /// </para>
    /// </remarks>
    public void SetTemperature(double temperature)
    {
        if (temperature <= 0)
        {
            throw new ArgumentException("Temperature must be positive", nameof(temperature));
        }

        _temperature = temperature;
    }
}
