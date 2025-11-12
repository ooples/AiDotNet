using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Interfaces;

namespace AiDotNet.ReinforcementLearning.Policies;

/// <summary>
/// Implements a greedy policy that always selects the action with the highest Q-value.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// The greedy policy is the simplest exploitation strategy - it always selects the action that
/// currently has the highest estimated Q-value. This policy performs no exploration, so it's
/// typically used only for evaluation (testing) after training, not during training itself.
/// If used during training, it can lead to premature convergence to suboptimal policies.
/// </para>
/// <para><b>For Beginners:</b> This policy always picks the best known action, never exploring.
///
/// Characteristics:
/// - No randomness: Always picks the action with highest Q-value
/// - No exploration: Never tries suboptimal actions
/// - Deterministic: Same state always produces same action
/// - Pure exploitation: Uses only what has been learned
///
/// When to use:
/// - Evaluation/testing: See how well your trained agent performs
/// - Production: Deploy the agent to actually do its task
/// - Benchmarking: Compare different trained agents fairly
///
/// When NOT to use:
/// - During training: Agent won't discover better strategies
/// - Early in learning: Agent doesn't know enough yet
/// - Exploration needed: When environment requires trying different things
///
/// Think of it like:
/// - A student who only studies what they already know (no growth)
/// - A GPS that only shows the one route it knows (might miss better routes)
/// - A chess player who never tries new openings (limits improvement)
///
/// Greedy policy is what you graduate to after training with exploration (epsilon-greedy, softmax, etc.).
/// During training, you explore. During evaluation/production, you're greedy.
/// </para>
/// </remarks>
public class GreedyPolicy<T> : IPolicy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _actionSpaceSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="GreedyPolicy{T}"/> class.
    /// </summary>
    /// <param name="actionSpaceSize">The number of possible actions.</param>
    /// <remarks>
    /// <para>
    /// Creates a new greedy policy for the specified action space. The policy will always
    /// select the action with the highest Q-value when SelectAction is called.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a simple "always pick best" strategy.
    ///
    /// Parameters:
    /// - actionSpaceSize: How many actions the agent can choose from
    ///
    /// Usage example:
    /// ```
    /// // Create policy for environment with 4 actions
    /// var policy = new GreedyPolicy<double>(4);
    ///
    /// // During evaluation, always pick best action
    /// var qValues = agent.GetQValues(state);
    /// int action = policy.SelectAction(state, qValues); // Always picks highest Q
    /// ```
    ///
    /// No other parameters needed because greedy is simple: always pick the maximum.
    /// </para>
    /// </remarks>
    public GreedyPolicy(int actionSpaceSize)
    {
        if (actionSpaceSize <= 0)
        {
            throw new ArgumentException("Action space size must be positive", nameof(actionSpaceSize));
        }

        _actionSpaceSize = actionSpaceSize;
        _numOps = NumericOperations<T>.Instance;
    }

    /// <inheritdoc/>
    public int SelectAction(Tensor<T> state, Tensor<T>? qValues = null)
    {
        if (qValues == null)
        {
            throw new ArgumentNullException(nameof(qValues),
                "Q-values must be provided for greedy policy");
        }

        if (qValues.Length != _actionSpaceSize)
        {
            throw new ArgumentException(
                $"Q-values length ({qValues.Length}) must match action space size ({_actionSpaceSize})",
                nameof(qValues));
        }

        // Find action with highest Q-value
        int bestAction = 0;
        T bestValue = qValues[0];

        for (int i = 1; i < _actionSpaceSize; i++)
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
        if (qValues == null)
        {
            throw new ArgumentNullException(nameof(qValues),
                "Q-values must be provided for greedy policy");
        }

        // Find best action
        int bestAction = SelectAction(state, qValues);

        // Create one-hot probability distribution
        var probabilities = new T[_actionSpaceSize];
        for (int i = 0; i < _actionSpaceSize; i++)
        {
            probabilities[i] = _numOps.FromDouble(i == bestAction ? 1.0 : 0.0);
        }

        return new Tensor<T>(probabilities, [_actionSpaceSize]);
    }

    /// <inheritdoc/>
    public void Update()
    {
        // Greedy policy has no parameters to update
        // This method is a no-op for greedy policy
    }
}
