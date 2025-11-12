using AiDotNet.Reasoning.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for reward models that score reasoning quality for reinforcement learning.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A reward model is like a scoring system for reasoning. It assigns
/// "rewards" (scores) to reasoning steps or complete solutions, indicating how good they are.
///
/// This is crucial for training AI systems with reinforcement learning (RL), similar to how
/// ChatGPT o1/o3 and DeepSeek-R1 were trained.
///
/// Two main types:
///
/// **Process Reward Models (PRM):**
/// - Score individual reasoning steps
/// - Like getting points for showing your work correctly
/// - Helps identify where reasoning goes wrong
/// - Example: "Step 2 gets +0.9 reward (correct logic), Step 3 gets +0.3 (questionable)"
///
/// **Outcome Reward Models (ORM):**
/// - Score only the final answer
/// - Like getting points only for the correct final answer
/// - Simpler but less informative than PRM
/// - Example: "Final answer is correct: +1.0 reward"
///
/// Reward models enable:
/// - Training better reasoning systems
/// - Selecting high-quality reasoning paths
/// - Guiding search algorithms toward better solutions
/// </para>
/// </remarks>
public interface IRewardModel<T>
{
    /// <summary>
    /// Calculates the reward for a reasoning step.
    /// </summary>
    /// <param name="step">The reasoning step to score.</param>
    /// <param name="context">Context for evaluation.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reward score.</returns>
    Task<T> CalculateStepRewardAsync(
        ReasoningStep<T> step,
        ReasoningContext context,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Calculates the reward for a complete reasoning chain.
    /// </summary>
    /// <param name="chain">The reasoning chain to score.</param>
    /// <param name="correctAnswer">The known correct answer (if available).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Reward score.</returns>
    Task<T> CalculateChainRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the type of reward model (Process or Outcome).
    /// </summary>
    RewardModelType ModelType { get; }

    /// <summary>
    /// Gets the name of this reward model.
    /// </summary>
    string ModelName { get; }
}

/// <summary>
/// Types of reward models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This enum distinguishes between the two main types of reward models.
/// </para>
/// </remarks>
public enum RewardModelType
{
    /// <summary>
    /// Process Reward Model - scores individual reasoning steps.
    /// </summary>
    Process,

    /// <summary>
    /// Outcome Reward Model - scores only the final answer.
    /// </summary>
    Outcome,

    /// <summary>
    /// Hybrid - combines both process and outcome scoring.
    /// </summary>
    Hybrid
}
