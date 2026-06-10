namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Scores a captured <see cref="AgentTrajectory"/>, producing the reward signal the self-improving layer
/// optimizes against (higher is better). This is the seam between "how good was that run?" and every learning
/// mechanism that consumes the answer.
/// </summary>
/// <remarks>
/// <para>
/// Evaluators range from simple (exact-match against a known answer, length/cost penalties) to sophisticated
/// (an LLM-as-judge, or an adapter over the reasoning reward models). Keeping the contract trajectory-native
/// means routing/prompt-optimization/fine-tuning all share one definition of quality and can be swapped
/// without touching the learners.
/// </para>
/// <para><b>For Beginners:</b> A grader. You hand it one recorded run and it returns a score saying how good
/// the outcome was. Collect scores across many runs and the system can tell which behaviors to reinforce.
/// </para>
/// </remarks>
public interface ITrajectoryEvaluator
{
    /// <summary>
    /// Scores a trajectory. Higher is better; the scale is the evaluator's own (commonly 0–1).
    /// </summary>
    /// <param name="trajectory">The trajectory to score.</param>
    /// <param name="cancellationToken">Token used to cancel the evaluation.</param>
    /// <returns>The reward score.</returns>
    Task<double> EvaluateAsync(AgentTrajectory trajectory, CancellationToken cancellationToken = default);
}
