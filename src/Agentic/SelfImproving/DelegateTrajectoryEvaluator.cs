namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// An <see cref="ITrajectoryEvaluator"/> backed by a user-supplied scoring function — the general-purpose
/// hook for custom rewards (exact-match against a labeled answer, regex/JSON validity, cost penalties, or any
/// combination).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The do-it-yourself grader: you provide a small function that looks at a run and
/// returns a score, and this turns it into an evaluator the rest of the system can use.
/// </para>
/// </remarks>
public sealed class DelegateTrajectoryEvaluator : ITrajectoryEvaluator
{
    private readonly Func<AgentTrajectory, double> _score;

    /// <summary>
    /// Initializes a new evaluator from a synchronous scoring function.
    /// </summary>
    /// <param name="score">The function that scores a trajectory (higher is better).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="score"/> is <c>null</c>.</exception>
    public DelegateTrajectoryEvaluator(Func<AgentTrajectory, double> score)
    {
        Guard.NotNull(score);
        _score = score;
    }

    /// <inheritdoc/>
    public Task<double> EvaluateAsync(AgentTrajectory trajectory, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(trajectory);
        return Task.FromResult(_score(trajectory));
    }
}
