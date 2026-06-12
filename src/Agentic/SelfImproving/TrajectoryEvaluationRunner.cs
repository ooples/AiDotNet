namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Runs an <see cref="ITrajectoryEvaluator"/> over trajectories, annotating each with its
/// <see cref="AgentTrajectory.Reward"/> and producing an aggregate <see cref="EvaluationReport"/>. This is
/// the continuous-evaluation step that turns raw captured runs into a measurable quality signal.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Hand it a grader and a batch of recorded runs; it grades each one, writes the
/// score back onto the run, and hands you a summary report. Run it before and after a change to measure
/// whether the agents are improving.
/// </para>
/// </remarks>
public sealed class TrajectoryEvaluationRunner
{
    private readonly ITrajectoryEvaluator _evaluator;

    /// <summary>
    /// Initializes a new runner.
    /// </summary>
    /// <param name="evaluator">The evaluator used to score trajectories.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="evaluator"/> is <c>null</c>.</exception>
    public TrajectoryEvaluationRunner(ITrajectoryEvaluator evaluator)
    {
        Guard.NotNull(evaluator);
        _evaluator = evaluator;
    }

    /// <summary>
    /// Scores each trajectory (writing <see cref="AgentTrajectory.Reward"/>) and returns an aggregate report.
    /// </summary>
    /// <param name="trajectories">The trajectories to evaluate.</param>
    /// <param name="passThreshold">The reward at or above which a trajectory counts as a pass. Default 0.5.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>The aggregate <see cref="EvaluationReport"/>.</returns>
    public async Task<EvaluationReport> EvaluateAsync(
        IReadOnlyList<AgentTrajectory> trajectories,
        double passThreshold = 0.5,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(trajectories);

        if (trajectories.Count == 0)
        {
            return new EvaluationReport(0, 0, 0, 0, 0, passThreshold);
        }

        var sum = 0.0;
        var min = double.PositiveInfinity;
        var max = double.NegativeInfinity;
        var passes = 0;

        foreach (var trajectory in trajectories)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var reward = await _evaluator.EvaluateAsync(trajectory, cancellationToken).ConfigureAwait(false);
            trajectory.Reward = reward;

            sum += reward;
            min = Math.Min(min, reward);
            max = Math.Max(max, reward);
            if (reward >= passThreshold)
            {
                passes++;
            }
        }

        return new EvaluationReport(
            trajectories.Count,
            sum / trajectories.Count,
            min,
            max,
            (double)passes / trajectories.Count,
            passThreshold);
    }

    /// <summary>
    /// Scores every trajectory in a store and returns an aggregate report.
    /// </summary>
    /// <param name="store">The trajectory store to evaluate.</param>
    /// <param name="passThreshold">The reward at or above which a trajectory counts as a pass. Default 0.5.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>The aggregate <see cref="EvaluationReport"/>.</returns>
    public async Task<EvaluationReport> EvaluateStoreAsync(
        ITrajectoryStore store,
        double passThreshold = 0.5,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(store);
        var all = await store.GetAllAsync(cancellationToken).ConfigureAwait(false);
        return await EvaluateAsync(all, passThreshold, cancellationToken).ConfigureAwait(false);
    }
}
