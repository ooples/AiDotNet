namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Aggregate statistics from evaluating a set of trajectories: how many were scored, the reward
/// distribution, and the fraction that met a pass threshold. This is the scoreboard a self-improvement loop
/// watches to know whether it is getting better.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The report card for a batch of runs — average score, best and worst, and what
/// percentage "passed". Compare reports before and after a change to see whether the agents improved.
/// </para>
/// </remarks>
public sealed class EvaluationReport
{
    /// <summary>
    /// Initializes a new report.
    /// </summary>
    /// <param name="count">The number of trajectories scored.</param>
    /// <param name="meanReward">The mean reward.</param>
    /// <param name="minReward">The minimum reward.</param>
    /// <param name="maxReward">The maximum reward.</param>
    /// <param name="passRate">The fraction of trajectories with reward &gt;= <paramref name="passThreshold"/> (0–1).</param>
    /// <param name="passThreshold">The threshold used to compute <paramref name="passRate"/>.</param>
    public EvaluationReport(int count, double meanReward, double minReward, double maxReward, double passRate, double passThreshold)
    {
        Count = count;
        MeanReward = meanReward;
        MinReward = minReward;
        MaxReward = maxReward;
        PassRate = passRate;
        PassThreshold = passThreshold;
    }

    /// <summary>Gets the number of trajectories scored.</summary>
    public int Count { get; }

    /// <summary>Gets the mean reward across scored trajectories (0 when none).</summary>
    public double MeanReward { get; }

    /// <summary>Gets the minimum reward (0 when none).</summary>
    public double MinReward { get; }

    /// <summary>Gets the maximum reward (0 when none).</summary>
    public double MaxReward { get; }

    /// <summary>Gets the fraction of trajectories meeting the pass threshold (0–1).</summary>
    public double PassRate { get; }

    /// <summary>Gets the threshold used to compute <see cref="PassRate"/>.</summary>
    public double PassThreshold { get; }
}
