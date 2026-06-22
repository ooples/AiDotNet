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
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="count"/> is negative or <paramref name="passRate"/> is outside [0, 1].
    /// </exception>
    /// <exception cref="ArgumentException">Thrown when any reward statistic or the threshold is NaN or infinite.</exception>
    public EvaluationReport(int count, double meanReward, double minReward, double maxReward, double passRate, double passThreshold)
    {
        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Count cannot be negative.");
        }

        ThrowIfNotFinite(meanReward, nameof(meanReward));
        ThrowIfNotFinite(minReward, nameof(minReward));
        ThrowIfNotFinite(maxReward, nameof(maxReward));
        ThrowIfNotFinite(passThreshold, nameof(passThreshold));
        if (double.IsNaN(passRate) || passRate < 0.0 || passRate > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(passRate), passRate, "Pass rate must be within [0, 1].");
        }

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

    private static void ThrowIfNotFinite(double value, string parameterName)
    {
        // double.IsFinite is unavailable on net471 — spell out both halves.
        if (double.IsNaN(value) || double.IsInfinity(value))
        {
            throw new ArgumentException("Value must be a finite number.", parameterName);
        }
    }
}
