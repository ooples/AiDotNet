using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Incentive mechanism that rewards clients proportional to their evaluated contribution.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Clients need motivation to participate in federated learning.
/// This mechanism distributes a reward budget proportionally: clients who contribute more
/// valuable data get a bigger share. It also tracks trust over time — consistently helpful
/// clients earn higher trust scores, while erratic or low-quality contributors are flagged.</para>
///
/// <para><b>How rewards work:</b></para>
/// <list type="number">
/// <item><description>Contribution scores are computed by an <see cref="IClientContributionEvaluator{T}"/>.</description></item>
/// <item><description>Scores are clipped to non-negative values (negative contributors get 0).</description></item>
/// <item><description>The total budget is distributed proportionally to clipped scores.</description></item>
/// <item><description>A minimum floor ensures even low contributors get some baseline reward.</description></item>
/// </list>
///
/// <para><b>Trust scoring:</b> Trust is computed from contribution history using exponential
/// moving average with consistency bonus. Clients with stable, positive contributions earn
/// higher trust than those with sporadic or negative contributions.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class ContributionBasedIncentive<T> : FederatedLearningComponentBase<T>, IIncentiveMechanism<T>
{
    private readonly ContributionEvaluationOptions _options;

    /// <inheritdoc/>
    public string MechanismName => "ContributionBased";

    /// <summary>
    /// Minimum reward fraction allocated to each participating client (prevents zero rewards).
    /// </summary>
    private const double MinRewardFraction = 0.01;

    /// <summary>
    /// Decay factor for exponential moving average in trust computation.
    /// </summary>
    private const double TrustDecay = 0.9;

    /// <summary>
    /// Initializes a new instance of <see cref="ContributionBasedIncentive{T}"/>.
    /// </summary>
    /// <param name="options">Contribution evaluation configuration.</param>
    public ContributionBasedIncentive(ContributionEvaluationOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public Dictionary<int, double> ComputeRewards(
        Dictionary<int, double> contributionScores,
        double totalBudget = 1.0)
    {
        if (contributionScores is null) throw new ArgumentNullException(nameof(contributionScores));

        var rewards = new Dictionary<int, double>();
        int n = contributionScores.Count;

        if (n == 0) return rewards;

        // Clip negative contributions to zero
        var clippedScores = new Dictionary<int, double>();
        double totalScore = 0;

        foreach (var kvp in contributionScores)
        {
            double clipped = Math.Max(0, kvp.Value);
            clippedScores[kvp.Key] = clipped;
            totalScore += clipped;
        }

        // Allocate minimum floor to each client
        double floorTotal = n * MinRewardFraction * totalBudget;
        double remainingBudget = Math.Max(0, totalBudget - floorTotal);

        foreach (var kvp in clippedScores)
        {
            // Floor reward
            double reward = MinRewardFraction * totalBudget;

            // Proportional reward from remaining budget
            if (totalScore > 1e-12)
            {
                reward += remainingBudget * (kvp.Value / totalScore);
            }
            else
            {
                // Equal split if all contributions are zero
                reward += remainingBudget / n;
            }

            rewards[kvp.Key] = reward;
        }

        return rewards;
    }

    /// <inheritdoc/>
    public Dictionary<int, double> ComputeTrustScores(
        Dictionary<int, List<double>> contributionHistory)
    {
        if (contributionHistory is null) throw new ArgumentNullException(nameof(contributionHistory));

        var trustScores = new Dictionary<int, double>();

        foreach (var kvp in contributionHistory)
        {
            int clientId = kvp.Key;
            var history = kvp.Value;

            if (history.Count == 0)
            {
                trustScores[clientId] = 0.5; // Neutral trust for new clients
                continue;
            }

            // Exponential moving average of contributions
            double ema = history[0];
            for (int i = 1; i < history.Count; i++)
            {
                ema = TrustDecay * ema + (1.0 - TrustDecay) * history[i];
            }

            // Consistency bonus: low variance = more trustworthy
            double mean = 0;
            foreach (double v in history)
            {
                mean += v;
            }
            mean /= history.Count;

            double variance = 0;
            foreach (double v in history)
            {
                double diff = v - mean;
                variance += diff * diff;
            }
            variance /= history.Count;

            double consistency = 1.0 / (1.0 + Math.Sqrt(variance));

            // Participation bonus: more rounds = more trusted (logarithmic growth)
            double participationBonus = Math.Log(1.0 + history.Count) / Math.Log(1.0 + 100);
            participationBonus = Math.Min(1.0, participationBonus);

            // Combine: EMA (quality) * consistency * participation
            // Map to [0, 1] using sigmoid-like transformation
            double rawTrust = Math.Max(0, ema) * consistency * (0.5 + 0.5 * participationBonus);
            trustScores[clientId] = Sigmoid(rawTrust * 3.0 - 1.0); // Center around 0.5
        }

        return trustScores;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
