using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Enforces group fairness constraints during federated learning aggregation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Imagine a federation of hospitals: 8 large urban hospitals and 2 small
/// rural clinics. Without fairness constraints, the global model optimizes mostly for urban patients
/// (the majority). This constraint ensures rural patients get comparable model quality by adjusting
/// how much weight each group receives during aggregation.</para>
///
/// <para><b>Supported constraints:</b></para>
/// <list type="bullet">
/// <item><description><b>Demographic Parity:</b> All groups get equal representation in the model.</description></item>
/// <item><description><b>Equalized Odds:</b> Error rates are balanced across groups.</description></item>
/// <item><description><b>Equal Opportunity:</b> Correct positive predictions are balanced.</description></item>
/// <item><description><b>Minimax:</b> The worst-performing group gets boosted.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class GroupFairnessConstraint<T> : FederatedLearningComponentBase<T>, IFairnessConstraint<T>
{
    private readonly FederatedFairnessOptions _options;

    /// <inheritdoc/>
    public string ConstraintName => _options.ConstraintType.ToString();

    /// <summary>
    /// Initializes a new instance of <see cref="GroupFairnessConstraint{T}"/>.
    /// </summary>
    /// <param name="options">Fairness configuration.</param>
    public GroupFairnessConstraint(FederatedFairnessOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public double EvaluateFairness(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, int> clientGroups)
    {
        if (clientModels is null) throw new ArgumentNullException(nameof(clientModels));
        if (globalModel is null) throw new ArgumentNullException(nameof(globalModel));
        if (clientGroups is null) throw new ArgumentNullException(nameof(clientGroups));

        // Compute per-group model performance (using L2 distance as proxy)
        var groupPerformances = ComputeGroupPerformances(clientModels, globalModel, clientGroups);

        if (groupPerformances.Count < 2) return 0;

        return _options.ConstraintType switch
        {
            FairnessConstraintType.DemographicParity => ComputeDemographicParityViolation(groupPerformances),
            FairnessConstraintType.EqualizedOdds => ComputeEqualizedOddsViolation(groupPerformances),
            FairnessConstraintType.EqualOpportunity => ComputeEqualOpportunityViolation(groupPerformances),
            FairnessConstraintType.MinimaxFairness => ComputeMinimaxViolation(groupPerformances),
            _ => 0
        };
    }

    /// <inheritdoc/>
    public Dictionary<int, double> AdjustWeights(
        Dictionary<int, double> originalWeights,
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, int> clientGroups)
    {
        if (originalWeights is null) throw new ArgumentNullException(nameof(originalWeights));
        if (clientModels is null) throw new ArgumentNullException(nameof(clientModels));

        var adjustedWeights = new Dictionary<int, double>(originalWeights);

        if (_options.ConstraintType == FairnessConstraintType.None ||
            clientGroups is null || clientGroups.Count == 0)
        {
            return adjustedWeights;
        }

        // Compute per-group performance
        var groupPerformances = ComputeGroupPerformances(clientModels, globalModel, clientGroups);

        if (groupPerformances.Count < 2)
        {
            return adjustedWeights;
        }

        // Find the worst-performing group
        int worstGroup = -1;
        double worstPerformance = double.MaxValue;
        double bestPerformance = double.MinValue;

        foreach (var kvp in groupPerformances)
        {
            if (kvp.Value < worstPerformance)
            {
                worstPerformance = kvp.Value;
                worstGroup = kvp.Key;
            }

            bestPerformance = Math.Max(bestPerformance, kvp.Value);
        }

        double performanceGap = bestPerformance - worstPerformance;

        if (performanceGap < _options.FairnessThreshold || worstGroup < 0)
        {
            return adjustedWeights; // Fair enough, no adjustment needed
        }

        // Boost underperforming group(s)
        double boostFactor = 1.0 + _options.FairnessLambda * (performanceGap / Math.Max(1e-12, bestPerformance));

        if (_options.ConstraintType == FairnessConstraintType.MinimaxFairness)
        {
            boostFactor = _options.MinimaxBoostFactor;
        }

        foreach (var kvp in clientGroups)
        {
            int clientId = kvp.Key;
            int groupId = kvp.Value;

            if (!adjustedWeights.ContainsKey(clientId)) continue;

            if (groupId == worstGroup)
            {
                adjustedWeights[clientId] *= boostFactor;
            }
        }

        // Renormalize weights to sum to 1
        double totalWeight = 0;
        foreach (double w in adjustedWeights.Values)
        {
            totalWeight += w;
        }

        if (totalWeight > 1e-12)
        {
            var keys = new List<int>(adjustedWeights.Keys);
            foreach (int key in keys)
            {
                adjustedWeights[key] /= totalWeight;
            }
        }

        return adjustedWeights;
    }

    /// <inheritdoc/>
    public bool IsSatisfied(double fairnessViolation)
    {
        return fairnessViolation <= _options.FairnessThreshold;
    }

    private Dictionary<int, double> ComputeGroupPerformances(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, int> clientGroups)
    {
        // Group clients and compute average model quality per group
        var groupSumPerformance = new Dictionary<int, double>();
        var groupCounts = new Dictionary<int, int>();

        foreach (var kvp in clientGroups)
        {
            int clientId = kvp.Key;
            int groupId = kvp.Value;

            if (!clientModels.ContainsKey(clientId)) continue;

            double performance = ComputeModelQuality(clientModels[clientId], globalModel);

            if (!groupSumPerformance.ContainsKey(groupId))
            {
                groupSumPerformance[groupId] = 0;
                groupCounts[groupId] = 0;
            }

            groupSumPerformance[groupId] += performance;
            groupCounts[groupId]++;
        }

        var result = new Dictionary<int, double>();
        foreach (int groupId in groupSumPerformance.Keys)
        {
            result[groupId] = groupCounts[groupId] > 0
                ? groupSumPerformance[groupId] / groupCounts[groupId]
                : 0;
        }

        return result;
    }

    private double ComputeModelQuality(Tensor<T> clientModel, Tensor<T> globalModel)
    {
        // Use negative L2 distance as a proxy for model quality
        // Lower distance from global = client's model is more aligned
        int size = Math.Min(clientModel.Shape[0], globalModel.Shape[0]);
        double sumSq = 0;

        for (int i = 0; i < size; i++)
        {
            double diff = NumOps.ToDouble(clientModel[i]) - NumOps.ToDouble(globalModel[i]);
            sumSq += diff * diff;
        }

        return -Math.Sqrt(sumSq); // Higher is better
    }

    private static double ComputeDemographicParityViolation(Dictionary<int, double> groupPerformances)
    {
        // Max absolute difference between any two groups' average performance
        double maxDiff = 0;
        var values = new List<double>(groupPerformances.Values);

        for (int i = 0; i < values.Count; i++)
        {
            for (int j = i + 1; j < values.Count; j++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(values[i] - values[j]));
            }
        }

        return maxDiff;
    }

    private static double ComputeEqualizedOddsViolation(Dictionary<int, double> groupPerformances)
    {
        // For equalized odds, we need TPR and FPR per group.
        // Without actual predictions, we approximate using performance spread.
        return ComputeDemographicParityViolation(groupPerformances);
    }

    private static double ComputeEqualOpportunityViolation(Dictionary<int, double> groupPerformances)
    {
        // Equal opportunity focuses on TPR equality.
        // Approximated using performance range.
        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (double v in groupPerformances.Values)
        {
            min = Math.Min(min, v);
            max = Math.Max(max, v);
        }

        return max - min;
    }

    private static double ComputeMinimaxViolation(Dictionary<int, double> groupPerformances)
    {
        // Minimax: violation is how far the worst group is from the average
        double sum = 0;
        double min = double.MaxValue;

        foreach (double v in groupPerformances.Values)
        {
            sum += v;
            min = Math.Min(min, v);
        }

        double avg = groupPerformances.Count > 0 ? sum / groupPerformances.Count : 0;
        return Math.Abs(avg - min);
    }
}
