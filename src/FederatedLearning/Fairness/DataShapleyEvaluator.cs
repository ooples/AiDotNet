using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Data Shapley evaluator: efficient Monte Carlo approximation of Shapley values.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Exact Shapley values require evaluating all possible subsets (2^N),
/// which is impractical for more than ~15 clients. Data Shapley instead randomly samples client
/// orderings (permutations) and averages the marginal contributions. With enough samples, it
/// converges to the true Shapley value while running in polynomial time.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Randomly shuffle the list of clients (a permutation).</description></item>
/// <item><description>Add clients one by one in that order, measuring model performance at each step.</description></item>
/// <item><description>Each client's marginal contribution is the performance jump when they are added.</description></item>
/// <item><description>Repeat many times and average — this converges to the true Shapley value.</description></item>
/// </list>
///
/// <para><b>Recommended for:</b> Federations with 10+ clients where exact Shapley is too expensive.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class DataShapleyEvaluator<T> : FederatedLearningComponentBase<T>, IClientContributionEvaluator<T>
{
    private readonly ContributionEvaluationOptions _options;

    /// <inheritdoc/>
    public string MethodName => "DataShapley";

    /// <summary>
    /// Initializes a new instance of <see cref="DataShapleyEvaluator{T}"/>.
    /// </summary>
    /// <param name="options">Contribution evaluation configuration.</param>
    public DataShapleyEvaluator(ContributionEvaluationOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public Dictionary<int, double> EvaluateContributions(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        if (clientModels is null) throw new ArgumentNullException(nameof(clientModels));
        if (globalModel is null) throw new ArgumentNullException(nameof(globalModel));

        var clientIds = new List<int>(clientModels.Keys);
        int n = clientIds.Count;
        int modelSize = globalModel.Shape[0];

        var shapleyValues = new Dictionary<int, double>();
        var shapleyPrev = new Dictionary<int, double>();

        foreach (int id in clientIds)
        {
            shapleyValues[id] = 0;
            shapleyPrev[id] = 0;
        }

        if (n == 0) return shapleyValues;

        var rng = RandomHelper.CreateSecureRandom();
        var performanceCache = _options.UsePerformanceCache ? new Dictionary<long, double>() : null;

        int completedRounds = 0;
        for (int round = 0; round < _options.SamplingRounds; round++)
        {
            completedRounds = round + 1;
            // Generate random permutation using Fisher-Yates
            var perm = new List<int>(clientIds);
            for (int i = perm.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }

            // Build coalition incrementally
            double prevPerformance = 0;
            var coalitionModel = new Tensor<T>(new[] { modelSize });
            int coalitionCount = 0;

            for (int i = 0; i < n; i++)
            {
                int clientId = perm[i];
                var clientModel = clientModels[clientId];

                // Add client to coalition (incremental weighted average)
                coalitionCount++;
                coalitionModel = AddToRunningAverage(coalitionModel, clientModel, coalitionCount);

                // Evaluate coalition performance
                double currentPerformance;
                long cacheKey = ComputeCoalitionHash(perm, i + 1);

                if (performanceCache is not null && performanceCache.TryGetValue(cacheKey, out double cached))
                {
                    currentPerformance = cached;
                }
                else
                {
                    currentPerformance = EvaluateModel(coalitionModel, globalModel);
                    if (performanceCache is not null)
                    {
                        performanceCache[cacheKey] = currentPerformance;
                    }
                }

                // Marginal contribution of this client
                double marginal = currentPerformance - prevPerformance;
                shapleyValues[clientId] += marginal;
                prevPerformance = currentPerformance;
            }

            // Check for early convergence every 10 rounds
            if (round > 10 && round % 10 == 0)
            {
                double maxChange = 0;
                foreach (int id in clientIds)
                {
                    double currentAvg = shapleyValues[id] / (round + 1);
                    double prevAvg = shapleyPrev[id];
                    maxChange = Math.Max(maxChange, Math.Abs(currentAvg - prevAvg));
                }

                if (maxChange < _options.ConvergenceTolerance)
                {
                    break;
                }

                foreach (int id in clientIds)
                {
                    shapleyPrev[id] = shapleyValues[id] / (round + 1);
                }
            }
        }

        // Average over the rounds actually sampled (the loop can stop early on convergence).
        int effectiveRounds = Math.Max(1, completedRounds);
        var keys = new List<int>(shapleyValues.Keys);
        foreach (int key in keys)
        {
            shapleyValues[key] /= effectiveRounds;
        }

        // Normalize to [0, 1]
        NormalizeScores(shapleyValues);

        return shapleyValues;
    }

    /// <inheritdoc/>
    public HashSet<int> IdentifyFreeRiders(Dictionary<int, double> contributionScores)
    {
        var freeRiders = new HashSet<int>();
        if (contributionScores.Count == 0) return freeRiders;

        double avgScore = 0;
        foreach (double score in contributionScores.Values)
        {
            avgScore += score;
        }
        avgScore /= contributionScores.Count;

        double threshold = avgScore * _options.FreeRiderThreshold;

        foreach (var kvp in contributionScores)
        {
            if (kvp.Value < threshold)
            {
                freeRiders.Add(kvp.Key);
            }
        }

        return freeRiders;
    }

    private double EvaluateModel(Tensor<T> candidateModel, Tensor<T> referenceModel)
    {
        int size = Math.Min(candidateModel.Shape[0], referenceModel.Shape[0]);
        if (size == 0)
        {
            return 0;
        }

        var diff = Engine.TensorSubtract(
            Engine.TensorSlice(candidateModel, new[] { 0 }, new[] { size }),
            Engine.TensorSlice(referenceModel, new[] { 0 }, new[] { size }));

        // ‖diff‖ = m · ‖diff / m‖ with m = max |diff|, so the sum of squares cannot overflow narrow T.
        T maxAbs = Engine.TensorMaxValue(Engine.TensorAbs(diff));
        if (!NumOps.GreaterThan(maxAbs, NumOps.Zero))
        {
            return 0;
        }

        T sumSq = Engine.TensorSum(Engine.TensorSquare(Engine.TensorDivideScalar(diff, maxAbs)));
        return -NumOps.ToDouble(maxAbs) * Math.Sqrt(NumOps.ToDouble(sumSq));
    }

    /// <summary>
    /// Running average new_avg = old_avg + (new_val - old_avg) / count over the entries the client model
    /// covers; entries past the client's length keep the old average.
    /// </summary>
    private Tensor<T> AddToRunningAverage(Tensor<T> average, Tensor<T> clientModel, int count)
    {
        int modelSize = average.Shape[0];
        int covered = Math.Min(modelSize, clientModel.Shape[0]);
        if (covered == 0)
        {
            return average;
        }

        var head = Engine.TensorSlice(average, new[] { 0 }, new[] { covered });
        var delta = Engine.TensorDivideScalar(
            Engine.TensorSubtract(Engine.TensorSlice(clientModel, new[] { 0 }, new[] { covered }), head),
            NumOps.FromDouble(count));
        var updatedHead = Engine.TensorAdd(head, delta);
        if (covered == modelSize)
        {
            return updatedHead;
        }

        var tail = Engine.TensorSlice(average, new[] { covered }, new[] { modelSize - covered });
        return Engine.TensorConcatenate(new[] { updatedHead, tail }, 0);
    }

    private static long ComputeCoalitionHash(List<int> permutation, int count)
    {
        // Simple hash: sort the first 'count' elements and combine
        var subset = new List<int>();
        for (int i = 0; i < count; i++)
        {
            subset.Add(permutation[i]);
        }

        subset.Sort();

        long hash = 17;
        foreach (int id in subset)
        {
            hash = hash * 31 + id;
        }

        return hash;
    }

    private static void NormalizeScores(Dictionary<int, double> scores)
    {
        if (scores.Count == 0) return;

        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (double v in scores.Values)
        {
            min = Math.Min(min, v);
            max = Math.Max(max, v);
        }

        double range = max - min;
        if (range < 1e-12) return;

        var keys = new List<int>(scores.Keys);
        foreach (int key in keys)
        {
            scores[key] = (scores[key] - min) / range;
        }
    }
}
