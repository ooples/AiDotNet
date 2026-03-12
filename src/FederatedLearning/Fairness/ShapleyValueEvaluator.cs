using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Exact Shapley value evaluator: computes each client's marginal contribution across all coalitions.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The Shapley value (from game theory, Nobel Prize 2012) is the fairest
/// way to divide credit among participants. It answers: "On average, how much does each client improve
/// the model when added to any possible subset of other clients?"</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Consider every possible subset of clients (coalition).</description></item>
/// <item><description>For each coalition, measure model performance with and without a target client.</description></item>
/// <item><description>Average the marginal improvements across all coalitions to get the Shapley value.</description></item>
/// </list>
///
/// <para><b>Warning:</b> Exact Shapley requires evaluating 2^N coalitions, making it impractical for
/// more than ~15 clients. Use <see cref="DataShapleyEvaluator{T}"/> for larger federations.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class ShapleyValueEvaluator<T> : FederatedLearningComponentBase<T>, IClientContributionEvaluator<T>
{
    private readonly ContributionEvaluationOptions _options;

    /// <inheritdoc/>
    public string MethodName => "ShapleyValue";

    /// <summary>
    /// Initializes a new instance of <see cref="ShapleyValueEvaluator{T}"/>.
    /// </summary>
    /// <param name="options">Contribution evaluation configuration.</param>
    public ShapleyValueEvaluator(ContributionEvaluationOptions options)
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
        var shapleyValues = new Dictionary<int, double>();

        foreach (int clientId in clientIds)
        {
            shapleyValues[clientId] = 0;
        }

        if (n == 0) return shapleyValues;

        // For small N, enumerate all subsets; for larger N, fall back to sampling
        if (n > 15)
        {
            return ComputeApproximateShapley(clientModels, globalModel, clientIds);
        }

        int modelSize = globalModel.Shape[0];

        // Enumerate all 2^N subsets
        int totalSubsets = 1 << n;

        for (int mask = 0; mask < totalSubsets; mask++)
        {
            // Build the coalition model by aggregating included clients
            var coalitionModel = AggregateCoalition(clientModels, clientIds, mask, modelSize);
            double coalitionPerformance = EvaluateModel(coalitionModel, globalModel);

            // For each client NOT in the coalition, compute their marginal contribution
            for (int j = 0; j < n; j++)
            {
                if ((mask & (1 << j)) != 0) continue; // Client already in coalition

                int maskWithClient = mask | (1 << j);
                var coalitionWithClient = AggregateCoalition(clientModels, clientIds, maskWithClient, modelSize);
                double performanceWithClient = EvaluateModel(coalitionWithClient, globalModel);

                double marginal = performanceWithClient - coalitionPerformance;

                // Shapley weight: |S|! * (N - |S| - 1)! / N!
                int coalitionSize = CountBits(mask);
                double weight = ShapleyWeight(coalitionSize, n);

                shapleyValues[clientIds[j]] += weight * marginal;
            }
        }

        // Normalize to [0, 1] range
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

    private Dictionary<int, double> ComputeApproximateShapley(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        List<int> clientIds)
    {
        // Fall back to sampling for large N
        int n = clientIds.Count;
        int modelSize = globalModel.Shape[0];
        var shapleyValues = new Dictionary<int, double>();

        foreach (int id in clientIds)
        {
            shapleyValues[id] = 0;
        }

        var rng = RandomHelper.CreateSecureRandom();
        int rounds = Math.Min(_options.SamplingRounds, 1000);

        for (int round = 0; round < rounds; round++)
        {
            // Random permutation
            var perm = new List<int>(clientIds);
            for (int i = perm.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }

            double prevPerformance = 0; // Empty coalition performance
            int currentMask = 0;

            for (int i = 0; i < n; i++)
            {
                int clientIdx = clientIds.IndexOf(perm[i]);
                currentMask |= (1 << clientIdx);

                var coalitionModel = AggregateCoalition(clientModels, clientIds, currentMask, modelSize);
                double currentPerformance = EvaluateModel(coalitionModel, globalModel);

                shapleyValues[perm[i]] += (currentPerformance - prevPerformance) / rounds;
                prevPerformance = currentPerformance;
            }
        }

        NormalizeScores(shapleyValues);
        return shapleyValues;
    }

    private Tensor<T> AggregateCoalition(
        Dictionary<int, Tensor<T>> clientModels,
        List<int> clientIds,
        int mask,
        int modelSize)
    {
        var result = new Tensor<T>(new[] { modelSize });
        int count = 0;

        for (int i = 0; i < clientIds.Count; i++)
        {
            if ((mask & (1 << i)) == 0) continue;

            var model = clientModels[clientIds[i]];
            for (int j = 0; j < modelSize && j < model.Shape[0]; j++)
            {
                double current = NumOps.ToDouble(result[j]);
                current += NumOps.ToDouble(model[j]);
                result[j] = NumOps.FromDouble(current);
            }

            count++;
        }

        if (count > 0)
        {
            for (int j = 0; j < modelSize; j++)
            {
                result[j] = NumOps.FromDouble(NumOps.ToDouble(result[j]) / count);
            }
        }

        return result;
    }

    private double EvaluateModel(Tensor<T> candidateModel, Tensor<T> referenceModel)
    {
        // Evaluate model quality as negative L2 distance from the reference (global) model
        // Lower distance = higher performance
        int size = Math.Min(candidateModel.Shape[0], referenceModel.Shape[0]);
        double sumSq = 0;

        for (int i = 0; i < size; i++)
        {
            double diff = NumOps.ToDouble(candidateModel[i]) - NumOps.ToDouble(referenceModel[i]);
            sumSq += diff * diff;
        }

        // Return negative distance so higher = better
        return -Math.Sqrt(sumSq);
    }

    private static double ShapleyWeight(int coalitionSize, int totalPlayers)
    {
        // |S|! * (N - |S| - 1)! / N!
        double numerator = Factorial(coalitionSize) * Factorial(totalPlayers - coalitionSize - 1);
        double denominator = Factorial(totalPlayers);

        return denominator > 0 ? numerator / denominator : 0;
    }

    private static double Factorial(int n)
    {
        if (n <= 1) return 1;
        double result = 1;
        for (int i = 2; i <= n; i++)
        {
            result *= i;
        }

        return result;
    }

    private static int CountBits(int mask)
    {
        int count = 0;
        while (mask != 0)
        {
            count += mask & 1;
            mask >>= 1;
        }

        return count;
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
