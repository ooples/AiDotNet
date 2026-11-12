using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

        // A zero-length model has nothing to evaluate: every coalition value is 0.
        if (globalModel.Shape[0] == 0) return shapleyValues;

        // For small N, enumerate all subsets; for larger N, fall back to sampling
        if (n > 15)
        {
            return ComputeApproximateShapley(clientModels, globalModel, clientIds);
        }

        int modelSize = globalModel.Shape[0];

        // Enumerate all 2^N subsets: evaluate every coalition once, then accumulate the weighted marginals.
        int totalSubsets = 1 << n;
        var coalitionValues = EvaluateAllCoalitions(clientModels, clientIds, globalModel, modelSize);

        for (int mask = 0; mask < totalSubsets; mask++)
        {
            double coalitionPerformance = coalitionValues[mask];

            // Shapley weight: |S|! * (N - |S| - 1)! / N!
            double weight = ShapleyWeight(CountBits(mask), n);

            // For each client NOT in the coalition, compute their marginal contribution
            for (int j = 0; j < n; j++)
            {
                if ((mask & (1 << j)) != 0) continue; // Client already in coalition

                double marginal = coalitionValues[mask | (1 << j)] - coalitionPerformance;
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
        var reference = FitToLength(globalModel, modelSize).Reshape(new[] { 1, modelSize });

        // The empty coalition aggregates to the zero model, exactly as in the enumerated path.
        double emptyPerformance = NegativeRowDistances(new Tensor<T>(new[] { 1, modelSize }), reference)[0];

        for (int round = 0; round < rounds; round++)
        {
            // Random permutation
            var perm = new List<int>(clientIds);
            for (int i = perm.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }

            // Prefix means of the permuted clients: row i is the coalition of the first i + 1 clients
            // (a running sum, so any number of clients works without a bit mask).
            var ordered = new Tensor<T>[n];
            for (int i = 0; i < n; i++)
            {
                ordered[i] = FitToLength(clientModels[perm[i]], modelSize).Reshape(new[] { 1, modelSize });
            }

            var counts = new double[n];
            for (int i = 0; i < n; i++)
            {
                counts[i] = i + 1;
            }

            var prefixSums = Engine.TensorCumSum(Engine.TensorConcatenate(ordered, 0), 0);
            var prefixMeans = Engine.TensorDivide(prefixSums,
                Engine.TensorBroadcastTo(new Tensor<double>(counts, new[] { n, 1 }).Cast<T>(), new[] { n, modelSize }));
            var performances = NegativeRowDistances(prefixMeans, reference);

            double prevPerformance = emptyPerformance;
            for (int i = 0; i < n; i++)
            {
                shapleyValues[perm[i]] += (performances[i] - prevPerformance) / rounds;
                prevPerformance = performances[i];
            }
        }

        NormalizeScores(shapleyValues);
        return shapleyValues;
    }

    /// <summary>
    /// Values v(S) = -‖mean(S) - global‖ of every coalition bit mask; the empty coalition is the zero model.
    /// Coalitions are averaged as a batched product of membership weights and the client matrix.
    /// </summary>
    private double[] EvaluateAllCoalitions(
        Dictionary<int, Tensor<T>> clientModels, List<int> clientIds, Tensor<T> globalModel, int modelSize)
    {
        int n = clientIds.Count;
        int totalSubsets = 1 << n;
        var rows = new Tensor<T>[n];
        for (int i = 0; i < n; i++)
        {
            rows[i] = FitToLength(clientModels[clientIds[i]], modelSize).Reshape(new[] { 1, modelSize });
        }

        var clientMatrix = Engine.TensorConcatenate(rows, 0); // [n, modelSize]
        var reference = FitToLength(globalModel, modelSize).Reshape(new[] { 1, modelSize });
        var values = new double[totalSubsets];

        // Batch the masks so the coalition block stays around a million entries.
        int batch = Math.Max(1, Math.Min(totalSubsets, (1 << 20) / Math.Max(1, modelSize)));
        for (int start = 0; start < totalSubsets; start += batch)
        {
            int size = Math.Min(batch, totalSubsets - start);
            var membership = new double[size * n];
            for (int r = 0; r < size; r++)
            {
                int mask = start + r;
                int count = CountBits(mask);
                for (int i = 0; i < n; i++)
                {
                    if ((mask & (1 << i)) != 0)
                    {
                        membership[r * n + i] = 1.0 / count;
                    }
                }
            }

            var coalitions = Engine.TensorMatMul(new Tensor<double>(membership, new[] { size, n }).Cast<T>(), clientMatrix);
            var batchValues = NegativeRowDistances(coalitions, reference);
            Array.Copy(batchValues, 0, values, start, size);
        }

        return values;
    }

    /// <summary>
    /// -‖row - reference‖ for every row. Each difference is divided by its largest magnitude before squaring
    /// so the sum of squares cannot overflow narrow T.
    /// </summary>
    private double[] NegativeRowDistances(Tensor<T> rows, Tensor<T> reference)
    {
        // Evaluate model quality as negative L2 distance from the reference (global) model
        // Lower distance = higher performance
        int count = rows.Shape[0];
        var shape = new[] { count, rows.Shape[1] };
        var diff = Engine.TensorSubtract(rows, Engine.TensorBroadcastTo(reference, shape));
        var rowMax = Engine.ReduceMax(Engine.TensorAbs(diff), new[] { 1 }, true);
        var one = Engine.TensorFullLike(rowMax, NumOps.One);
        var safeMax = Engine.TensorWhere(Engine.TensorGreaterThan(rowMax, NumOps.Zero), rowMax, one);
        var scaled = Engine.TensorDivide(diff, Engine.TensorBroadcastTo(safeMax, shape));
        var scaledNorm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(scaled), new[] { 1 }, true));
        var distances = Engine.TensorNegate(Engine.TensorMultiply(rowMax, scaledNorm)).Cast<double>().ToArray();

        // Return negative distance so higher = better
        return distances;
    }

    /// <summary>
    /// Copies the first <paramref name="length"/> entries of a flattened model, zero-padding short models.
    /// </summary>
    private Tensor<T> FitToLength(Tensor<T> model, int length)
    {
        var flat = model.Reshape(new[] { model.Length });
        int covered = Math.Min(length, model.Length);
        if (covered == length)
        {
            return covered == model.Length ? flat : Engine.TensorSlice(flat, new[] { 0 }, new[] { length });
        }

        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { length - covered }) }, 0);
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
