using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Prototypical contribution evaluator: measures client value using prototype representations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Instead of expensive Shapley value computation (which requires
/// testing many model combinations), this method uses a shortcut: compare what each client
/// "knows" (represented as prototype vectors — like a summary of their data) against the
/// global model. Clients whose prototypes align well with the global direction but also bring
/// unique information score highest.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Compute each client's prototype as the mean of their model updates.</description></item>
/// <item><description>Measure alignment: how well does the client's direction match the global model?</description></item>
/// <item><description>Measure diversity: does the client bring unique information not covered by others?</description></item>
/// <item><description>Combine alignment and diversity into a contribution score.</description></item>
/// </list>
///
/// <para><b>Advantage:</b> Constant cost per round O(N * D), where N = clients, D = model size.
/// Much cheaper than Shapley for large federations.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class PrototypicalContributionEvaluator<T> : FederatedLearningComponentBase<T>, IClientContributionEvaluator<T>
{
    private readonly ContributionEvaluationOptions _options;

    /// <inheritdoc/>
    public string MethodName => "Prototypical";

    /// <summary>
    /// Initializes a new instance of <see cref="PrototypicalContributionEvaluator{T}"/>.
    /// </summary>
    /// <param name="options">Contribution evaluation configuration.</param>
    public PrototypicalContributionEvaluator(ContributionEvaluationOptions options)
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

        var scores = new Dictionary<int, double>();
        var clientIds = new List<int>(clientModels.Keys);
        int modelSize = globalModel.Shape[0];
        int k = clientIds.Count;

        if (k == 0) return scores;

        // Compute prototypes: mean update direction for each client
        var prototypes = new Tensor<T>[k];
        for (int c = 0; c < k; c++)
        {
            int clientId = clientIds[c];
            var prototype = FitToLength(clientModels[clientId], modelSize);

            // If we have history, use the mean over all rounds for a more stable prototype
            if (clientHistories is not null && clientHistories.TryGetValue(clientId, out var history) && history.Count > 1)
            {
                prototype = new Tensor<T>(new[] { modelSize });
                foreach (var update in history)
                {
                    prototype = Engine.TensorAdd(prototype, FitToLength(update, modelSize));
                }

                prototype = Engine.TensorDivideScalar(prototype, NumOps.FromDouble(history.Count));
            }

            prototypes[c] = prototype.Reshape(new[] { 1, modelSize });
        }

        // Unit prototypes and the global direction (first modelSize entries of the global model).
        var (units, magnitudes) = NormalizeRows(Engine.TensorConcatenate(prototypes, 0));
        var (globalUnit, _) = NormalizeRows(FitToLength(globalModel, modelSize).Reshape(new[] { 1, modelSize }));
        var alignments = Engine.TensorMatMul(units, Engine.TensorTranspose(globalUnit)).Cast<double>().ToArray();
        var cosines = Engine.TensorMatMul(units, Engine.TensorTranspose(units)).Cast<double>().ToArray();

        // Score each client
        for (int c = 0; c < k; c++)
        {
            double alignment = alignments[c];
            double diversity = ComputeDiversity(c, cosines, k);
            double magnitude = magnitudes[c];

            // Contribution = alignment * diversity * magnitude
            // Alignment rewards clients moving in the right direction
            // Diversity rewards clients bringing unique information
            // Magnitude rewards clients with substantial updates
            scores[clientIds[c]] = Math.Max(0, alignment) * (1.0 + diversity) * (1.0 + Math.Log(1.0 + magnitude));
        }

        NormalizeScores(scores);

        return scores;
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
    private static double ComputeDiversity(int index, double[] cosines, int count)
    {
        // Diversity: average angular distance from other clients
        // Higher = more unique contribution
        if (count < 2)
        {
            return 0;
        }

        double totalDistance = 0;
        for (int other = 0; other < count; other++)
        {
            if (other == index) continue;
            totalDistance += 1.0 - Math.Abs(cosines[index * count + other]); // Angular distance
        }

        return totalDistance / (count - 1);
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

    /// <summary>
    /// Splits each row r of <paramref name="rows"/> into its norm and unit direction, scaling by max |r| first so
    /// the squared norm cannot overflow narrow T. Zero rows get norm 0 and a zero direction.
    /// </summary>
    private (Tensor<T> Units, double[] Norms) NormalizeRows(Tensor<T> rows)
    {
        int count = rows.Shape[0];
        int width = rows.Shape[1];
        var shape = new[] { count, width };
        var one = Engine.TensorFullLike(new Tensor<T>(new[] { count, 1 }), NumOps.One);

        var rowMax = Engine.ReduceMax(Engine.TensorAbs(rows), new[] { 1 }, true); // [count, 1]
        var safeMax = Engine.TensorWhere(Engine.TensorGreaterThan(rowMax, NumOps.Zero), rowMax, one);
        var scaled = Engine.TensorDivide(rows, Engine.TensorBroadcastTo(safeMax, shape));
        var scaledNorm = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(scaled), new[] { 1 }, true));
        var safeNorm = Engine.TensorWhere(Engine.TensorGreaterThan(scaledNorm, NumOps.Zero), scaledNorm, one);
        var units = Engine.TensorDivide(scaled, Engine.TensorBroadcastTo(safeNorm, shape));

        var norms = Engine.TensorMultiply(rowMax, scaledNorm).Cast<double>().ToArray();
        return (units, norms);
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
