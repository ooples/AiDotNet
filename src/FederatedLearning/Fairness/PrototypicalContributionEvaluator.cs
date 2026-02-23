using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

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

        if (clientIds.Count == 0) return scores;

        // Compute prototypes: mean update direction for each client
        var prototypes = new Dictionary<int, double[]>();
        foreach (int clientId in clientIds)
        {
            var prototype = new double[modelSize];
            var model = clientModels[clientId];

            for (int i = 0; i < modelSize && i < model.Shape[0]; i++)
            {
                prototype[i] = NumOps.ToDouble(model[i]);
            }

            // If we have history, use the mean over all rounds for a more stable prototype
            if (clientHistories is not null && clientHistories.ContainsKey(clientId))
            {
                var history = clientHistories[clientId];
                if (history.Count > 1)
                {
                    prototype = new double[modelSize];
                    foreach (var update in history)
                    {
                        for (int i = 0; i < modelSize && i < update.Shape[0]; i++)
                        {
                            prototype[i] += NumOps.ToDouble(update[i]);
                        }
                    }

                    for (int i = 0; i < modelSize; i++)
                    {
                        prototype[i] /= history.Count;
                    }
                }
            }

            prototypes[clientId] = prototype;
        }

        // Compute global direction (mean of all client models)
        var globalDirection = new double[modelSize];
        for (int i = 0; i < modelSize; i++)
        {
            globalDirection[i] = NumOps.ToDouble(globalModel[i]);
        }

        // Score each client
        foreach (int clientId in clientIds)
        {
            double alignment = ComputeAlignment(prototypes[clientId], globalDirection);
            double diversity = ComputeDiversity(clientId, prototypes, clientIds);
            double magnitude = ComputeMagnitude(prototypes[clientId]);

            // Contribution = alignment * diversity * magnitude
            // Alignment rewards clients moving in the right direction
            // Diversity rewards clients bringing unique information
            // Magnitude rewards clients with substantial updates
            scores[clientId] = Math.Max(0, alignment) * (1.0 + diversity) * (1.0 + Math.Log(1.0 +magnitude));
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

    private static double ComputeAlignment(double[] prototype, double[] globalDirection)
    {
        // Cosine similarity between client prototype and global direction
        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < prototype.Length; i++)
        {
            dot += prototype[i] * globalDirection[i];
            normA += prototype[i] * prototype[i];
            normB += globalDirection[i] * globalDirection[i];
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-12 ? dot / denom : 0;
    }

    private static double ComputeDiversity(int clientId, Dictionary<int, double[]> prototypes, List<int> clientIds)
    {
        // Diversity: average angular distance from other clients
        // Higher = more unique contribution
        var clientProto = prototypes[clientId];
        double totalDistance = 0;
        int count = 0;

        foreach (int otherId in clientIds)
        {
            if (otherId == clientId) continue;

            var otherProto = prototypes[otherId];
            double similarity = CosineSimilarity(clientProto, otherProto);
            totalDistance += 1.0 - Math.Abs(similarity); // Angular distance
            count++;
        }

        return count > 0 ? totalDistance / count : 0;
    }

    private static double CosineSimilarity(double[] a, double[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        int size = Math.Min(a.Length, b.Length);

        for (int i = 0; i < size; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-12 ? dot / denom : 0;
    }

    private static double ComputeMagnitude(double[] prototype)
    {
        double sumSq = 0;
        foreach (double v in prototype)
        {
            sumSq += v * v;
        }

        return Math.Sqrt(sumSq);
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
