using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Fairness;

/// <summary>
/// Implements Lightweight Shapley — O(n) Shapley value approximation using gradient similarity.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Exact Shapley values require evaluating all 2^n subsets of
/// clients — impossibly expensive for even 20 clients. Lightweight Shapley approximates
/// each client's contribution by measuring how similar their gradient is to the ideal global
/// gradient. Clients whose updates are well-aligned with the consensus get high Shapley values,
/// while adversarial or low-quality updates get low values. This runs in O(n) time.</para>
///
/// <para>Approximation:</para>
/// <code>
/// shapley_k ≈ cos_sim(gradient_k, gradient_global) * ||gradient_k|| / ||gradient_global||
/// </code>
///
/// <para>Reference: Lightweight Shapley for Federated Contribution Evaluation (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class LightweightShapleyEvaluator<T> : Infrastructure.FederatedLearningComponentBase<T>, IClientContributionEvaluator<T>
{
    private readonly double _freeRiderThreshold;

    /// <inheritdoc/>
    public string MethodName => "LightweightShapley";

    /// <summary>
    /// Creates a new Lightweight Shapley evaluator.
    /// </summary>
    /// <param name="freeRiderThreshold">Contribution score below which a client is a free-rider. Default: 0.1.</param>
    public LightweightShapleyEvaluator(double freeRiderThreshold = 0.1)
    {
        if (freeRiderThreshold < 0 || freeRiderThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(freeRiderThreshold), "Threshold must be in [0, 1].");
        }

        _freeRiderThreshold = freeRiderThreshold;
    }

    /// <inheritdoc/>
    public Dictionary<int, double> EvaluateContributions(
        Dictionary<int, Tensor<T>> clientModels,
        Tensor<T> globalModel,
        Dictionary<int, List<Tensor<T>>> clientHistories)
    {
        Guard.NotNull(clientModels);
        Guard.NotNull(globalModel);
        var scores = new Dictionary<int, double>();
        if (clientModels.Count == 0)
        {
            return scores;
        }

        // Compute global gradient norm.
        double globalNorm2 = 0;
        for (int i = 0; i < globalModel.Length; i++)
        {
            double v = NumOps.ToDouble(globalModel[i]);
            globalNorm2 += v * v;
        }

        double globalNorm = Math.Sqrt(globalNorm2);

        foreach (var (clientId, clientModel) in clientModels)
        {
            double dot = 0, clientNorm2 = 0;
            if (clientModel.Length != globalModel.Length)
            {
                throw new ArgumentException(
                    $"Client {clientId} model length {clientModel.Length} does not match global model length {globalModel.Length}.");
            }

            int len = clientModel.Length;

            for (int i = 0; i < len; i++)
            {
                double cv = NumOps.ToDouble(clientModel[i]);
                double gv = NumOps.ToDouble(globalModel[i]);
                dot += cv * gv;
                clientNorm2 += cv * cv;
            }

            double clientNorm = Math.Sqrt(clientNorm2);
            double cosSim = (clientNorm > 0 && globalNorm > 0) ? dot / (clientNorm * globalNorm) : 0;
            double magnitude = globalNorm > 0 ? clientNorm / globalNorm : 0;

            scores[clientId] = Math.Max(0, cosSim) * magnitude;
        }

        // Normalize to [0, 1].
        double maxScore = scores.Values.Any() ? scores.Values.Max() : 1;
        if (maxScore > 0)
        {
            foreach (var key in scores.Keys.ToArray())
            {
                scores[key] /= maxScore;
            }
        }

        return scores;
    }

    /// <inheritdoc/>
    public HashSet<int> IdentifyFreeRiders(Dictionary<int, double> contributionScores)
    {
        var freeRiders = new HashSet<int>();
        foreach (var (clientId, score) in contributionScores)
        {
            if (score < _freeRiderThreshold)
            {
                freeRiders.Add(clientId);
            }
        }

        return freeRiders;
    }

    /// <summary>Gets the free-rider threshold.</summary>
    public double FreeRiderThreshold => _freeRiderThreshold;
}
