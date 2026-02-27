namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements FedPAC (Personalization via Aggregation and Calibration) with prototype alignment.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> FedPAC personalizes in two steps. First, it calibrates the
/// aggregation itself — instead of averaging all clients equally, each client aggregates only
/// from "similar" clients (measured by prototype similarity). Second, it aligns class prototypes
/// (average feature vectors per class) across clients so that the shared feature space has
/// consistent semantics. This is especially effective when clients have different class
/// distributions (label skew).</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Each client computes class prototypes: p_c = mean(features where label=c)</item>
/// <item>Clients share prototypes (not raw data) with server</item>
/// <item>Server computes similarity between client prototypes</item>
/// <item>Each client aggregates models from similar clients (weighted by prototype similarity)</item>
/// <item>Local calibration step aligns features to global prototypes</item>
/// </list>
///
/// <para>Reference: FedPAC: Personalization via Aggregation and Calibration (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedPACPersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _similarityThreshold;
    private readonly double _calibrationWeight;
    private Dictionary<int, Dictionary<int, T[]>>? _clientPrototypes;

    /// <summary>
    /// Creates a new FedPAC personalization strategy.
    /// </summary>
    /// <param name="similarityThreshold">Minimum prototype similarity to include a client. Default: 0.3.</param>
    /// <param name="calibrationWeight">Weight of the calibration loss term. Default: 0.1.</param>
    public FedPACPersonalization(double similarityThreshold = 0.3, double calibrationWeight = 0.1)
    {
        if (similarityThreshold < 0 || similarityThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(similarityThreshold), "Similarity threshold must be in [0, 1].");
        }

        if (calibrationWeight < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(calibrationWeight), "Calibration weight must be non-negative.");
        }

        _similarityThreshold = similarityThreshold;
        _calibrationWeight = calibrationWeight;
    }

    /// <summary>
    /// Registers class prototypes for a client.
    /// </summary>
    /// <param name="clientId">Client identifier.</param>
    /// <param name="prototypes">Dictionary of class label to prototype feature vector.</param>
    public void RegisterPrototypes(int clientId, Dictionary<int, T[]> prototypes)
    {
        _clientPrototypes ??= new Dictionary<int, Dictionary<int, T[]>>();
        _clientPrototypes[clientId] = prototypes;
    }

    /// <summary>
    /// Computes personalized aggregation weights based on prototype similarity.
    /// </summary>
    /// <param name="targetClientId">Client requesting personalized weights.</param>
    /// <returns>Dictionary of clientId to weight. Only similar clients are included.</returns>
    public Dictionary<int, double> ComputePersonalizedWeights(int targetClientId)
    {
        if (_clientPrototypes == null || !_clientPrototypes.ContainsKey(targetClientId))
        {
            throw new InvalidOperationException("Prototypes not registered for target client.");
        }

        var targetProtos = _clientPrototypes[targetClientId];
        var weights = new Dictionary<int, double>();

        foreach (var (clientId, clientProtos) in _clientPrototypes)
        {
            double similarity = ComputePrototypeSimilarity(targetProtos, clientProtos);
            if (similarity >= _similarityThreshold)
            {
                weights[clientId] = similarity;
            }
        }

        // Normalize weights.
        double totalWeight = weights.Values.Sum();
        if (totalWeight > 0)
        {
            foreach (var key in weights.Keys.ToArray())
            {
                weights[key] /= totalWeight;
            }
        }

        return weights;
    }

    private double ComputePrototypeSimilarity(
        Dictionary<int, T[]> protosA,
        Dictionary<int, T[]> protosB)
    {
        var commonClasses = protosA.Keys.Intersect(protosB.Keys).ToList();
        if (commonClasses.Count == 0)
        {
            return 0;
        }

        double totalSim = 0;
        foreach (var classLabel in commonClasses)
        {
            var pA = protosA[classLabel];
            var pB = protosB[classLabel];
            int len = Math.Min(pA.Length, pB.Length);

            double dot = 0, normA = 0, normB = 0;
            for (int i = 0; i < len; i++)
            {
                double a = NumOps.ToDouble(pA[i]);
                double b = NumOps.ToDouble(pB[i]);
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }

            double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
            totalSim += denom > 0 ? dot / denom : 0;
        }

        return totalSim / commonClasses.Count;
    }

    /// <summary>Gets the similarity threshold for client inclusion.</summary>
    public double SimilarityThreshold => _similarityThreshold;

    /// <summary>Gets the calibration loss weight.</summary>
    public double CalibrationWeight => _calibrationWeight;
}
