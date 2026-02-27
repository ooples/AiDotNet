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
        Guard.NotNull(prototypes);
        _clientPrototypes ??= new Dictionary<int, Dictionary<int, T[]>>();

        // Defensive copy to prevent external mutation of internal state.
        var copy = new Dictionary<int, T[]>(prototypes.Count);
        foreach (var (classLabel, proto) in prototypes)
        {
            copy[classLabel] = (T[])proto.Clone();
        }

        _clientPrototypes[clientId] = copy;
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
            if (pA.Length != pB.Length)
            {
                throw new ArgumentException(
                    $"Prototype dimension mismatch for class {classLabel}: {pA.Length} != {pB.Length}.");
            }

            int len = pA.Length;

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

    /// <summary>
    /// Computes class prototypes from a client's local data: p_c = mean(features where label = c).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A "prototype" is the average feature vector for all examples
    /// of a given class. It represents the "center" of that class in feature space. By comparing
    /// prototypes across clients, we can measure how similar their data distributions are without
    /// ever sharing the raw data.</para>
    /// </remarks>
    /// <param name="features">Feature vectors extracted by the model for each local sample.</param>
    /// <param name="labels">Class labels for each sample.</param>
    /// <returns>Dictionary of class label to prototype (mean feature vector).</returns>
    public Dictionary<int, T[]> ComputeClassPrototypes(T[][] features, int[] labels)
    {
        Guard.NotNull(features);
        Guard.NotNull(labels);
        if (features.Length != labels.Length)
        {
            throw new ArgumentException("Features and labels must have equal length.");
        }

        if (features.Length == 0)
        {
            return new Dictionary<int, T[]>();
        }

        int featureDim = features[0].Length;
        var classSums = new Dictionary<int, (double[] Sum, int Count)>();

        for (int i = 0; i < features.Length; i++)
        {
            int label = labels[i];
            if (!classSums.TryGetValue(label, out var entry))
            {
                entry = (new double[featureDim], 0);
            }

            for (int d = 0; d < Math.Min(featureDim, features[i].Length); d++)
            {
                entry.Sum[d] += NumOps.ToDouble(features[i][d]);
            }

            classSums[label] = (entry.Sum, entry.Count + 1);
        }

        var prototypes = new Dictionary<int, T[]>(classSums.Count);
        foreach (var (label, (sum, count)) in classSums)
        {
            var proto = new T[featureDim];
            for (int d = 0; d < featureDim; d++)
            {
                proto[d] = NumOps.FromDouble(sum[d] / count);
            }

            prototypes[label] = proto;
        }

        return prototypes;
    }

    /// <summary>
    /// Computes global prototypes by averaging client prototypes per class (weighted by sample count).
    /// </summary>
    /// <param name="clientPrototypes">Per-client prototypes: clientId → (classLabel → prototype).</param>
    /// <param name="clientSampleCounts">Per-client per-class sample counts for proper weighting.</param>
    /// <returns>Global prototypes per class.</returns>
    public Dictionary<int, T[]> ComputeGlobalPrototypes(
        Dictionary<int, Dictionary<int, T[]>> clientPrototypes,
        Dictionary<int, Dictionary<int, int>>? clientSampleCounts = null)
    {
        Guard.NotNull(clientPrototypes);
        // Collect all class labels.
        var allClasses = new HashSet<int>();
        foreach (var protos in clientPrototypes.Values)
        {
            foreach (var classLabel in protos.Keys)
            {
                allClasses.Add(classLabel);
            }
        }

        var globalProtos = new Dictionary<int, T[]>();
        foreach (int classLabel in allClasses)
        {
            double[]? sumProto = null;
            double totalWeight = 0;

            foreach (var (clientId, protos) in clientPrototypes)
            {
                if (!protos.TryGetValue(classLabel, out var proto))
                {
                    continue;
                }

                double weight = clientSampleCounts?.GetValueOrDefault(clientId)?.GetValueOrDefault(classLabel, 1) ?? 1.0;

                if (sumProto == null)
                {
                    sumProto = new double[proto.Length];
                }

                for (int d = 0; d < proto.Length; d++)
                {
                    sumProto[d] += weight * NumOps.ToDouble(proto[d]);
                }

                totalWeight += weight;
            }

            if (sumProto != null && totalWeight > 0)
            {
                var globalProto = new T[sumProto.Length];
                for (int d = 0; d < sumProto.Length; d++)
                {
                    globalProto[d] = NumOps.FromDouble(sumProto[d] / totalWeight);
                }

                globalProtos[classLabel] = globalProto;
            }
        }

        return globalProtos;
    }

    /// <summary>
    /// Computes the calibration loss that aligns local features to global prototypes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After computing global prototypes, we want each client's features
    /// for class C to be close to the global prototype for class C. The calibration loss penalizes
    /// the L2 distance between local prototypes and global prototypes, pulling them together.
    /// This ensures that class C means the same thing across all clients in the shared feature space.</para>
    /// </remarks>
    /// <param name="localPrototypes">This client's class prototypes.</param>
    /// <param name="globalPrototypes">Global prototypes from ComputeGlobalPrototypes.</param>
    /// <returns>Calibration loss (weighted L2 distance between local and global prototypes).</returns>
    public double ComputeCalibrationLoss(
        Dictionary<int, T[]> localPrototypes,
        Dictionary<int, T[]> globalPrototypes)
    {
        Guard.NotNull(localPrototypes);
        Guard.NotNull(globalPrototypes);
        double totalLoss = 0;
        int numClasses = 0;

        foreach (var (classLabel, localProto) in localPrototypes)
        {
            if (!globalPrototypes.TryGetValue(classLabel, out var globalProto))
            {
                continue;
            }

            int dim = Math.Min(localProto.Length, globalProto.Length);
            double l2sq = 0;
            for (int d = 0; d < dim; d++)
            {
                double diff = NumOps.ToDouble(localProto[d]) - NumOps.ToDouble(globalProto[d]);
                l2sq += diff * diff;
            }

            totalLoss += l2sq;
            numClasses++;
        }

        return numClasses > 0 ? _calibrationWeight * totalLoss / numClasses : 0;
    }

    /// <summary>Gets the similarity threshold for client inclusion.</summary>
    public double SimilarityThreshold => _similarityThreshold;

    /// <summary>Gets the calibration loss weight.</summary>
    public double CalibrationWeight => _calibrationWeight;
}
