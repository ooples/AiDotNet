namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FedMA (Federated Matched Averaging) aggregation strategy.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Neural networks have a "permutation problem" — two networks
/// can compute the same function but have their neurons in different orders. FedMA solves this
/// by finding the best alignment (matching) of neurons between client models before averaging
/// them, producing a more accurate global model.</para>
///
/// <para>The algorithm:</para>
/// <list type="number">
/// <item>For each layer, compute a cost matrix between client neurons using weight similarity</item>
/// <item>Solve the assignment problem (Hungarian algorithm) to find optimal neuron matching</item>
/// <item>Permute client model weights to align with the reference</item>
/// <item>Average the aligned models</item>
/// </list>
///
/// <para>Reference: Wang, H., et al. (2020). "Federated Learning with Matched Averaging."
/// ICLR 2020.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedMaAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly int _matchingIterations;
    private readonly double _matchingThreshold;

    /// <summary>
    /// Initializes a new instance of the <see cref="FedMaAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="matchingIterations">Number of matching refinement iterations. Default: 1.</param>
    /// <param name="matchingThreshold">Cosine similarity threshold for matching. Default: 0.5.</param>
    public FedMaAggregationStrategy(int matchingIterations = 1, double matchingThreshold = 0.5)
    {
        if (matchingIterations < 1)
        {
            throw new ArgumentException("Matching iterations must be at least 1.", nameof(matchingIterations));
        }

        if (matchingThreshold < 0 || matchingThreshold > 1)
        {
            throw new ArgumentException("Matching threshold must be between 0 and 1.", nameof(matchingThreshold));
        }

        _matchingIterations = matchingIterations;
        _matchingThreshold = matchingThreshold;
    }

    /// <inheritdoc/>
    public override Dictionary<string, T[]> Aggregate(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientModels == null || clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be null or empty.", nameof(clientModels));
        }

        if (clientModels.Count == 1)
        {
            // Return a defensive copy to prevent callers from mutating the client's model.
            var single = clientModels.First().Value;
            return single.ToDictionary(kv => kv.Key, kv => (T[])kv.Value.Clone());
        }

        var referenceClientId = clientModels.Keys.First();
        var referenceModel = clientModels[referenceClientId];
        var layerNames = referenceModel.Keys.ToArray();

        double totalWeight = GetTotalWeightOrThrow(clientWeights, clientModels.Keys, nameof(clientWeights));

        // Step 1: For each non-reference client, compute optimal neuron permutations
        // using the Hungarian algorithm to align neurons to the reference model.
        var permutedModels = new Dictionary<int, Dictionary<string, T[]>>();
        permutedModels[referenceClientId] = referenceModel;

        foreach (var kvp in clientModels)
        {
            if (kvp.Key == referenceClientId)
            {
                continue;
            }

            permutedModels[kvp.Key] = MatchAndPermute(referenceModel, kvp.Value, layerNames);
        }

        // Step 2: Iterate matching refinement — recompute reference as current average, re-match.
        for (int iter = 1; iter < _matchingIterations; iter++)
        {
            // Compute current weighted average as new reference.
            var currentAvg = WeightedAverage(permutedModels, clientWeights, totalWeight, layerNames, referenceModel);

            // Re-match all clients to the updated reference.
            foreach (var kvp in clientModels)
            {
                permutedModels[kvp.Key] = MatchAndPermute(currentAvg, kvp.Value, layerNames);
            }
        }

        // Step 3: Final weighted average of the matched/permuted models.
        return WeightedAverage(permutedModels, clientWeights, totalWeight, layerNames, referenceModel);
    }

    private Dictionary<string, T[]> MatchAndPermute(
        Dictionary<string, T[]> reference,
        Dictionary<string, T[]> client,
        string[] layerNames)
    {
        var permuted = new Dictionary<string, T[]>(reference.Count);

        foreach (var layerName in layerNames)
        {
            if (!client.TryGetValue(layerName, out var clientParams))
            {
                throw new ArgumentException($"Client is missing layer '{layerName}'.");
            }

            var refParams = reference[layerName];
            int layerSize = refParams.Length;

            if (clientParams.Length != layerSize)
            {
                throw new ArgumentException($"Layer '{layerName}' length mismatch.");
            }

            // Determine neuron count: assume neurons are contiguous blocks.
            // For a weight matrix of size [out_features x in_features] stored flat,
            // each neuron (row) has in_features elements.
            // We use a heuristic: try to find the largest divisor that gives a reasonable neuron size.
            int neuronCount = EstimateNeuronCount(layerSize);
            int neuronSize = layerSize / neuronCount;

            if (neuronCount <= 1 || neuronSize <= 0)
            {
                // Can't meaningfully permute, just copy.
                permuted[layerName] = (T[])clientParams.Clone();
                continue;
            }

            // Build cost matrix: cost[i][j] = negative cosine similarity between
            // reference neuron i and client neuron j.
            var costMatrix = new double[neuronCount, neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                for (int j = 0; j < neuronCount; j++)
                {
                    costMatrix[i, j] = -CosineSimilarity(refParams, i * neuronSize, clientParams, j * neuronSize, neuronSize);
                }
            }

            // Solve assignment using Hungarian algorithm.
            int[] assignment = HungarianAlgorithm(costMatrix, neuronCount);

            // Compute average matching quality to decide if permutation is worthwhile.
            // Applying the threshold per-neuron breaks the bijection guarantee of the
            // Hungarian algorithm, so we evaluate quality holistically per layer.
            double avgSimilarity = 0;
            for (int i = 0; i < neuronCount; i++)
            {
                avgSimilarity += -costMatrix[i, assignment[i]];
            }

            avgSimilarity /= neuronCount;

            if (avgSimilarity < _matchingThreshold)
            {
                // Poor overall match quality: skip permutation for this layer to preserve bijection.
                permuted[layerName] = (T[])clientParams.Clone();
                continue;
            }

            // Apply full permutation (bijection preserved).
            var permutedParams = new T[layerSize];
            for (int i = 0; i < neuronCount; i++)
            {
                int srcNeuron = assignment[i];
                Array.Copy(clientParams, srcNeuron * neuronSize, permutedParams, i * neuronSize, neuronSize);
            }

            permuted[layerName] = permutedParams;
        }

        return permuted;
    }

    private Dictionary<string, T[]> WeightedAverage(
        Dictionary<int, Dictionary<string, T[]>> models,
        Dictionary<int, double> weights,
        double totalWeight,
        string[] layerNames,
        Dictionary<string, T[]> template)
    {
        var result = new Dictionary<string, T[]>(template.Count, template.Comparer);

        foreach (var layerName in layerNames)
        {
            int layerSize = template[layerName].Length;
            var aggregated = CreateZeroInitializedLayer(layerSize);

            foreach (var (clientId, model) in models)
            {
                if (!weights.TryGetValue(clientId, out var w))
                {
                    throw new ArgumentException($"Missing weight for client {clientId}.");
                }

                var clientParams = model[layerName];
                var normalizedWeight = NumOps.FromDouble(w / totalWeight);

                for (int i = 0; i < layerSize; i++)
                {
                    aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(clientParams[i], normalizedWeight));
                }
            }

            result[layerName] = aggregated;
        }

        return result;
    }

    private static double CosineSimilarity(T[] a, int offsetA, T[] b, int offsetB, int length)
    {
        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < length; i++)
        {
            double va = NumOps.ToDouble(a[offsetA + i]);
            double vb = NumOps.ToDouble(b[offsetB + i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-10 ? dot / denom : 0;
    }

    /// <summary>
    /// Estimates the number of neurons in a flattened weight layer.
    /// </summary>
    /// <remarks>
    /// Heuristic: finds the largest divisor of <paramref name="layerSize"/> (up to sqrt)
    /// such that each neuron has at least 4 parameters, then also checks common
    /// power-of-2 layer widths. Falls back to 1 if no reasonable factorization exists.
    /// </remarks>
    private static int EstimateNeuronCount(int layerSize)
    {
        int bestCount = 1;
        int sqrtSize = (int)Math.Sqrt(layerSize);

        for (int n = 2; n <= sqrtSize; n++)
        {
            if (layerSize % n == 0 && layerSize / n >= 4)
            {
                bestCount = n;
            }
        }

        // Also check if layerSize / small_number is a good candidate.
        int[] candidates = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
        foreach (int c in candidates)
        {
            if (c > layerSize)
            {
                break;
            }

            if (layerSize % c == 0 && layerSize / c >= 4)
            {
                bestCount = Math.Max(bestCount, c);
            }
        }

        return bestCount;
    }

    /// <summary>
    /// Hungarian algorithm (Kuhn-Munkres) for optimal assignment.
    /// Finds a minimum-cost perfect matching in a bipartite graph.
    /// </summary>
    private static int[] HungarianAlgorithm(double[,] costMatrix, int n)
    {
        // Based on the Jonker-Volgenant algorithm for the linear assignment problem.
        var u = new double[n + 1]; // potential for rows
        var v = new double[n + 1]; // potential for columns
        var assignment = new int[n + 1]; // assignment[j] = row assigned to column j
        var way = new int[n + 1]; // way[j] = previous column in augmenting path

        for (int i = 1; i <= n; i++)
        {
            assignment[0] = i;
            int j0 = 0;
            var minv = new double[n + 1];
            var used = new bool[n + 1];

            for (int j = 0; j <= n; j++)
            {
                minv[j] = double.MaxValue;
                used[j] = false;
            }

            do
            {
                used[j0] = true;
                int i0 = assignment[j0];
                double delta = double.MaxValue;
                int j1 = -1;

                for (int j = 1; j <= n; j++)
                {
                    if (!used[j])
                    {
                        double cur = costMatrix[i0 - 1, j - 1] - u[i0] - v[j];
                        if (cur < minv[j])
                        {
                            minv[j] = cur;
                            way[j] = j0;
                        }

                        if (minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for (int j = 0; j <= n; j++)
                {
                    if (used[j])
                    {
                        u[assignment[j]] += delta;
                        v[j] -= delta;
                    }
                    else
                    {
                        minv[j] -= delta;
                    }
                }

                j0 = j1;
            }
            while (assignment[j0] != 0);

            do
            {
                int j1 = way[j0];
                assignment[j0] = assignment[j1];
                j0 = j1;
            }
            while (j0 != 0);
        }

        // Convert to 0-indexed: result[row] = assigned column
        var result = new int[n];
        for (int j = 1; j <= n; j++)
        {
            result[assignment[j] - 1] = j - 1;
        }

        return result;
    }

    /// <summary>
    /// Gets the number of matching refinement iterations.
    /// </summary>
    public int MatchingIterations => _matchingIterations;

    /// <summary>
    /// Gets the cosine similarity threshold for neuron matching.
    /// </summary>
    public double MatchingThreshold => _matchingThreshold;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FedMA(iters={_matchingIterations},\u03c4={_matchingThreshold})";
}
