using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements FLAME (Filtering via cosine similarity + Adaptive clipping + Noise) for
/// Byzantine-robust federated learning with backdoor resistance.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Backdoor attacks in federated learning try to implant hidden
/// triggers in the global model. FLAME defends against this with a three-step approach:
/// (1) use HDBSCAN clustering on cosine distances to identify the honest majority
/// cluster, (2) clip surviving models towards the previous global model to a common distance to
/// prevent magnitude-based attacks, and (3) add calibrated noise to the aggregated result to erase
/// any residual backdoor signal.</para>
///
/// <para>Pipeline (Algorithm 1 of the paper):</para>
/// <list type="number">
/// <item>Flatten all client models into vectors and compute pairwise cosine distances</item>
/// <item>HDBSCAN (min_samples = 1, min_cluster_size = n/2 + 1) admits the majority cluster</item>
/// <item>Clipping bound S_t = median of the distances ||G_{t-1} - W_i|| of all models</item>
/// <item>Clip admitted models towards G_{t-1} to S_t and average them</item>
/// <item>Add Gaussian noise with std = <c>NoiseMultiplier</c> * S_t</item>
/// </list>
///
/// <para>Reference: Nguyen, T. D., et al. (2022). "FLAME: Taming Backdoors in Federated
/// Learning." USENIX Security 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class FlameAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly double _noiseMultiplier;
    private readonly int _minClusterSize;
    private readonly int _seed;
    private int _roundCounter;
    private Dictionary<string, T[]>? _previousGlobal;

    /// <summary>
    /// Initializes a new instance of the <see cref="FlameAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="noiseMultiplier">Gaussian noise multiplier relative to clip norm. Default: 0.001.</param>
    /// <param name="minClusterSize">Minimum number of admitted clients. The paper's HDBSCAN already requires
    /// a majority cluster of n/2 + 1 models; a larger value raises that floor. Default: 2.</param>
    /// <param name="seed">Random seed for noise generation. Default: 42.</param>
    public FlameAggregationStrategy(double noiseMultiplier = 0.001, int minClusterSize = 2, int seed = 42)
    {
        if (noiseMultiplier < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(noiseMultiplier), "Noise multiplier must be non-negative.");
        }

        if (minClusterSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minClusterSize), "Minimum cluster size must be at least 1.");
        }

        _noiseMultiplier = noiseMultiplier;
        _minClusterSize = minClusterSize;
        _seed = seed;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>Follows Algorithm 1 of the FLAME paper. The previous global model G_{t-1} is the result this
    /// instance returned last round (zero before the first round, so the first round clips the models
    /// themselves); call <see cref="SetGlobalModel"/> to start from a known global model.</para>
    /// <list type="number">
    /// <item>Cosine distances c_ij = 1 - cos(W_i, W_j) between the local models.</item>
    /// <item>HDBSCAN with min_samples = 1 and min_cluster_size = n/2 + 1 (single cluster allowed) admits the
    /// majority cluster; everything else is noise.</item>
    /// <item>Clipping bound S_t = median of e_i = ||G_{t-1} - W_i|| over all n models; admitted models are
    /// clipped to G_{t-1} + (W_i - G_{t-1}) * min(1, S_t / e_i).</item>
    /// <item>G_t is the plain average of the clipped admitted models (FLAME does not weight clients, since
    /// self-reported sample counts can be inflated by attackers).</item>
    /// <item>Gaussian noise with std = lambda * S_t is added (lambda = <c>NoiseMultiplier</c>).</item>
    /// </list>
    /// </remarks>
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
            var single = clientModels.First().Value;
            var copy = new Dictionary<string, T[]>(single.Count, single.Comparer);
            foreach (var kv in single) copy[kv.Key] = (T[])kv.Value.Clone();
            RememberGlobalModel(copy);
            return copy;
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();
        int totalParams = layerNames.Sum(ln => referenceModel[ln].Length);
        var clientIds = clientModels.Keys.ToList();
        int n = clientIds.Count;

        // Validate all clients have identical layer structure before flattening.
        foreach (var clientId in clientIds)
        {
            var model = clientModels[clientId];
            foreach (var layerName in layerNames)
            {
                if (!model.TryGetValue(layerName, out var layer))
                {
                    throw new ArgumentException(
                        $"Client {clientId} missing layer '{layerName}'.", nameof(clientModels));
                }

                if (layer.Length != referenceModel[layerName].Length)
                {
                    throw new ArgumentException(
                        $"Client {clientId} layer '{layerName}' length mismatch: {layer.Length} != {referenceModel[layerName].Length}.",
                        nameof(clientModels));
                }
            }
        }

        // Flatten the client models into rows of W [n, totalParams].
        var flat = new T[n * totalParams];
        for (int c = 0; c < n; c++)
        {
            var model = clientModels[clientIds[c]];
            int offset = c * totalParams;
            foreach (var layerName in layerNames)
            {
                var layer = model[layerName];
                layer.AsSpan().CopyTo(flat.AsSpan(offset, layer.Length));
                offset += layer.Length;
            }
        }

        var models = new Tensor<T>(flat, new[] { n, totalParams });

        // Step 1: HDBSCAN on cosine distances admits the majority cluster.
        var admitted = IdentifyHonestCluster(models, n);

        // If clustering fails (e.g., minClusterSize > n), fall back to all clients.
        if (admitted.Length == 0)
        {
            admitted = Enumerable.Range(0, n).ToArray();
        }

        // Step 2: Adaptive clipping bound S_t = median distance of all models to G_{t-1}. The distances are taken
        // pre-scaled and kept in double: a squared Half distance overflows once a model is ~256 away from G_{t-1}.
        var previousGlobal = GetPreviousGlobal(layerNames, referenceModel, totalParams);
        var deltas = Engine.TensorSubtract(
            models, Engine.TensorBroadcastTo(previousGlobal.Reshape(new[] { 1, totalParams }), new[] { n, totalParams }));
        var distances = Infrastructure.FederatedTensorMath<T>.NormalizeRows(deltas).Norms;
        var sortedDistances = (double[])distances.Clone();
        Array.Sort(sortedDistances);
        double clipBound = n % 2 == 1
            ? sortedDistances[n / 2]
            : (sortedDistances[n / 2 - 1] + sortedDistances[n / 2]) / 2.0;

        // Clip factors min(1, S_t / e_i) of the admitted models; a model at distance 0 is not scaled.
        var admittedFactors = new double[admitted.Length];
        for (int a = 0; a < admitted.Length; a++)
        {
            double distance = distances[admitted[a]];
            admittedFactors[a] = distance > clipBound ? clipBound / distance : 1.0;
        }

        // Step 3: G_t = G_{t-1} + mean over admitted models of the clipped deltas.
        var admittedIndex = new Tensor<int>(admitted, new[] { admitted.Length });
        var admittedDeltas = Engine.TensorIndexSelect(deltas, admittedIndex, 0);
        var factorColumn = new Tensor<double>(admittedFactors, new[] { admitted.Length, 1 }).Cast<T>();
        var clippedDeltas = Engine.TensorMultiply(
            admittedDeltas, Engine.TensorBroadcastTo(factorColumn, new[] { admitted.Length, totalParams }));
        var meanClippedDelta = Engine.ReduceMean(clippedDeltas, new[] { 0 }, false);
        var aggregatedTensor = Engine.TensorAdd(previousGlobal, meanClippedDelta);

        // Step 4: Add calibrated Gaussian noise with std = lambda * S_t.
        if (_noiseMultiplier > 0)
        {
            double noiseStd = _noiseMultiplier * clipBound;
            var rng = RandomHelper.CreateSeededRandom(_seed + _roundCounter++);
            aggregatedTensor = Engine.TensorAdd(aggregatedTensor, GaussianNoise(rng, totalParams, noiseStd));
        }

        var aggregated = aggregatedTensor.ToVector();

        // Unflatten into the layer dictionary.
        var result = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        var aggregatedSpan = aggregated.AsSpan();
        int resultOffset = 0;
        foreach (var layerName in layerNames)
        {
            int length = referenceModel[layerName].Length;
            result[layerName] = aggregatedSpan.Slice(resultOffset, length).ToArray();
            resultOffset += length;
        }

        RememberGlobalModel(result);
        return result;
    }

    /// <summary>
    /// Box-Muller draws N(0, std^2), one (u1, u2) pair per element in order.
    /// </summary>
    private static Tensor<T> GaussianNoise(Random rng, int count, double std)
    {
        var noise = new double[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[i] = std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        return new Tensor<double>(noise, new[] { count }).Cast<T>();
    }

    /// <summary>
    /// Sets the global model G_{t-1} that the next round clips towards.
    /// </summary>
    /// <param name="globalModel">The current global model parameters, by layer.</param>
    public void SetGlobalModel(Dictionary<string, T[]> globalModel)
    {
        Guard.NotNull(globalModel);
        RememberGlobalModel(globalModel);
    }

    private void RememberGlobalModel(Dictionary<string, T[]> model)
    {
        _previousGlobal = new Dictionary<string, T[]>(model.Count, model.Comparer);
        foreach (var kv in model)
        {
            _previousGlobal[kv.Key] = (T[])kv.Value.Clone();
        }
    }

    /// <summary>
    /// Returns the flattened previous global model, or zeros when there is none with this layer layout.
    /// </summary>
    private Tensor<T> GetPreviousGlobal(string[] layerNames, Dictionary<string, T[]> referenceModel, int totalParams)
    {
        var previous = _previousGlobal;
        if (previous is null || previous.Count != layerNames.Length)
        {
            return new Tensor<T>(new[] { totalParams });
        }

        var flat = new T[totalParams];
        int offset = 0;
        foreach (var layerName in layerNames)
        {
            if (!previous.TryGetValue(layerName, out var layer) || layer.Length != referenceModel[layerName].Length)
            {
                return new Tensor<T>(new[] { totalParams });
            }

            layer.AsSpan().CopyTo(flat.AsSpan(offset, layer.Length));
            offset += layer.Length;
        }

        return new Tensor<T>(flat, new[] { totalParams });
    }

    /// <summary>
    /// Identifies the honest majority cluster with HDBSCAN on cosine distances between client models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Honest clients' models point in similar directions, so they form one
    /// dense group; poisoned models sit apart from it. FLAME runs HDBSCAN with min_samples = 1 (the mutual
    /// reachability distance is then the plain distance, i.e. single linkage) and
    /// min_cluster_size = n/2 + 1 with a single cluster allowed.</para>
    /// <para>Because no split can leave two children of more than n/2 points, the condensed tree has only
    /// the root, and HDBSCAN labels as members the points that are still together when the cluster finally
    /// breaks into pieces smaller than min_cluster_size. In single-linkage terms that is the first connected
    /// component, as the distance threshold grows, that holds at least min_cluster_size models. A
    /// minimum admitted size of <c>MinClusterSize</c> is also enforced.</para>
    /// </remarks>
    /// <returns>Indices of the admitted clients in ascending order, or an empty array if none qualify.</returns>
    private int[] IdentifyHonestCluster(Tensor<T> models, int n)
    {
        int requiredSize = Math.Max(n / 2 + 1, _minClusterSize);
        if (requiredSize > n)
        {
            return Array.Empty<int>();
        }

        // Pairwise cosine distances d(i, j) = 1 - cos(W_i, W_j); a zero model is at distance 1 from all. The models
        // are normalized to unit rows first, so the Gram matrix holds the cosines directly and its dot products
        // cannot overflow narrow T the way W W^T and the norm products do.
        var units = Infrastructure.FederatedTensorMath<T>.NormalizeRows(models).Units;
        var cosine = Engine.TensorMatMul(units, Engine.TensorTranspose(units));
        var cosineDistances = Engine.TensorAddScalar(Engine.TensorNegate(cosine), NumOps.One);

        // Upper-triangle edges (i < j) in a fixed order.
        int edgeCount = n * (n - 1) / 2;
        var edgeU = new int[edgeCount];
        var edgeV = new int[edgeCount];
        var flatIndex = new int[edgeCount];
        int e = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                edgeU[e] = i;
                edgeV[e] = j;
                flatIndex[e] = i * n + j;
                e++;
            }
        }

        var weights = Engine.TensorGather(
            cosineDistances.Reshape(new[] { n * n }), new Tensor<int>(flatIndex, new[] { edgeCount }), 0);
        var (sortedWeights, order) = Engine.TensorSort(weights, 0, false);
        var edgeOrder = order.ToArray();

        // Kruskal: find the first edge whose union creates a component of the required size.
        var parent = Enumerable.Range(0, n).ToArray();
        var size = Enumerable.Repeat(1, n).ToArray();
        int criticalRank = -1;
        for (int r = 0; r < edgeCount && criticalRank < 0; r++)
        {
            int edge = edgeOrder[r];
            int merged = Union(parent, size, edgeU[edge], edgeV[edge]);
            if (merged >= requiredSize)
            {
                criticalRank = r;
            }
        }

        if (criticalRank < 0)
        {
            return Array.Empty<int>();
        }

        // Every edge at or below the critical distance joins the cluster at the same level.
        T criticalWeight = sortedWeights[criticalRank];
        int edgesAtOrBelow = edgeCount - Engine.TensorCountNonzero(Engine.TensorGreaterThan(weights, criticalWeight));
        parent = Enumerable.Range(0, n).ToArray();
        size = Enumerable.Repeat(1, n).ToArray();
        for (int r = 0; r < edgesAtOrBelow; r++)
        {
            int edge = edgeOrder[r];
            Union(parent, size, edgeU[edge], edgeV[edge]);
        }

        int root = Find(parent, edgeU[edgeOrder[criticalRank]]);
        return Enumerable.Range(0, n).Where(i => Find(parent, i) == root).ToArray();
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }

        return x;
    }

    /// <summary>Unions the sets of a and b and returns the size of the resulting set.</summary>
    private static int Union(int[] parent, int[] size, int a, int b)
    {
        int rootA = Find(parent, a);
        int rootB = Find(parent, b);
        if (rootA == rootB)
        {
            return size[rootA];
        }

        if (size[rootA] < size[rootB])
        {
            (rootA, rootB) = (rootB, rootA);
        }

        parent[rootB] = rootA;
        size[rootA] += size[rootB];
        return size[rootA];
    }

    /// <summary>Gets the noise multiplier for backdoor-erasure noise injection (not formal DP).</summary>
    public double NoiseMultiplier => _noiseMultiplier;

    /// <summary>Gets the minimum cluster size for HDBSCAN.</summary>
    public int MinClusterSize => _minClusterSize;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"FLAME(k={_minClusterSize},\u03c3={_noiseMultiplier})";
}
