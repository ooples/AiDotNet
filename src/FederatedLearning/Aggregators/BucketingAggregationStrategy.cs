namespace AiDotNet.FederatedLearning.Aggregators;

/// <summary>
/// Implements the Bucketing meta-strategy for Byzantine-robust federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Bucketing is not an aggregation rule itself — it is a
/// "wrapper" that improves any existing robust aggregation rule. Before running the inner
/// aggregator, clients are randomly shuffled into equal-sized buckets. Within each bucket
/// the client updates are averaged, producing one "super-update" per bucket. The inner robust
/// aggregator then operates on these super-updates instead of individual client updates.</para>
///
/// <para>This provably increases the <em>breakdown point</em> (the fraction of adversaries
/// the defense can tolerate) for any sub-quadratic robust aggregator.</para>
///
/// <para>Algorithm:</para>
/// <list type="number">
/// <item>Randomly shuffle client IDs</item>
/// <item>Partition into <c>NumBuckets</c> equal-sized groups</item>
/// <item>Average updates within each bucket</item>
/// <item>Run the inner aggregation strategy on the bucket averages</item>
/// </list>
///
/// <para>Reference: Karimireddy, S. P., et al. (2022). "Byzantine-Robust Learning on
/// Heterogeneous Datasets via Bucketing." ICML 2022.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class BucketingAggregationStrategy<T> : ParameterDictionaryAggregationStrategyBase<T>
{
    private readonly ParameterDictionaryAggregationStrategyBase<T> _innerStrategy;
    private readonly int _numBuckets;
    private readonly int _seed;
    private int _roundCounter;

    /// <summary>
    /// Initializes a new instance of the <see cref="BucketingAggregationStrategy{T}"/> class.
    /// </summary>
    /// <param name="innerStrategy">The robust aggregation strategy to apply on bucket averages.</param>
    /// <param name="numBuckets">Number of buckets to partition clients into. Default: 3.</param>
    /// <param name="seed">Random seed for shuffling. Default: 42.</param>
    public BucketingAggregationStrategy(
        ParameterDictionaryAggregationStrategyBase<T> innerStrategy,
        int numBuckets = 3,
        int seed = 42)
    {
        _innerStrategy = innerStrategy ?? throw new ArgumentNullException(nameof(innerStrategy));

        if (numBuckets < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numBuckets), "Number of buckets must be at least 2.");
        }

        _numBuckets = numBuckets;
        _seed = seed;
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

        Guard.NotNull(clientWeights);

        if (clientModels.Count == 1)
        {
            var single = clientModels.First().Value;
            return single.ToDictionary(kv => kv.Key, kv => (T[])kv.Value.Clone());
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();

        // Validate all clients have matching layer structure before bucketing.
        foreach (var (clientId, model) in clientModels)
        {
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

        // Shuffle client IDs with round-varying seed to ensure different permutations each round.
        var clientIds = clientModels.Keys.ToList();
        var rng = new Random(_seed + _roundCounter);
        _roundCounter++;
        for (int i = clientIds.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (clientIds[i], clientIds[j]) = (clientIds[j], clientIds[i]);
        }

        // Partition into buckets.
        int effectiveBuckets = Math.Min(_numBuckets, clientIds.Count);
        var bucketModels = new Dictionary<int, Dictionary<string, T[]>>(effectiveBuckets);
        var bucketWeights = new Dictionary<int, double>(effectiveBuckets);

        int baseSize = clientIds.Count / effectiveBuckets;
        int remainder = clientIds.Count % effectiveBuckets;
        int offset = 0;

        for (int b = 0; b < effectiveBuckets; b++)
        {
            int bucketSize = baseSize + (b < remainder ? 1 : 0);
            if (bucketSize == 0)
            {
                continue;
            }

            // Average updates within this bucket.
            var bucketAvg = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
            foreach (var layerName in layerNames)
            {
                bucketAvg[layerName] = CreateZeroInitializedLayer(referenceModel[layerName].Length);
            }

            double totalBucketWeight = 0;
            for (int i = offset; i < offset + bucketSize; i++)
            {
                int clientId = clientIds[i];
                double w = clientWeights.TryGetValue(clientId, out var cw) ? cw : 1.0;
                totalBucketWeight += w;
                var clientModel = clientModels[clientId];

                foreach (var layerName in layerNames)
                {
                    var cp = clientModel[layerName];
                    var bp = bucketAvg[layerName];
                    for (int j = 0; j < bp.Length; j++)
                    {
                        bp[j] = NumOps.Add(bp[j], NumOps.Multiply(cp[j], NumOps.FromDouble(w)));
                    }
                }
            }

            // Normalize by total bucket weight.
            if (totalBucketWeight > 0)
            {
                var invWeight = NumOps.FromDouble(1.0 / totalBucketWeight);
                foreach (var layerName in layerNames)
                {
                    var bp = bucketAvg[layerName];
                    for (int j = 0; j < bp.Length; j++)
                    {
                        bp[j] = NumOps.Multiply(bp[j], invWeight);
                    }
                }
            }

            bucketModels[b] = bucketAvg;
            bucketWeights[b] = totalBucketWeight;
            offset += bucketSize;
        }

        // Delegate to the inner robust aggregation strategy.
        return _innerStrategy.Aggregate(bucketModels, bucketWeights);
    }

    /// <summary>Gets the number of buckets.</summary>
    public int NumBuckets => _numBuckets;

    /// <summary>Gets the inner robust aggregation strategy.</summary>
    public ParameterDictionaryAggregationStrategyBase<T> InnerStrategy => _innerStrategy;

    /// <inheritdoc/>
    public override string GetStrategyName() => $"Bucketing(k={_numBuckets},{_innerStrategy.GetStrategyName()})";
}
