namespace AiDotNet.FederatedLearning.Compression;

/// <summary>
/// Implements FedDT — Decision-tree-based compression for heterogeneous federated architectures.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Most gradient compression methods assume all clients have the same
/// model architecture. FedDT handles heterogeneous architectures by compressing model updates into
/// decision-tree representations. Each client distills its local model changes into a lightweight
/// decision tree, sends only the tree structure (much smaller than full gradients), and the server
/// merges the trees. This enables FL across different model architectures with minimal communication.</para>
///
/// <para>Algorithm:</para>
/// <code>
/// Client side:
///   1. Train local model, compute parameter delta
///   2. Partition parameters into bins (leaves of a decision tree)
///   3. Send tree: (split_thresholds, leaf_averages)  // much smaller than full delta
///
/// Server side:
///   1. Receive client trees
///   2. Merge by weighted averaging of leaf values at matching regions
///   3. Reconstruct approximate global update from merged tree
/// </code>
///
/// <para>Reference: FedDT: Decision-Tree Compression for Heterogeneous Federated Learning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class FedDTCompressor<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly int _maxTreeDepth;
    private readonly int _minLeafSize;
    private readonly double _pruningThreshold;

    /// <summary>
    /// Creates a new FedDT compressor.
    /// </summary>
    /// <param name="maxTreeDepth">Maximum depth of the compression tree. Deeper = more accurate but larger. Default: 8.</param>
    /// <param name="minLeafSize">Minimum number of parameters per leaf node. Default: 64.</param>
    /// <param name="pruningThreshold">Threshold below which leaf deltas are pruned to zero. Default: 1e-4.</param>
    public FedDTCompressor(
        int maxTreeDepth = 8,
        int minLeafSize = 64,
        double pruningThreshold = 1e-4)
    {
        if (maxTreeDepth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTreeDepth), "Tree depth must be positive.");
        }

        if (minLeafSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minLeafSize), "Minimum leaf size must be positive.");
        }

        if (pruningThreshold < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pruningThreshold), "Pruning threshold must be non-negative.");
        }

        _maxTreeDepth = maxTreeDepth;
        _minLeafSize = minLeafSize;
        _pruningThreshold = pruningThreshold;
    }

    /// <summary>
    /// Compresses a parameter update into a decision-tree representation.
    /// </summary>
    /// <param name="parameterDelta">The parameter update (current - previous).</param>
    /// <returns>Compressed tree representation: (splitPoints, leafValues, leafCounts).</returns>
    public CompressedTree Compress(Dictionary<string, T[]> parameterDelta)
    {
        Guard.NotNull(parameterDelta);
        var allSplits = new Dictionary<string, double[]>();
        var allLeafValues = new Dictionary<string, double[]>();
        var allLeafCounts = new Dictionary<string, int[]>();

        foreach (var (layerName, delta) in parameterDelta)
        {
            var values = new double[delta.Length];
            for (int i = 0; i < delta.Length; i++)
            {
                values[i] = NumOps.ToDouble(delta[i]);
            }

            // Build binary partition tree over parameter indices.
            var splits = new List<double>();
            var leafVals = new List<double>();
            var leafCounts = new List<int>();

            BuildTree(values, 0, values.Length, 0, splits, leafVals, leafCounts);

            allSplits[layerName] = splits.ToArray();
            allLeafValues[layerName] = leafVals.ToArray();
            allLeafCounts[layerName] = leafCounts.ToArray();
        }

        return new CompressedTree(allSplits, allLeafValues, allLeafCounts);
    }

    private void BuildTree(double[] values, int start, int end, int depth,
        List<double> splits, List<double> leafValues, List<int> leafCounts)
    {
        int count = end - start;

        // Compute variance to decide whether to split further.
        double sum = 0, sumSq = 0;
        for (int i = start; i < end; i++)
        {
            sum += values[i];
            sumSq += values[i] * values[i];
        }

        double mean = count > 0 ? sum / count : 0;
        double variance = count > 1 ? (sumSq / count - mean * mean) : 0;

        // Leaf conditions: max depth, min size, or variance below threshold.
        if (depth >= _maxTreeDepth || count <= _minLeafSize || variance < _pruningThreshold * _pruningThreshold)
        {
            // Prune near-zero leaves.
            if (Math.Abs(mean) < _pruningThreshold)
            {
                mean = 0;
            }

            leafValues.Add(mean);
            leafCounts.Add(count);
            return;
        }

        // Index-based split: always split at the midpoint of the index range.
        // This preserves the parameter-to-position mapping. Value-based partitioning
        // would rearrange elements in-place and destroy positional correspondence,
        // causing decompression to assign leaf means to wrong parameter positions.
        int mid = start + count / 2;
        splits.Add(mid);

        BuildTree(values, start, mid, depth + 1, splits, leafValues, leafCounts);
        BuildTree(values, mid, end, depth + 1, splits, leafValues, leafCounts);
    }

    /// <summary>
    /// Decompresses a tree representation back into parameter updates.
    /// </summary>
    /// <param name="tree">The compressed tree.</param>
    /// <param name="templateDelta">Template with correct layer names and sizes.</param>
    /// <returns>Reconstructed parameter update.</returns>
    public Dictionary<string, T[]> Decompress(CompressedTree tree, Dictionary<string, T[]> templateDelta)
    {
        Guard.NotNull(tree);
        Guard.NotNull(templateDelta);
        var result = new Dictionary<string, T[]>();

        foreach (var (layerName, template) in templateDelta)
        {
            var reconstructed = new T[template.Length];

            if (tree.LeafValues.TryGetValue(layerName, out var leafVals) &&
                tree.LeafCounts.TryGetValue(layerName, out var leafCounts))
            {
                int paramIdx = 0;
                for (int leaf = 0; leaf < leafVals.Length && paramIdx < template.Length; leaf++)
                {
                    int count = leafCounts[leaf];
                    double val = leafVals[leaf];
                    for (int i = 0; i < count && paramIdx < template.Length; i++)
                    {
                        reconstructed[paramIdx++] = NumOps.FromDouble(val);
                    }
                }

                // Fill remaining with zero.
                for (; paramIdx < template.Length; paramIdx++)
                {
                    reconstructed[paramIdx] = NumOps.Zero;
                }
            }
            else
            {
                for (int i = 0; i < template.Length; i++)
                {
                    reconstructed[i] = NumOps.Zero;
                }
            }

            result[layerName] = reconstructed;
        }

        return result;
    }

    /// <summary>
    /// Merges multiple compressed trees via weighted averaging of leaf values.
    /// </summary>
    /// <param name="clientTrees">Compressed trees from each client.</param>
    /// <param name="clientWeights">Aggregation weight per client.</param>
    /// <param name="templateDelta">Template for layer sizes.</param>
    /// <returns>Merged parameter update.</returns>
    public Dictionary<string, T[]> MergeTrees(
        Dictionary<int, CompressedTree> clientTrees,
        Dictionary<int, double> clientWeights,
        Dictionary<string, T[]> templateDelta)
    {
        Guard.NotNull(clientTrees);
        Guard.NotNull(clientWeights);
        Guard.NotNull(templateDelta);

        var result = new Dictionary<string, T[]>();
        double totalWeight = clientWeights.Values.Sum();

        // Decompress each client tree once for all layers, rather than per-layer.
        // This avoids L × C redundant decompressions and dictionary allocations.
        var decompressedClients = new Dictionary<int, Dictionary<string, T[]>>();
        foreach (var (clientId, tree) in clientTrees)
        {
            decompressedClients[clientId] = Decompress(tree, templateDelta);
        }

        foreach (var (layerName, template) in templateDelta)
        {
            var merged = new double[template.Length];

            foreach (var (clientId, decompressed) in decompressedClients)
            {
                double w = clientWeights.GetValueOrDefault(clientId, 1.0);

                if (decompressed.TryGetValue(layerName, out var layerVals))
                {
                    for (int i = 0; i < layerVals.Length; i++)
                    {
                        merged[i] += w * NumOps.ToDouble(layerVals[i]);
                    }
                }
            }

            var mergedT = new T[template.Length];
            for (int i = 0; i < template.Length; i++)
            {
                mergedT[i] = NumOps.FromDouble(totalWeight > 0 ? merged[i] / totalWeight : 0);
            }

            result[layerName] = mergedT;
        }

        return result;
    }

    /// <summary>
    /// Estimates the compression ratio achieved by the tree representation.
    /// </summary>
    /// <remarks>
    /// The estimate counts all elements equally regardless of type (doubles vs ints),
    /// which is a reasonable approximation for relative comparisons.
    /// </remarks>
    /// <param name="originalSize">Number of parameters in the original update.</param>
    /// <param name="tree">The compressed tree.</param>
    /// <returns>Compression ratio (original / compressed). Higher = more compression.</returns>
    public double EstimateCompressionRatio(int originalSize, CompressedTree tree)
    {
        Guard.NotNull(tree);
        int compressedSize = 0;
        foreach (var leafVals in tree.LeafValues.Values)
        {
            compressedSize += leafVals.Length; // leaf values
        }

        foreach (var splits in tree.SplitPoints.Values)
        {
            compressedSize += splits.Length; // split points
        }

        foreach (var counts in tree.LeafCounts.Values)
        {
            compressedSize += counts.Length; // leaf counts (stored as ints)
        }

        return compressedSize > 0 ? (double)originalSize / compressedSize : 1.0;
    }

    /// <summary>Gets the maximum tree depth.</summary>
    public int MaxTreeDepth => _maxTreeDepth;

    /// <summary>Gets the minimum leaf size.</summary>
    public int MinLeafSize => _minLeafSize;

    /// <summary>Gets the pruning threshold.</summary>
    public double PruningThreshold => _pruningThreshold;

    /// <summary>
    /// Represents a compressed decision-tree encoding of a parameter update.
    /// </summary>
    public sealed class CompressedTree
    {
        /// <summary>Creates a new compressed tree.</summary>
        public CompressedTree(
            Dictionary<string, double[]> splitPoints,
            Dictionary<string, double[]> leafValues,
            Dictionary<string, int[]> leafCounts)
        {
            Guard.NotNull(splitPoints);
            Guard.NotNull(leafValues);
            Guard.NotNull(leafCounts);
            SplitPoints = splitPoints;
            LeafValues = leafValues;
            LeafCounts = leafCounts;
        }

        /// <summary>Split thresholds per layer.</summary>
        public Dictionary<string, double[]> SplitPoints { get; }

        /// <summary>Leaf average values per layer.</summary>
        public Dictionary<string, double[]> LeafValues { get; }

        /// <summary>Number of parameters per leaf per layer.</summary>
        public Dictionary<string, int[]> LeafCounts { get; }
    }
}
