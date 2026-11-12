using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
        var allSplits = new Dictionary<string, int[]>();
        var allLeafValues = new Dictionary<string, T[]>();
        var allLeafCounts = new Dictionary<string, int[]>();

        foreach (var (layerName, delta) in parameterDelta)
        {
            if (delta == null || delta.Length == 0)
            {
                throw new ArgumentException(
                    $"Layer '{layerName}' has null or empty parameter delta.", nameof(parameterDelta));
            }

            // Build binary partition tree over parameter indices.
            var (splits, leafValues, leafCounts) = BuildTree(delta);
            allSplits[layerName] = splits;
            allLeafValues[layerName] = leafValues;
            allLeafCounts[layerName] = leafCounts;
        }

        return new CompressedTree(allSplits, allLeafValues, allLeafCounts);
    }

    /// <summary>
    /// Builds the index-midpoint partition tree of one layer.
    /// </summary>
    /// <remarks>
    /// <para>A node over [start, end) becomes a leaf at the maximum depth, when it holds at most
    /// <c>MinLeafSize</c> parameters, or when the standard deviation of its values is below the pruning
    /// threshold; otherwise it splits at the index midpoint (which preserves the parameter-to-position
    /// mapping). Leaf means below the pruning threshold in magnitude are pruned to zero.</para>
    /// <para>Because splits are positional, all nodes of one depth are evaluated together: one segment
    /// index-add gives their sums, and a second over the squared deviations gives their variances.
    /// Leaves are returned left to right and split points in pre-order.</para>
    /// </remarks>
    private (int[] Splits, T[] LeafValues, int[] LeafCounts) BuildTree(T[] delta)
    {
        int n = delta.Length;
        var values = new Tensor<T>(delta, new[] { n });
        var threshold = NumOps.FromDouble(_pruningThreshold);

        var active = new List<(int Start, int End)> { (0, n) };
        var splitNodes = new List<(int Start, int Depth, int Mid)>();
        var leafNodes = new List<(int Start, int Count, int MeanIndex)>();
        var depthMeans = new List<Tensor<T>>();
        int meanOffset = 0;

        for (int depth = 0; active.Count > 0; depth++)
        {
            int segments = active.Count;

            // Element -> active segment; elements already in leaves go to the spare bucket `segments`.
            var segmentOf = new int[n];
            segmentOf.AsSpan().Fill(segments);
            var counts = new int[segments];
            for (int s = 0; s < segments; s++)
            {
                var (start, end) = active[s];
                segmentOf.AsSpan(start, end - start).Fill(s);
                counts[s] = end - start;
            }

            var segmentIndex = new Tensor<int>(segmentOf, new[] { n });
            var countsT = new Tensor<int>(counts, new[] { segments }).Cast<T>();

            // Segment means, then the (two-pass) segment variances.
            var sums = Engine.TensorIndexAdd(new Tensor<T>(new[] { segments + 1 }), 0, segmentIndex, values);
            var means = Engine.TensorDivide(Engine.TensorSlice(sums, new[] { 0 }, new[] { segments }), countsT);
            var meanPerElement = Engine.TensorGather(
                Engine.TensorConcatenate(new[] { means, new Tensor<T>(new[] { 1 }) }, 0), segmentIndex, 0);
            var squaredDeviation = Engine.TensorSquare(Engine.TensorSubtract(values, meanPerElement));
            var squaredSums = Engine.TensorIndexAdd(new Tensor<T>(new[] { segments + 1 }), 0, segmentIndex, squaredDeviation);
            var stdDev = Engine.TensorSqrt(
                Engine.TensorDivide(Engine.TensorSlice(squaredSums, new[] { 0 }, new[] { segments }), countsT));
            var smallSpread = Engine.TensorLessThan(stdDev, threshold).Cast<int>().ToArray();

            // Prune near-zero leaf means.
            depthMeans.Add(Engine.TensorWhere(
                Engine.TensorLessThan(Engine.TensorAbs(means), threshold), new Tensor<T>(new[] { segments }), means));

            var next = new List<(int Start, int End)>();
            for (int s = 0; s < segments; s++)
            {
                var (start, end) = active[s];
                int count = counts[s];
                if (depth >= _maxTreeDepth || count <= _minLeafSize || smallSpread[s] != 0)
                {
                    leafNodes.Add((start, count, meanOffset + s));
                    continue;
                }

                int mid = start + count / 2;
                splitNodes.Add((start, depth, mid));
                next.Add((start, mid));
                next.Add((mid, end));
            }

            meanOffset += segments;
            active = next;
        }

        var orderedLeaves = leafNodes.OrderBy(l => l.Start).ToArray();
        var meanIndex = new Tensor<int>(orderedLeaves.Select(l => l.MeanIndex).ToArray(), new[] { orderedLeaves.Length });
        var leafValues = Engine.TensorGather(Engine.TensorConcatenate(depthMeans.ToArray(), 0), meanIndex, 0).ToArray();
        var splits = splitNodes.OrderBy(s => s.Start).ThenBy(s => s.Depth).Select(s => s.Mid).ToArray();
        return (splits, leafValues, orderedLeaves.Select(l => l.Count).ToArray());
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
            int length = template.Length;
            if (tree.LeafValues.TryGetValue(layerName, out var leafVals) &&
                tree.LeafCounts.TryGetValue(layerName, out var leafCounts))
            {
                // Position -> leaf; positions past the last leaf read the extra zero entry.
                int leaves = Math.Min(leafVals.Length, leafCounts.Length);
                var leafOf = new int[length];
                leafOf.AsSpan().Fill(leaves);
                int paramIdx = 0;
                for (int leaf = 0; leaf < leaves && paramIdx < length; leaf++)
                {
                    int count = Math.Min(Math.Max(leafCounts[leaf], 0), length - paramIdx);
                    leafOf.AsSpan(paramIdx, count).Fill(leaf);
                    paramIdx += count;
                }

                var source = new T[leaves + 1];
                Array.Copy(leafVals, source, leaves);
                source[leaves] = NumOps.Zero;
                result[layerName] = Engine.TensorGather(
                    new Tensor<T>(source, new[] { leaves + 1 }), new Tensor<int>(leafOf, new[] { length }), 0).ToArray();
            }
            else
            {
                result[layerName] = new Tensor<T>(new[] { length }).ToArray();
            }
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
            var merged = new Tensor<T>(new[] { template.Length });

            foreach (var (clientId, decompressed) in decompressedClients)
            {
                double w = clientWeights.GetValueOrDefault(clientId, 1.0);

                if (decompressed.TryGetValue(layerName, out var layerVals))
                {
                    merged = Engine.TensorAdd(
                        merged,
                        Engine.TensorMultiplyScalar(new Tensor<T>(layerVals, new[] { layerVals.Length }), NumOps.FromDouble(w)));
                }
            }

            result[layerName] = totalWeight > 0
                ? Engine.TensorDivideScalar(merged, NumOps.FromDouble(totalWeight)).ToArray()
                : new Tensor<T>(new[] { template.Length }).ToArray();
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
            Dictionary<string, int[]> splitPoints,
            Dictionary<string, T[]> leafValues,
            Dictionary<string, int[]> leafCounts)
        {
            Guard.NotNull(splitPoints);
            Guard.NotNull(leafValues);
            Guard.NotNull(leafCounts);
            SplitPoints = splitPoints;
            LeafValues = leafValues;
            LeafCounts = leafCounts;
        }

        /// <summary>Split positions (parameter indices, in pre-order) per layer.</summary>
        public Dictionary<string, int[]> SplitPoints { get; }

        /// <summary>Leaf average values per layer, left to right.</summary>
        public Dictionary<string, T[]> LeafValues { get; }

        /// <summary>Number of parameters per leaf per layer.</summary>
        public Dictionary<string, int[]> LeafCounts { get; }
    }
}
