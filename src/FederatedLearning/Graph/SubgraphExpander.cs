using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Expands a local subgraph with pseudo-nodes to approximate missing cross-client neighbors.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a graph is split across clients, GNNs on each client "see" only
/// their local subgraph. Nodes near the boundary are missing their cross-client neighbors, which
/// degrades message passing. The SubgraphExpander adds "pseudo-nodes" to fill these gaps:</para>
///
/// <list type="bullet">
/// <item><description><b>ZeroFill:</b> Add zero-vector nodes. Fast but loses information.</description></item>
/// <item><description><b>FeatureAverage:</b> Add nodes with the average feature vector of known nodes.
/// Good approximation for homophilic graphs (neighbors tend to be similar).</description></item>
/// <item><description><b>GeneratorBased:</b> Use a learned generator to produce realistic features.
/// Best quality but requires a pre-trained generator model.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class SubgraphExpander<T> : FederatedLearningComponentBase<T>
{
    private readonly FederatedGraphOptions _options;

    /// <summary>
    /// Initializes a new instance of <see cref="SubgraphExpander{T}"/>.
    /// </summary>
    /// <param name="options">Graph FL configuration.</param>
    public SubgraphExpander(FederatedGraphOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Expands a local subgraph with pseudo-nodes for missing cross-client neighbors.
    /// </summary>
    /// <param name="adjacency">Local adjacency matrix (flattened, NxN).</param>
    /// <param name="nodeFeatures">Local node feature matrix (flattened, NxF or N elements).</param>
    /// <param name="crossClientFeatures">Available feature data from other clients' border nodes.</param>
    /// <returns>Expanded subgraph with pseudo-nodes added.</returns>
    public ExpandedSubgraph<T> Expand(
        Tensor<T> adjacency,
        Tensor<T> nodeFeatures,
        Dictionary<int, Tensor<T>> crossClientFeatures)
    {
        if (_options.PseudoNodeStrategy == PseudoNodeStrategy.None)
        {
            return new ExpandedSubgraph<T>
            {
                Adjacency = adjacency,
                NodeFeatures = nodeFeatures,
                PseudoNodeCount = 0,
                OriginalNodeCount = EstimateNodeCount(adjacency)
            };
        }

        int numOriginalNodes = EstimateNodeCount(adjacency);
        int featureDim = EstimateFeatureDim(nodeFeatures, numOriginalNodes);

        // Determine how many pseudo-nodes to add
        int pseudoCount = EstimatePseudoNodeCount(crossClientFeatures, numOriginalNodes);

        if (pseudoCount == 0)
        {
            return new ExpandedSubgraph<T>
            {
                Adjacency = adjacency,
                NodeFeatures = nodeFeatures,
                PseudoNodeCount = 0,
                OriginalNodeCount = numOriginalNodes
            };
        }

        int totalNodes = numOriginalNodes + pseudoCount;

        // Build expanded adjacency (totalNodes x totalNodes, flattened): the original block in the top-left
        // corner, zeros elsewhere.
        var originalAdj = FitToLength(adjacency, numOriginalNodes * numOriginalNodes)
            .Reshape(new[] { numOriginalNodes, numOriginalNodes });
        var expandedAdj = Engine.TensorConcatenate(new[]
        {
            Engine.TensorConcatenate(new[] { originalAdj, new Tensor<T>(new[] { numOriginalNodes, pseudoCount }) }, 1),
            new Tensor<T>(new[] { pseudoCount, totalNodes }),
        }, 0).Reshape(new[] { totalNodes * totalNodes });

        // Build expanded features: original node rows followed by the generated pseudo-node rows
        var originalFeatures = FitToLength(nodeFeatures, numOriginalNodes * featureDim);
        var pseudoFeatures = GeneratePseudoNodeFeatures(originalFeatures, numOriginalNodes, pseudoCount, featureDim, crossClientFeatures);
        var expandedFeatures = Engine.TensorConcatenate(new[] { originalFeatures, pseudoFeatures }, 0);

        // Connect pseudo-nodes to border nodes in adjacency
        expandedAdj = ConnectPseudoNodes(expandedAdj, numOriginalNodes, pseudoCount, totalNodes);

        return new ExpandedSubgraph<T>
        {
            Adjacency = expandedAdj,
            NodeFeatures = expandedFeatures,
            PseudoNodeCount = pseudoCount,
            OriginalNodeCount = numOriginalNodes
        };
    }

    /// <summary>
    /// Builds the flattened [pseudoCount * featureDim] features of the pseudo-nodes.
    /// </summary>
    private Tensor<T> GeneratePseudoNodeFeatures(
        Tensor<T> originalFeatures,
        int numOriginal,
        int pseudoCount,
        int featureDim,
        Dictionary<int, Tensor<T>> crossClientFeatures)
    {
        switch (_options.PseudoNodeStrategy)
        {
            case PseudoNodeStrategy.FeatureAverage:
                return RepeatAverage(originalFeatures, numOriginal, pseudoCount, featureDim);

            case PseudoNodeStrategy.GeneratorBased:
                return FillWithGenerator(originalFeatures, crossClientFeatures, numOriginal, pseudoCount, featureDim);

            default:
                // ZeroFill: pseudo-node features stay zero
                return new Tensor<T>(new[] { pseudoCount * featureDim });
        }
    }

    /// <summary>
    /// <paramref name="rows"/> copies of the mean original feature vector, flattened.
    /// </summary>
    private Tensor<T> RepeatAverage(Tensor<T> originalFeatures, int numOriginal, int rows, int featureDim)
    {
        if (rows == 0 || featureDim == 0)
        {
            return new Tensor<T>(new[] { rows * featureDim });
        }

        // Compute average feature vector
        var avg = numOriginal == 0
            ? new Tensor<T>(new[] { 1, featureDim })
            : Engine.ReduceMean(originalFeatures.Reshape(new[] { numOriginal, featureDim }), new[] { 0 }, true);
        return Engine.TensorBroadcastTo(avg, new[] { rows, featureDim }).Reshape(new[] { rows * featureDim });
    }

    private Tensor<T> FillWithGenerator(
        Tensor<T> originalFeatures,
        Dictionary<int, Tensor<T>> crossClientFeatures,
        int numOriginal, int pseudoCount, int featureDim)
    {
        // Use cross-client features if available, otherwise fall back to noisy average
        if (crossClientFeatures.Count > 0)
        {
            // Use actual cross-client border node features, in client order, up to pseudoCount rows
            var parts = new List<Tensor<T>>();
            int filled = 0;
            foreach (var kvp in crossClientFeatures)
            {
                if (filled >= pseudoCount) break;

                var otherFeatures = kvp.Value;
                int otherNodes = otherFeatures.Length / Math.Max(1, featureDim);
                int take = Math.Min(otherNodes, pseudoCount - filled);
                if (take > 0 && featureDim > 0)
                {
                    parts.Add(FitToLength(otherFeatures, take * featureDim));
                }

                filled += take;
            }

            // Fill the remaining pseudo-nodes (after the copied ones) with the average
            if (filled < pseudoCount)
            {
                parts.Add(RepeatAverage(originalFeatures, numOriginal, pseudoCount - filled, featureDim));
            }

            return parts.Count == 0 ? new Tensor<T>(new[] { 0 }) : Engine.TensorConcatenate(parts.ToArray(), 0);
        }

        // No cross-client features: fall back to noisy average
        var average = RepeatAverage(originalFeatures, numOriginal, pseudoCount, featureDim);

        // Add small noise for diversity
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var noise = new double[pseudoCount * featureDim];
        for (int k = 0; k < noise.Length; k++)
        {
            noise[k] = (rng.NextDouble() - 0.5) * 0.1; // Small noise
        }

        return Engine.TensorAdd(average, new Tensor<double>(noise, new[] { noise.Length }).Cast<T>());
    }

    /// <summary>
    /// Copies the first <paramref name="length"/> entries of a flattened tensor, zero-padding short ones.
    /// </summary>
    private Tensor<T> FitToLength(Tensor<T> source, int length)
    {
        var flat = source.Reshape(new[] { source.Length });
        int covered = Math.Min(length, source.Length);
        if (covered == length)
        {
            return covered == source.Length ? flat : Engine.TensorSlice(flat, new[] { 0 }, new[] { length });
        }

        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { length - covered }) }, 0);
    }

    private Tensor<T> ConnectPseudoNodes(Tensor<T> adjacency, int numOriginal, int pseudoCount, int totalNodes)
    {
        // Connect each pseudo-node to a random subset of border nodes
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        int connectionsPerPseudo = Math.Min(3, numOriginal); // Connect to up to 3 original nodes
        var connected = new bool[totalNodes * totalNodes];

        for (int p = 0; p < pseudoCount; p++)
        {
            int pseudoIdx = numOriginal + p;

            for (int c = 0; c < connectionsPerPseudo; c++)
            {
                int targetNode = rng.Next(numOriginal);
                connected[pseudoIdx * totalNodes + targetNode] = true;
                connected[targetNode * totalNodes + pseudoIdx] = true;
            }
        }

        return Engine.TensorWhere(connected, Engine.TensorFullLike(adjacency, NumOps.One), adjacency);
    }

    private int EstimatePseudoNodeCount(Dictionary<int, Tensor<T>> crossClientFeatures, int originalNodes)
    {
        if (_options.PseudoNodeStrategy == PseudoNodeStrategy.None)
        {
            return 0;
        }

        // Add pseudo-nodes proportional to border nodes (estimated as 20% of original nodes)
        int borderEstimate = Math.Max(1, originalNodes / 5);
        return Math.Min(borderEstimate, originalNodes); // Don't more than double the graph
    }

    private static int EstimateNodeCount(Tensor<T> adjacency)
    {
        return (int)Math.Sqrt(adjacency.Shape[0]);
    }

    private static int EstimateFeatureDim(Tensor<T> features, int numNodes)
    {
        if (features.Shape.Length > 1)
        {
            return features.Shape[1];
        }

        return numNodes > 0 ? features.Shape[0] / numNodes : features.Shape[0];
    }
}

/// <summary>
/// Represents a subgraph expanded with pseudo-nodes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExpandedSubgraph<T>
{
    /// <summary>Expanded adjacency matrix (flattened).</summary>
    public Tensor<T> Adjacency { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>Expanded node features (flattened, [totalNodes * featureDim]).</summary>
    public Tensor<T> NodeFeatures { get; set; } = new Tensor<T>(new[] { 0 });

    /// <summary>Number of pseudo-nodes added.</summary>
    public int PseudoNodeCount { get; set; }

    /// <summary>Number of original nodes.</summary>
    public int OriginalNodeCount { get; set; }
}
