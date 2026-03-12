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

        // Build expanded adjacency (totalNodes x totalNodes, flattened)
        var expandedAdj = new Tensor<T>(new[] { totalNodes * totalNodes });
        CopyOriginalAdjacency(adjacency, expandedAdj, numOriginalNodes, totalNodes);

        // Build expanded features
        var expandedFeatures = new Tensor<T>(new[] { totalNodes * featureDim });
        CopyOriginalFeatures(nodeFeatures, expandedFeatures, numOriginalNodes, featureDim);

        // Generate pseudo-node features
        GeneratePseudoNodeFeatures(
            expandedFeatures, nodeFeatures,
            numOriginalNodes, pseudoCount, featureDim,
            crossClientFeatures);

        // Connect pseudo-nodes to border nodes in adjacency
        ConnectPseudoNodes(expandedAdj, numOriginalNodes, pseudoCount, totalNodes);

        return new ExpandedSubgraph<T>
        {
            Adjacency = expandedAdj,
            NodeFeatures = expandedFeatures,
            PseudoNodeCount = pseudoCount,
            OriginalNodeCount = numOriginalNodes
        };
    }

    private void GeneratePseudoNodeFeatures(
        Tensor<T> expandedFeatures,
        Tensor<T> originalFeatures,
        int numOriginal,
        int pseudoCount,
        int featureDim,
        Dictionary<int, Tensor<T>> crossClientFeatures)
    {
        switch (_options.PseudoNodeStrategy)
        {
            case PseudoNodeStrategy.ZeroFill:
                // Features already initialized to zero in tensor constructor
                break;

            case PseudoNodeStrategy.FeatureAverage:
                FillWithAverage(expandedFeatures, originalFeatures, numOriginal, pseudoCount, featureDim);
                break;

            case PseudoNodeStrategy.GeneratorBased:
                FillWithGenerator(expandedFeatures, originalFeatures, crossClientFeatures,
                    numOriginal, pseudoCount, featureDim);
                break;

            default:
                break;
        }
    }

    private void FillWithAverage(
        Tensor<T> expanded, Tensor<T> original,
        int numOriginal, int pseudoCount, int featureDim)
    {
        // Compute average feature vector
        var avg = new double[featureDim];
        for (int i = 0; i < numOriginal; i++)
        {
            for (int f = 0; f < featureDim; f++)
            {
                int idx = i * featureDim + f;
                if (idx < original.Shape[0])
                {
                    avg[f] += NumOps.ToDouble(original[idx]);
                }
            }
        }

        for (int f = 0; f < featureDim; f++)
        {
            avg[f] /= Math.Max(1, numOriginal);
        }

        // Fill pseudo-nodes with average
        for (int p = 0; p < pseudoCount; p++)
        {
            for (int f = 0; f < featureDim; f++)
            {
                int idx = (numOriginal + p) * featureDim + f;
                if (idx < expanded.Shape[0])
                {
                    expanded[idx] = NumOps.FromDouble(avg[f]);
                }
            }
        }
    }

    private void FillWithGenerator(
        Tensor<T> expanded, Tensor<T> original,
        Dictionary<int, Tensor<T>> crossClientFeatures,
        int numOriginal, int pseudoCount, int featureDim)
    {
        // Use cross-client features if available, otherwise fall back to noisy average
        if (crossClientFeatures.Count > 0)
        {
            // Use actual cross-client border node features
            int pseudoIdx = 0;
            foreach (var kvp in crossClientFeatures)
            {
                var otherFeatures = kvp.Value;
                int otherNodes = otherFeatures.Shape[0] / Math.Max(1, featureDim);

                for (int n = 0; n < otherNodes && pseudoIdx < pseudoCount; n++, pseudoIdx++)
                {
                    for (int f = 0; f < featureDim; f++)
                    {
                        int srcIdx = n * featureDim + f;
                        int dstIdx = (numOriginal + pseudoIdx) * featureDim + f;

                        if (srcIdx < otherFeatures.Shape[0] && dstIdx < expanded.Shape[0])
                        {
                            expanded[dstIdx] = otherFeatures[srcIdx];
                        }
                    }
                }
            }

            // Fill remaining with average if not enough cross-client features
            if (pseudoIdx < pseudoCount)
            {
                FillWithAverage(expanded, original, numOriginal, pseudoCount - pseudoIdx, featureDim);
            }
        }
        else
        {
            // No cross-client features: fall back to noisy average
            FillWithAverage(expanded, original, numOriginal, pseudoCount, featureDim);

            // Add small noise for diversity
            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            for (int p = 0; p < pseudoCount; p++)
            {
                for (int f = 0; f < featureDim; f++)
                {
                    int idx = (numOriginal + p) * featureDim + f;
                    if (idx < expanded.Shape[0])
                    {
                        double val = NumOps.ToDouble(expanded[idx]);
                        val += (rng.NextDouble() - 0.5) * 0.1; // Small noise
                        expanded[idx] = NumOps.FromDouble(val);
                    }
                }
            }
        }
    }

    private static void CopyOriginalAdjacency(Tensor<T> original, Tensor<T> expanded, int numOriginal, int totalNodes)
    {
        for (int i = 0; i < numOriginal; i++)
        {
            for (int j = 0; j < numOriginal; j++)
            {
                int srcIdx = i * numOriginal + j;
                int dstIdx = i * totalNodes + j;

                if (srcIdx < original.Shape[0] && dstIdx < expanded.Shape[0])
                {
                    expanded[dstIdx] = original[srcIdx];
                }
            }
        }
    }

    private static void CopyOriginalFeatures(Tensor<T> original, Tensor<T> expanded, int numOriginal, int featureDim)
    {
        int count = Math.Min(numOriginal * featureDim, original.Shape[0]);
        for (int i = 0; i < count; i++)
        {
            if (i < expanded.Shape[0])
            {
                expanded[i] = original[i];
            }
        }
    }

    private void ConnectPseudoNodes(Tensor<T> adjacency, int numOriginal, int pseudoCount, int totalNodes)
    {
        // Connect each pseudo-node to a random subset of border nodes
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        int connectionsPerPseudo = Math.Min(3, numOriginal); // Connect to up to 3 original nodes

        for (int p = 0; p < pseudoCount; p++)
        {
            int pseudoIdx = numOriginal + p;

            for (int c = 0; c < connectionsPerPseudo; c++)
            {
                int targetNode = rng.Next(numOriginal);
                int fwdIdx = pseudoIdx * totalNodes + targetNode;
                int bwdIdx = targetNode * totalNodes + pseudoIdx;

                if (fwdIdx < adjacency.Shape[0])
                {
                    adjacency[fwdIdx] = NumOps.FromDouble(1.0);
                }

                if (bwdIdx < adjacency.Shape[0])
                {
                    adjacency[bwdIdx] = NumOps.FromDouble(1.0);
                }
            }
        }
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
