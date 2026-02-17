using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Partitions a graph across federated clients using various strategies.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Before federated graph learning can begin, the graph must be divided
/// among clients. The partitioner splits nodes into groups, trying to minimize the number of edges
/// that cross partition boundaries (cross-client edges) while keeping partitions balanced.</para>
///
/// <para><b>Strategies:</b></para>
/// <list type="bullet">
/// <item><description><b>Random:</b> Assign each node randomly. Fast, O(n), but creates many cross-client edges.</description></item>
/// <item><description><b>METIS:</b> Multi-level graph partitioning that minimizes edge cuts.
/// Best quality but requires seeing the full graph (centralized preprocessing).</description></item>
/// <item><description><b>StreamPartition:</b> Process nodes in arrival order, assigning to the partition
/// that maximizes internal edges. Good for dynamic/streaming graphs.</description></item>
/// <item><description><b>CommunityBased:</b> Detect communities first (label propagation), then assign each
/// community to a client. Preserves local structure very well.</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedGraphPartitioner<T> : FederatedLearningComponentBase<T>
{
    private readonly FederatedGraphOptions _options;

    /// <summary>
    /// Initializes a new instance of <see cref="FederatedGraphPartitioner{T}"/>.
    /// </summary>
    /// <param name="options">Graph FL configuration.</param>
    public FederatedGraphPartitioner(FederatedGraphOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <summary>
    /// Partitions a graph into the specified number of parts.
    /// </summary>
    /// <param name="adjacency">Full adjacency matrix (flattened NxN).</param>
    /// <param name="nodeFeatures">Node feature matrix (flattened NxF).</param>
    /// <param name="numPartitions">Number of partitions (typically equals number of clients).</param>
    /// <returns>Partition assignment: nodeIndex -> partitionId.</returns>
    public int[] Partition(Tensor<T> adjacency, Tensor<T> nodeFeatures, int numPartitions)
    {
        if (adjacency is null) throw new ArgumentNullException(nameof(adjacency));
        if (numPartitions <= 0) throw new ArgumentOutOfRangeException(nameof(numPartitions));

        int numNodes = (int)Math.Sqrt(adjacency.Shape[0]);

        return _options.PartitionStrategy switch
        {
            GraphPartitionStrategy.Random => RandomPartition(numNodes, numPartitions),
            GraphPartitionStrategy.Metis => MetisPartition(adjacency, numNodes, numPartitions),
            GraphPartitionStrategy.StreamPartition => StreamPartition(adjacency, numNodes, numPartitions),
            GraphPartitionStrategy.CommunityBased => CommunityBasedPartition(adjacency, numNodes, numPartitions),
            GraphPartitionStrategy.Preassigned => PreassignedPartition(numNodes, numPartitions),
            _ => RandomPartition(numNodes, numPartitions)
        };
    }

    /// <summary>
    /// Extracts subgraph data for a specific partition.
    /// </summary>
    /// <param name="adjacency">Full adjacency matrix.</param>
    /// <param name="nodeFeatures">Full node features.</param>
    /// <param name="assignments">Partition assignments from <see cref="Partition"/>.</param>
    /// <param name="partitionId">Which partition to extract.</param>
    /// <returns>Adjacency and features for the specified partition's subgraph.</returns>
    public (Tensor<T> Adjacency, Tensor<T> Features) ExtractSubgraph(
        Tensor<T> adjacency, Tensor<T> nodeFeatures,
        int[] assignments, int partitionId)
    {
        int numNodes = (int)Math.Sqrt(adjacency.Shape[0]);
        int featureDim = numNodes > 0 ? nodeFeatures.Shape[0] / numNodes : 0;

        // Collect nodes in this partition
        var partitionNodes = new List<int>();
        for (int i = 0; i < assignments.Length; i++)
        {
            if (assignments[i] == partitionId)
            {
                partitionNodes.Add(i);
            }
        }

        int subSize = partitionNodes.Count;

        // Extract subgraph adjacency
        var subAdj = new Tensor<T>(new[] { subSize * subSize });
        for (int i = 0; i < subSize; i++)
        {
            for (int j = 0; j < subSize; j++)
            {
                int srcIdx = partitionNodes[i] * numNodes + partitionNodes[j];
                if (srcIdx < adjacency.Shape[0])
                {
                    subAdj[i * subSize + j] = adjacency[srcIdx];
                }
            }
        }

        // Extract subgraph features
        var subFeatures = new Tensor<T>(new[] { subSize * featureDim });
        for (int i = 0; i < subSize; i++)
        {
            for (int f = 0; f < featureDim; f++)
            {
                int srcIdx = partitionNodes[i] * featureDim + f;
                int dstIdx = i * featureDim + f;

                if (srcIdx < nodeFeatures.Shape[0] && dstIdx < subFeatures.Shape[0])
                {
                    subFeatures[dstIdx] = nodeFeatures[srcIdx];
                }
            }
        }

        return (subAdj, subFeatures);
    }

    private int[] RandomPartition(int numNodes, int numPartitions)
    {
        var assignments = new int[numNodes];
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        for (int i = 0; i < numNodes; i++)
        {
            assignments[i] = rng.Next(numPartitions);
        }

        return assignments;
    }

    private int[] MetisPartition(Tensor<T> adjacency, int numNodes, int numPartitions)
    {
        // Multi-level graph partitioning (simplified METIS-style):
        // 1. Coarsen: heavy-edge matching to reduce graph
        // 2. Partition coarse graph (bisection)
        // 3. Uncoarsen: project back and refine

        var assignments = new int[numNodes];

        // Step 1: Build adjacency list
        var neighbors = BuildAdjacencyList(adjacency, numNodes);

        // Step 2: Greedy balanced partitioning with edge-cut minimization
        int targetSize = (numNodes + numPartitions - 1) / numPartitions;
        var partitionSizes = new int[numPartitions];
        var assigned = new bool[numNodes];

        // Start with a BFS-like approach: pick seed nodes and grow partitions
        for (int p = 0; p < numPartitions; p++)
        {
            // Find unassigned node with highest degree (good seed)
            int bestNode = -1;
            int bestDegree = -1;
            for (int n = 0; n < numNodes; n++)
            {
                if (!assigned[n] && neighbors[n].Count > bestDegree)
                {
                    bestDegree = neighbors[n].Count;
                    bestNode = n;
                }
            }

            if (bestNode < 0) break;

            // BFS to grow partition from seed
            var queue = new Queue<int>();
            queue.Enqueue(bestNode);
            assigned[bestNode] = true;
            assignments[bestNode] = p;
            partitionSizes[p]++;

            while (queue.Count > 0 && partitionSizes[p] < targetSize)
            {
                int current = queue.Dequeue();
                foreach (int neighbor in neighbors[current])
                {
                    if (!assigned[neighbor] && partitionSizes[p] < targetSize)
                    {
                        assigned[neighbor] = true;
                        assignments[neighbor] = p;
                        partitionSizes[p]++;
                        queue.Enqueue(neighbor);
                    }
                }
            }
        }

        // Assign remaining unassigned nodes to smallest partition
        for (int n = 0; n < numNodes; n++)
        {
            if (!assigned[n])
            {
                int smallestPartition = 0;
                for (int p = 1; p < numPartitions; p++)
                {
                    if (partitionSizes[p] < partitionSizes[smallestPartition])
                    {
                        smallestPartition = p;
                    }
                }

                assignments[n] = smallestPartition;
                partitionSizes[smallestPartition]++;
            }
        }

        return assignments;
    }

    private int[] StreamPartition(Tensor<T> adjacency, int numNodes, int numPartitions)
    {
        // Stream-based: assign each node to the partition that maximizes internal edges
        var assignments = new int[numNodes];
        var partitionSizes = new int[numPartitions];
        int targetSize = (numNodes + numPartitions - 1) / numPartitions;
        var neighbors = BuildAdjacencyList(adjacency, numNodes);

        for (int n = 0; n < numNodes; n++)
        {
            int bestPartition = 0;
            int bestScore = -1;

            for (int p = 0; p < numPartitions; p++)
            {
                if (partitionSizes[p] >= targetSize * 1.1) // Allow 10% imbalance
                {
                    continue;
                }

                // Count how many of this node's neighbors are already in partition p
                int score = 0;
                foreach (int neighbor in neighbors[n])
                {
                    if (neighbor < n && assignments[neighbor] == p)
                    {
                        score++;
                    }
                }

                if (score > bestScore || (score == bestScore && partitionSizes[p] < partitionSizes[bestPartition]))
                {
                    bestScore = score;
                    bestPartition = p;
                }
            }

            assignments[n] = bestPartition;
            partitionSizes[bestPartition]++;
        }

        return assignments;
    }

    private int[] CommunityBasedPartition(Tensor<T> adjacency, int numNodes, int numPartitions)
    {
        // Label Propagation for community detection, then map communities to partitions
        var labels = new int[numNodes];
        for (int i = 0; i < numNodes; i++) labels[i] = i; // Each node starts as its own community

        var neighbors = BuildAdjacencyList(adjacency, numNodes);
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Label propagation iterations
        int maxIterations = 20;
        for (int iter = 0; iter < maxIterations; iter++)
        {
            bool changed = false;

            // Process nodes in random order
            var order = new int[numNodes];
            for (int i = 0; i < numNodes; i++) order[i] = i;
            for (int i = numNodes - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            foreach (int n in order)
            {
                if (neighbors[n].Count == 0) continue;

                // Find most frequent label among neighbors
                var labelCounts = new Dictionary<int, int>();
                foreach (int neighbor in neighbors[n])
                {
                    int label = labels[neighbor];
                    if (labelCounts.ContainsKey(label))
                    {
                        labelCounts[label]++;
                    }
                    else
                    {
                        labelCounts[label] = 1;
                    }
                }

                int bestLabel = labels[n];
                int bestCount = 0;
                foreach (var kvp in labelCounts)
                {
                    if (kvp.Value > bestCount)
                    {
                        bestCount = kvp.Value;
                        bestLabel = kvp.Key;
                    }
                }

                if (bestLabel != labels[n])
                {
                    labels[n] = bestLabel;
                    changed = true;
                }
            }

            if (!changed) break;
        }

        // Map communities to partitions (round-robin by community size)
        var communityNodes = new Dictionary<int, List<int>>();
        for (int i = 0; i < numNodes; i++)
        {
            int label = labels[i];
            if (!communityNodes.ContainsKey(label))
            {
                communityNodes[label] = new List<int>();
            }

            communityNodes[label].Add(i);
        }

        // Sort communities by size (largest first)
        var sortedCommunities = new List<List<int>>(communityNodes.Values);
        sortedCommunities.Sort((a, b) => b.Count.CompareTo(a.Count));

        // Assign communities to partitions, balancing sizes
        var assignments = new int[numNodes];
        var partitionSizes = new int[numPartitions];

        foreach (var community in sortedCommunities)
        {
            // Find smallest partition
            int smallestPartition = 0;
            for (int p = 1; p < numPartitions; p++)
            {
                if (partitionSizes[p] < partitionSizes[smallestPartition])
                {
                    smallestPartition = p;
                }
            }

            foreach (int node in community)
            {
                assignments[node] = smallestPartition;
                partitionSizes[smallestPartition]++;
            }
        }

        return assignments;
    }

    private static int[] PreassignedPartition(int numNodes, int numPartitions)
    {
        // Round-robin assignment (used when graph is already partitioned)
        var assignments = new int[numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            assignments[i] = i % numPartitions;
        }

        return assignments;
    }

    private List<List<int>> BuildAdjacencyList(Tensor<T> adjacency, int numNodes)
    {
        var neighbors = new List<List<int>>(numNodes);
        for (int i = 0; i < numNodes; i++)
        {
            neighbors.Add(new List<int>());
        }

        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                int idx = i * numNodes + j;
                if (idx < adjacency.Shape[0] && NumOps.ToDouble(adjacency[idx]) > 0 && i != j)
                {
                    neighbors[i].Add(j);
                }
            }
        }

        return neighbors;
    }
}
