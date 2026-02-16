using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

/// <summary>
/// Implements the Leiden algorithm for community detection in knowledge graphs.
/// </summary>
/// <typeparam name="T">The numeric type used for graph operations.</typeparam>
/// <remarks>
/// <para>
/// The Leiden algorithm (Traag et al., 2019) improves on the Louvain algorithm by guaranteeing
/// well-connected communities. It consists of three phases repeated iteratively:
/// 1. Local moving: greedily move nodes to maximize modularity
/// 2. Refinement: ensure communities are internally connected
/// 3. Aggregation: merge communities into super-nodes and repeat
/// </para>
/// <para><b>For Beginners:</b> Community detection finds groups of nodes that are more connected
/// to each other than to the rest of the graph.
///
/// Think of a school: students naturally form groups (sports team, drama club, study group).
/// The Leiden algorithm automatically discovers these groups by looking at who connects to whom.
///
/// It produces a hierarchy:
/// - Fine level: Small friend groups
/// - Coarser level: Clubs and teams
/// - Coarsest level: Grade levels or departments
/// </para>
/// </remarks>
public class LeidenCommunityDetector<T>
{
    /// <summary>
    /// Runs the Leiden algorithm on the given knowledge graph.
    /// </summary>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <param name="options">Algorithm options (resolution, iterations, seed).</param>
    /// <returns>Community detection results with hierarchical partitions.</returns>
    public LeidenResult Detect(KnowledgeGraph<T> graph, LeidenOptions? options = null)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));

        var opts = options ?? new LeidenOptions();
        var rng = opts.Seed.HasValue ? new Random(opts.Seed.Value) : new Random();
        double resolution = opts.GetEffectiveResolution();
        int maxIterations = opts.GetEffectiveMaxIterations();

        // Build adjacency structure
        var nodes = graph.GetAllNodes().Select(n => n.Id).ToList();
        if (nodes.Count == 0)
            return new LeidenResult();

        var adjacency = BuildAdjacency(graph, nodes);
        double totalWeight = adjacency.Values.SelectMany(v => v.Values).Sum() / 2.0;
        if (totalWeight < 1e-12) totalWeight = 1.0;

        // Initialize: each node in its own community
        var partition = new Dictionary<string, int>();
        for (int i = 0; i < nodes.Count; i++)
            partition[nodes[i]] = i;

        var hierarchicalPartitions = new List<Dictionary<string, int>>();
        var modularityScores = new List<double>();

        // Current-level node list and adjacency (initially the original graph)
        var currentNodes = new List<string>(nodes);
        var currentAdj = adjacency;
        double currentTotalWeight = totalWeight;

        // Tracks cumulative mapping: for each original node, which super-node it belongs to
        // at the current aggregation level. Initially each node maps to itself.
        var originalToCurrentNode = new Dictionary<string, string>(nodes.Count);
        foreach (var nodeId in nodes)
            originalToCurrentNode[nodeId] = nodeId;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Phase 1: Local moving
            bool moved = LocalMoving(currentNodes, currentAdj, partition, resolution, currentTotalWeight, rng);

            // Phase 2: Refinement — ensure connectivity within communities
            RefinePartition(currentNodes, currentAdj, partition, resolution, currentTotalWeight, rng);

            // Map partition back to original node IDs so all levels are consistent
            var levelPartition = new Dictionary<string, int>();
            foreach (var originalNodeId in nodes)
            {
                var currentNodeId = originalToCurrentNode[originalNodeId];
                if (partition.TryGetValue(currentNodeId, out int comm))
                    levelPartition[originalNodeId] = comm;
            }

            hierarchicalPartitions.Add(levelPartition);
            modularityScores.Add(ComputeModularity(currentNodes, currentAdj, partition, resolution, currentTotalWeight));

            if (!moved) break;

            // Phase 3: Aggregation — build super-node graph
            var (superNodes, superAdj, nodeMapping) = AggregateGraph(currentNodes, currentAdj, partition);

            if (superNodes.Count >= currentNodes.Count) break; // No further coarsening possible

            // Update cumulative mapping: original → new super-node
            foreach (var originalNodeId in nodes)
            {
                var currentNodeId = originalToCurrentNode[originalNodeId];
                if (nodeMapping.TryGetValue(currentNodeId, out var superNodeId))
                    originalToCurrentNode[originalNodeId] = superNodeId;
            }

            // Update partition for super-node level
            currentNodes = superNodes;
            currentAdj = superAdj;
            partition = [];
            for (int i = 0; i < superNodes.Count; i++)
                partition[superNodes[i]] = i;
        }

        // Finest-level partition is always the first (level 0)
        var finalPartition = hierarchicalPartitions.Count > 0
            ? hierarchicalPartitions[0]
            : new Dictionary<string, int>(partition);

        return new LeidenResult
        {
            HierarchicalPartitions = hierarchicalPartitions,
            Communities = finalPartition,
            ModularityScores = modularityScores
        };
    }

    private static Dictionary<string, Dictionary<string, double>> BuildAdjacency(
        KnowledgeGraph<T> graph, List<string> nodes)
    {
        var adj = new Dictionary<string, Dictionary<string, double>>();
        foreach (var nodeId in nodes)
            adj[nodeId] = [];

        foreach (var edge in graph.GetAllEdges())
        {
            if (!adj.ContainsKey(edge.SourceId) || !adj.ContainsKey(edge.TargetId))
                continue;

            // Undirected: add both directions
            adj[edge.SourceId].TryGetValue(edge.TargetId, out double w1);
            adj[edge.SourceId][edge.TargetId] = w1 + edge.Weight;

            adj[edge.TargetId].TryGetValue(edge.SourceId, out double w2);
            adj[edge.TargetId][edge.SourceId] = w2 + edge.Weight;
        }

        return adj;
    }

    private static bool LocalMoving(
        List<string> nodes,
        Dictionary<string, Dictionary<string, double>> adjacency,
        Dictionary<string, int> partition,
        double resolution, double totalWeight, Random rng)
    {
        bool anyMoved = false;
        var nodeOrder = new List<string>(nodes);
        Shuffle(nodeOrder, rng);

        // Precompute node strengths and community strength aggregates
        var nodeStrengths = new Dictionary<string, double>(nodes.Count);
        var communityStrengths = new Dictionary<int, double>();

        foreach (var nodeId in nodes)
        {
            double strength = adjacency.ContainsKey(nodeId) ? adjacency[nodeId].Values.Sum() : 0.0;
            nodeStrengths[nodeId] = strength;

            int comm = partition[nodeId];
            communityStrengths.TryGetValue(comm, out double cs);
            communityStrengths[comm] = cs + strength;
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            foreach (var nodeId in nodeOrder)
            {
                if (!adjacency.ContainsKey(nodeId)) continue;

                int currentCommunity = partition[nodeId];
                double bestDelta = 0.0;
                int bestCommunity = currentCommunity;
                double nodeStrength = nodeStrengths[nodeId];

                // Compute weights to each neighboring community
                var communityWeights = new Dictionary<int, double>();
                foreach (var (neighbor, weight) in adjacency[nodeId])
                {
                    int neighborComm = partition[neighbor];
                    communityWeights.TryGetValue(neighborComm, out double cw);
                    communityWeights[neighborComm] = cw + weight;
                }

                // Community strength excluding this node
                double currentCommStrengthWithout = communityStrengths.GetValueOrDefault(currentCommunity) - nodeStrength;
                communityWeights.TryGetValue(currentCommunity, out double weightToCurrent);

                // Try moving to each neighboring community
                foreach (var (comm, weightToComm) in communityWeights)
                {
                    if (comm == currentCommunity) continue;

                    double commStrength = communityStrengths.GetValueOrDefault(comm);

                    double delta = (weightToComm - weightToCurrent) / totalWeight
                                   - resolution * nodeStrength * (commStrength - currentCommStrengthWithout) / (2.0 * totalWeight * totalWeight);

                    if (delta > bestDelta)
                    {
                        bestDelta = delta;
                        bestCommunity = comm;
                    }
                }

                if (bestCommunity != currentCommunity)
                {
                    // Update community strengths incrementally
                    communityStrengths[currentCommunity] = communityStrengths.GetValueOrDefault(currentCommunity) - nodeStrength;
                    communityStrengths.TryGetValue(bestCommunity, out double bestCs);
                    communityStrengths[bestCommunity] = bestCs + nodeStrength;

                    partition[nodeId] = bestCommunity;
                    changed = true;
                    anyMoved = true;
                }
            }
        }

        return anyMoved;
    }

    private static void RefinePartition(
        List<string> nodes,
        Dictionary<string, Dictionary<string, double>> adjacency,
        Dictionary<string, int> partition,
        double resolution, double totalWeight, Random rng)
    {
        // For each community, check internal connectivity
        // If a community has disconnected components, split them
        var communities = new Dictionary<int, List<string>>();
        foreach (var nodeId in nodes)
        {
            int comm = partition[nodeId];
            if (!communities.ContainsKey(comm))
                communities[comm] = [];
            communities[comm].Add(nodeId);
        }

        int nextCommunityId = communities.Keys.DefaultIfEmpty(-1).Max() + 1;

        foreach (var (comm, members) in communities)
        {
            if (members.Count <= 1) continue;

            // BFS to find connected components within the community
            var visited = new HashSet<string>();
            var components = new List<List<string>>();

            foreach (var member in members)
            {
                if (visited.Contains(member)) continue;

                var component = new List<string>();
                var queue = new Queue<string>();
                queue.Enqueue(member);
                visited.Add(member);

                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    component.Add(current);

                    if (!adjacency.ContainsKey(current)) continue;
                    foreach (var (neighbor, _) in adjacency[current])
                    {
                        if (!visited.Contains(neighbor) && partition[neighbor] == comm)
                        {
                            visited.Add(neighbor);
                            queue.Enqueue(neighbor);
                        }
                    }
                }

                components.Add(component);
            }

            // If multiple components, assign each to a separate community
            if (components.Count > 1)
            {
                // Keep the largest component in the original community
                var largest = components.OrderByDescending(c => c.Count).First();
                foreach (var comp in components)
                {
                    if (comp == largest) continue;
                    foreach (var nodeId in comp)
                    {
                        partition[nodeId] = nextCommunityId;
                    }
                    nextCommunityId++;
                }
            }
        }
    }

    private static (List<string> superNodes, Dictionary<string, Dictionary<string, double>> superAdj, Dictionary<string, string> nodeMapping)
        AggregateGraph(
            List<string> nodes,
            Dictionary<string, Dictionary<string, double>> adjacency,
            Dictionary<string, int> partition)
    {
        // Map each community to a super-node
        var communityIds = partition.Values.Distinct().OrderBy(x => x).ToList();
        var superNodes = communityIds.Select(c => $"super_{c}").ToList();
        var commToSuper = communityIds.ToDictionary(c => c, c => $"super_{c}");
        var nodeMapping = new Dictionary<string, string>();
        foreach (var nodeId in nodes)
        {
            nodeMapping[nodeId] = commToSuper[partition[nodeId]];
        }

        // Build super-node adjacency
        var superAdj = new Dictionary<string, Dictionary<string, double>>();
        foreach (var sn in superNodes)
            superAdj[sn] = [];

        foreach (var nodeId in nodes)
        {
            if (!adjacency.ContainsKey(nodeId)) continue;
            string sourceSuper = nodeMapping[nodeId];

            foreach (var (neighbor, weight) in adjacency[nodeId])
            {
                if (!nodeMapping.ContainsKey(neighbor)) continue;
                string targetSuper = nodeMapping[neighbor];
                if (sourceSuper == targetSuper) continue; // Skip self-loops

                superAdj[sourceSuper].TryGetValue(targetSuper, out double w);
                superAdj[sourceSuper][targetSuper] = w + weight;
            }
        }

        return (superNodes, superAdj, nodeMapping);
    }

    private static double ComputeModularity(
        List<string> nodes,
        Dictionary<string, Dictionary<string, double>> adjacency,
        Dictionary<string, int> partition,
        double resolution, double totalWeight)
    {
        if (totalWeight < 1e-12) return 0.0;

        // O(n + |E|) modularity computation using per-community aggregates
        // Q = (1/2m) * Σ_c [ L_c - resolution * (S_c)^2 / (2m) ]
        // where L_c = sum of edge weights within community c, S_c = sum of node strengths in c

        var communityInternalWeight = new Dictionary<int, double>();
        var communityStrength = new Dictionary<int, double>();

        foreach (var nodeId in nodes)
        {
            int comm = partition[nodeId];
            double strength = adjacency.ContainsKey(nodeId) ? adjacency[nodeId].Values.Sum() : 0.0;
            communityStrength.TryGetValue(comm, out double cs);
            communityStrength[comm] = cs + strength;

            if (!adjacency.ContainsKey(nodeId)) continue;
            foreach (var (neighbor, weight) in adjacency[nodeId])
            {
                if (partition.TryGetValue(neighbor, out int neighborComm) && neighborComm == comm)
                {
                    communityInternalWeight.TryGetValue(comm, out double lc);
                    communityInternalWeight[comm] = lc + weight; // Each edge counted twice (both directions)
                }
            }
        }

        double modularity = 0.0;
        foreach (var comm in communityStrength.Keys)
        {
            communityInternalWeight.TryGetValue(comm, out double lc);
            double sc = communityStrength[comm];
            // Standard modularity: Q = (1/2m) Σ[Aij - ki*kj/2m] δ(ci,cj)
            // lc sums both directions of each internal edge, so lc/2 = actual internal weight.
            modularity += lc / 2.0 - resolution * sc * sc / (2.0 * totalWeight);
        }

        return modularity / (2.0 * totalWeight);
    }

    private static void Shuffle<TItem>(List<TItem> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
