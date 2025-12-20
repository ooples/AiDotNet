using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Provides graph analytics algorithms for analyzing knowledge graphs.
/// </summary>
/// <remarks>
/// <para>
/// This class implements common graph algorithms used to analyze the structure
/// and importance of nodes and edges in a knowledge graph.
/// </para>
/// <para><b>For Beginners:</b> Graph analytics help you understand your graph.
///
/// Think of a social network:
/// - PageRank: Who are the most influential people?
/// - Degree Centrality: Who has the most connections?
/// - Closeness Centrality: Who can reach everyone quickly?
/// - Betweenness Centrality: Who connects different groups?
///
/// These algorithms answer "who's important?" and "how are things connected?"
/// </para>
/// </remarks>
public static class GraphAnalytics
{
    /// <summary>
    /// Calculates PageRank scores for all nodes in the graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <param name="dampingFactor">The damping factor (default: 0.85). Must be between 0 and 1.</param>
    /// <param name="maxIterations">Maximum number of iterations (default: 100).</param>
    /// <param name="convergenceThreshold">Convergence threshold (default: 0.0001).</param>
    /// <returns>Dictionary mapping node IDs to their PageRank scores.</returns>
    /// <remarks>
    /// <para>
    /// PageRank is an algorithm used by Google to rank web pages. In a knowledge graph,
    /// it identifies the most "important" or "central" nodes based on the structure
    /// of relationships. Nodes pointed to by many important nodes get higher scores.
    /// </para>
    /// <para><b>For Beginners:</b> PageRank finds the most important nodes.
    ///
    /// Imagine a citation network:
    /// - Papers cited by many other papers get high PageRank
    /// - Papers cited by highly-cited papers get even higher PageRank
    /// - It's like asking: "Which papers are most influential?"
    ///
    /// The algorithm:
    /// 1. Start: All nodes have equal rank
    /// 2. Iterate: Each node shares its rank with nodes it points to
    /// 3. Damping: 85% of rank flows through edges, 15% jumps randomly
    /// 4. Repeat until scores stabilize
    ///
    /// Higher PageRank = More important/central in the graph
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when graph is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when dampingFactor is not between 0 and 1.</exception>
    public static Dictionary<string, double> CalculatePageRank<T>(
        KnowledgeGraph<T> graph,
        double dampingFactor = 0.85,
        int maxIterations = 100,
        double convergenceThreshold = 0.0001)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));
        if (dampingFactor < 0 || dampingFactor > 1)
            throw new ArgumentOutOfRangeException(nameof(dampingFactor), "Damping factor must be between 0 and 1");

        var nodes = graph.GetAllNodes().ToList();
        if (nodes.Count == 0)
            return new Dictionary<string, double>();

        var nodeCount = nodes.Count;
        var ranks = new Dictionary<string, double>();
        var newRanks = new Dictionary<string, double>();

        // Initialize all nodes with equal rank
        var initialRank = 1.0 / nodeCount;
        foreach (var node in nodes)
        {
            ranks[node.Id] = initialRank;
            newRanks[node.Id] = 0.0;
        }

        // Iterate until convergence or max iterations
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Calculate new ranks
            foreach (var node in nodes)
            {
                var incomingEdges = graph.GetIncomingEdges(node.Id).ToList();
                double rankSum = 0.0;

                foreach (var edge in incomingEdges)
                {
                    var sourceNode = graph.GetNode(edge.SourceId);
                    if (sourceNode != null)
                    {
                        var outgoingCount = graph.GetOutgoingEdges(edge.SourceId).Count();
                        if (outgoingCount > 0)
                        {
                            rankSum += ranks[edge.SourceId] / outgoingCount;
                        }
                    }
                }

                // PageRank formula: PR(A) = (1-d)/N + d * sum(PR(Ti)/C(Ti))
                newRanks[node.Id] = (1 - dampingFactor) / nodeCount + dampingFactor * rankSum;
            }

            // Check for convergence
            double maxChange = 0.0;
            foreach (var node in nodes)
            {
                var change = Math.Abs(newRanks[node.Id] - ranks[node.Id]);
                if (change > maxChange)
                    maxChange = change;
                ranks[node.Id] = newRanks[node.Id];
            }

            if (maxChange < convergenceThreshold)
                break;
        }

        return ranks;
    }

    /// <summary>
    /// Calculates degree centrality for all nodes in the graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <param name="normalized">Whether to normalize scores by the maximum possible degree (default: true).</param>
    /// <returns>Dictionary mapping node IDs to their degree centrality scores.</returns>
    /// <remarks>
    /// <para>
    /// Degree centrality is the simplest centrality measure. It counts the number of
    /// edges connected to each node. Nodes with more connections are considered more central.
    /// </para>
    /// <para><b>For Beginners:</b> Degree centrality counts connections.
    ///
    /// In a social network:
    /// - Person with 100 friends has higher degree centrality than person with 10 friends
    /// - It's the simplest measure: "Who knows the most people?"
    ///
    /// Types:
    /// - Out-degree: How many edges go OUT (how many people do you follow?)
    /// - In-degree: How many edges come IN (how many followers do you have?)
    /// - Total degree: Sum of both
    ///
    /// This implementation calculates total degree (in + out).
    ///
    /// Normalized: Divides by (N-1) where N is total nodes, giving a score from 0 to 1.
    /// </para>
    /// </remarks>
    public static Dictionary<string, double> CalculateDegreeCentrality<T>(
        KnowledgeGraph<T> graph,
        bool normalized = true)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        var nodes = graph.GetAllNodes().ToList();
        var centrality = new Dictionary<string, double>();

        if (nodes.Count == 0)
            return centrality;

        foreach (var node in nodes)
        {
            var outDegree = graph.GetOutgoingEdges(node.Id).Count();
            var inDegree = graph.GetIncomingEdges(node.Id).Count();
            var totalDegree = outDegree + inDegree;

            // Normalize by the maximum possible degree (n-1) for undirected,
            // or 2(n-1) for directed graphs
            centrality[node.Id] = (normalized && nodes.Count > 1)
                ? totalDegree / (2.0 * (nodes.Count - 1))
                : totalDegree;
        }

        return centrality;
    }

    /// <summary>
    /// Calculates closeness centrality for all nodes in the graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <returns>Dictionary mapping node IDs to their closeness centrality scores.</returns>
    /// <remarks>
    /// <para>
    /// Closeness centrality measures how close a node is to all other nodes in the graph.
    /// It's calculated as the inverse of the average shortest path distance to all other nodes.
    /// Nodes that can quickly reach all others have high closeness centrality.
    /// </para>
    /// <para><b>For Beginners:</b> Closeness centrality measures "how close" you are to everyone.
    ///
    /// Think of an airport network:
    /// - Hub airports (like Atlanta) can reach anywhere with few layovers → high closeness
    /// - Remote airports need many connections → low closeness
    ///
    /// Algorithm:
    /// 1. For each node, find shortest path to every other node
    /// 2. Calculate average distance
    /// 3. Closeness = 1 / average_distance
    ///
    /// Higher score = Can reach everyone more quickly
    ///
    /// Note: If nodes are unreachable, they're excluded from the calculation.
    /// </para>
    /// </remarks>
    public static Dictionary<string, double> CalculateClosenessCentrality<T>(
        KnowledgeGraph<T> graph)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        var nodes = graph.GetAllNodes().ToList();
        var centrality = new Dictionary<string, double>();

        if (nodes.Count <= 1)
        {
            foreach (var node in nodes)
                centrality[node.Id] = 0.0;
            return centrality;
        }

        foreach (var node in nodes)
        {
            var distances = BreadthFirstSearchDistances(graph, node.Id);
            var reachableNodes = distances.Values.Where(d => d > 0 && d < int.MaxValue).ToList();

            if (reachableNodes.Count == 0)
            {
                centrality[node.Id] = 0.0;
            }
            else
            {
                var averageDistance = reachableNodes.Average();
                centrality[node.Id] = (nodes.Count - 1) / (averageDistance * reachableNodes.Count);
            }
        }

        return centrality;
    }

    /// <summary>
    /// Calculates betweenness centrality for all nodes in the graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <param name="normalized">Whether to normalize scores (default: true).</param>
    /// <returns>Dictionary mapping node IDs to their betweenness centrality scores.</returns>
    /// <remarks>
    /// <para>
    /// Betweenness centrality measures how often a node appears on shortest paths between
    /// other nodes. Nodes that act as "bridges" between different parts of the graph
    /// have high betweenness centrality.
    /// </para>
    /// <para><b>For Beginners:</b> Betweenness centrality finds "bridge" nodes.
    ///
    /// Think of a transportation network:
    /// - A bridge connecting two cities has high betweenness
    /// - Many shortest paths go through the bridge
    /// - If you remove it, people must take longer routes
    ///
    /// In social networks:
    /// - People connecting different friend groups have high betweenness
    /// - They're "brokers" or "gatekeepers" of information
    ///
    /// Algorithm (simplified Brandes' algorithm):
    /// 1. For all pairs of nodes, find shortest paths
    /// 2. Count how many paths go through each node
    /// 3. Higher count = Higher betweenness
    ///
    /// High betweenness = Important connector/bridge in the network
    /// </para>
    /// </remarks>
    public static Dictionary<string, double> CalculateBetweennessCentrality<T>(
        KnowledgeGraph<T> graph,
        bool normalized = true)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        var nodes = graph.GetAllNodes().ToList();
        var betweenness = new Dictionary<string, double>();

        // Initialize all to 0
        foreach (var node in nodes)
            betweenness[node.Id] = 0.0;

        if (nodes.Count <= 2)
            return betweenness;

        // For each node as source
        foreach (var source in nodes)
        {
            var stack = new Stack<string>();
            var paths = new Dictionary<string, List<string>>();
            var pathCounts = new Dictionary<string, int>();
            var distances = new Dictionary<string, int>();
            var dependencies = new Dictionary<string, double>();

            foreach (var node in nodes)
            {
                paths[node.Id] = new List<string>();
                pathCounts[node.Id] = 0;
                distances[node.Id] = -1;
                dependencies[node.Id] = 0.0;
            }

            pathCounts[source.Id] = 1;
            distances[source.Id] = 0;

            var queue = new Queue<string>();
            queue.Enqueue(source.Id);

            // BFS
            while (queue.Count > 0)
            {
                var v = queue.Dequeue();
                stack.Push(v);

                foreach (var edge in graph.GetOutgoingEdges(v))
                {
                    var w = edge.TargetId;
                    // First time we see w?
                    if (distances[w] < 0)
                    {
                        queue.Enqueue(w);
                        distances[w] = distances[v] + 1;
                    }

                    // Shortest path to w via v?
                    if (distances[w] == distances[v] + 1)
                    {
                        pathCounts[w] += pathCounts[v];
                        paths[w].Add(v);
                    }
                }
            }

            // Accumulate dependencies
            while (stack.Count > 0)
            {
                var w = stack.Pop();
                foreach (var v in paths[w])
                {
                    dependencies[v] += (pathCounts[v] / (double)pathCounts[w]) * (1.0 + dependencies[w]);
                }

                if (w != source.Id)
                    betweenness[w] += dependencies[w];
            }
        }

        // Normalize if requested
        if (normalized && nodes.Count > 2)
        {
            var normFactor = (nodes.Count - 1) * (nodes.Count - 2);
            foreach (var node in nodes)
            {
                betweenness[node.Id] /= normFactor;
            }
        }

        return betweenness;
    }

    /// <summary>
    /// Performs breadth-first search to calculate distances from a source node to all others.
    /// </summary>
    private static Dictionary<string, int> BreadthFirstSearchDistances<T>(
        KnowledgeGraph<T> graph,
        string sourceId)
    {
        var distances = new Dictionary<string, int>();
        var nodes = graph.GetAllNodes();

        foreach (var node in nodes)
            distances[node.Id] = int.MaxValue;

        distances[sourceId] = 0;
        var queue = new Queue<string>();
        queue.Enqueue(sourceId);

        while (queue.Count > 0)
        {
            var current = queue.Dequeue();
            var currentDistance = distances[current];

            foreach (var edge in graph.GetOutgoingEdges(current).Where(e => distances[e.TargetId] == int.MaxValue))
            {
                distances[edge.TargetId] = currentDistance + 1;
                queue.Enqueue(edge.TargetId);
            }
        }

        return distances;
    }

    /// <summary>
    /// Identifies the top-k most central nodes based on a centrality measure.
    /// </summary>
    /// <param name="centrality">Dictionary of node IDs to centrality scores.</param>
    /// <param name="k">Number of top nodes to return.</param>
    /// <returns>List of (nodeId, score) tuples for the top-k nodes, ordered by descending score.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds the "most important" nodes.
    ///
    /// After running PageRank or centrality calculations, use this to get:
    /// - Top 10 most influential people (PageRank)
    /// - Top 5 most connected nodes (Degree)
    /// - Top 3 bridge nodes (Betweenness)
    ///
    /// Example:
    /// ```csharp
    /// var pageRank = GraphAnalytics.CalculatePageRank(graph);
    /// var top10 = GraphAnalytics.GetTopKNodes(pageRank, 10);
    /// // Returns the 10 most important nodes with their scores
    /// ```
    /// </para>
    /// </remarks>
    public static List<(string NodeId, double Score)> GetTopKNodes(
        Dictionary<string, double> centrality,
        int k)
    {
        if (centrality == null)
            throw new ArgumentNullException(nameof(centrality));

        return centrality
            .OrderByDescending(kvp => kvp.Value)
            .Take(k)
            .Select(kvp => (kvp.Key, kvp.Value))
            .ToList();
    }

    /// <summary>
    /// Finds all connected components in the graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <returns>List of connected components, each containing a set of node IDs.</returns>
    /// <remarks>
    /// <para>
    /// A connected component is a maximal subgraph where every node can reach every other node.
    /// This helps identify isolated clusters or communities in the graph.
    /// </para>
    /// <para><b>For Beginners:</b> Connected components find separate "islands" in your graph.
    ///
    /// Think of a social network:
    /// - Component 1: Alice's friend group (all connected to each other)
    /// - Component 2: Bob's friend group (completely separate from Alice's)
    /// - Component 3: Isolated person Charlie (no connections)
    ///
    /// Uses:
    /// - Find isolated communities
    /// - Detect fragmented knowledge bases
    /// - Identify separate discussion topics
    /// - Check if graph is fully connected
    ///
    /// Algorithm (Union-Find/DFS):
    /// 1. Start with first unvisited node
    /// 2. Find all nodes reachable from it (BFS/DFS)
    /// 3. That's one component
    /// 4. Repeat for remaining unvisited nodes
    ///
    /// Result: List of components, each containing node IDs in that island.
    /// </para>
    /// </remarks>
    public static List<HashSet<string>> FindConnectedComponents<T>(KnowledgeGraph<T> graph)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        var nodes = graph.GetAllNodes().ToList();
        var visited = new HashSet<string>();
        var components = new List<HashSet<string>>();

        foreach (var node in nodes.Where(n => !visited.Contains(n.Id)))
        {
            var component = new HashSet<string>();
            var queue = new Queue<string>();
            queue.Enqueue(node.Id);
            visited.Add(node.Id);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                component.Add(current);

                // Check outgoing edges
                foreach (var edge in graph.GetOutgoingEdges(current).Where(e => !visited.Contains(e.TargetId)))
                {
                    visited.Add(edge.TargetId);
                    queue.Enqueue(edge.TargetId);
                }

                // Check incoming edges (for undirected behavior)
                foreach (var edge in graph.GetIncomingEdges(current).Where(e => !visited.Contains(e.SourceId)))
                {
                    visited.Add(edge.SourceId);
                    queue.Enqueue(edge.SourceId);
                }
            }

            components.Add(component);
        }

        return components;
    }

    /// <summary>
    /// Detects communities using Label Propagation algorithm.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <param name="maxIterations">Maximum number of iterations (default: 100).</param>
    /// <returns>Dictionary mapping node IDs to their community labels.</returns>
    /// <remarks>
    /// <para>
    /// Label Propagation is a fast community detection algorithm. Each node starts with
    /// a unique label, then iteratively adopts the most common label among its neighbors.
    /// Nodes in the same community will converge to the same label.
    /// </para>
    /// <para><b>For Beginners:</b> Label Propagation finds groups of nodes that cluster together.
    ///
    /// Imagine a party where people wear colored hats:
    /// 1. Everyone starts with a random color
    /// 2. Every minute, you look at your friends' hats
    /// 3. You change to the most popular color among your friends
    /// 4. After a while, friend groups wear the same color
    ///
    /// In graphs:
    /// - Start: Each node has unique label
    /// - Iterate: Each node adopts most common neighbor label
    /// - Converge: Nodes in same community have same label
    ///
    /// Why it works:
    /// - Densely connected nodes influence each other
    /// - They converge to the same label
    /// - Weakly connected nodes drift apart
    ///
    /// Result: Community labels (numbers) for each node.
    /// Nodes with the same label are in the same community.
    ///
    /// Fast: O(k * E) where k = iterations, E = edges
    /// Great for: Large graphs, quick community detection
    /// </para>
    /// </remarks>
    public static Dictionary<string, int> DetectCommunitiesLabelPropagation<T>(
        KnowledgeGraph<T> graph,
        int maxIterations = 100)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        var nodes = graph.GetAllNodes().ToList();
        var labels = new Dictionary<string, int>();
        var random = RandomHelper.ThreadSafeRandom;

        // Initialize each node with unique label
        for (int i = 0; i < nodes.Count; i++)
        {
            labels[nodes[i].Id] = i;
        }

        // Iterate until convergence or max iterations
        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            bool changed = false;

            // Process nodes in random order
            var shuffledNodes = nodes.OrderBy(n => random.Next()).ToList();

            foreach (var node in shuffledNodes)
            {
                // Get labels of all neighbors
                var neighborLabels = new List<int>();

                foreach (var edge in graph.GetOutgoingEdges(node.Id))
                {
                    if (labels.TryGetValue(edge.TargetId, out var targetLabel))
                        neighborLabels.Add(targetLabel);
                }

                foreach (var edge in graph.GetIncomingEdges(node.Id))
                {
                    if (labels.TryGetValue(edge.SourceId, out var sourceLabel))
                        neighborLabels.Add(sourceLabel);
                }

                if (neighborLabels.Count == 0)
                    continue;

                // Find most common label
                var labelCounts = neighborLabels
                    .GroupBy(l => l)
                    .Select(g => new { Label = g.Key, Count = g.Count() })
                    .OrderByDescending(x => x.Count)
                    .ThenBy(x => random.Next()) // Random tie-breaking
                    .First();

                // Update label if different
                if (labels[node.Id] != labelCounts.Label)
                {
                    labels[node.Id] = labelCounts.Label;
                    changed = true;
                }
            }

            // Converged if no changes
            if (!changed)
                break;
        }

        return labels;
    }

    /// <summary>
    /// Calculates the clustering coefficient for each node.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <returns>Dictionary mapping node IDs to their clustering coefficients (0 to 1).</returns>
    /// <remarks>
    /// <para>
    /// The clustering coefficient measures how well a node's neighbors are connected to each other.
    /// A high coefficient means the node is part of a tightly-knit cluster.
    /// </para>
    /// <para><b>For Beginners:</b> Clustering coefficient measures how "clique-like" connections are.
    ///
    /// Think of your friend group:
    /// - High clustering: Your friends all know each other (tight group)
    /// - Low clustering: Your friends don't know each other (you're the hub)
    ///
    /// Formula:
    /// - Count how many of your friends are friends with each other
    /// - Divide by maximum possible friendships between them
    /// - Result: 0 (no friends know each other) to 1 (everyone knows everyone)
    ///
    /// Example:
    /// - You have 3 friends: Alice, Bob, Charlie
    /// - Maximum connections between them: 3 (Alice-Bob, Bob-Charlie, Alice-Charlie)
    /// - Actual connections: 2 (Alice-Bob, Bob-Charlie)
    /// - Clustering coefficient: 2/3 = 0.67
    ///
    /// Uses:
    /// - Identify tight communities
    /// - Find nodes embedded in groups vs connectors between groups
    /// - Measure graph's overall "cliquishness"
    ///
    /// High coefficient = Node in dense cluster
    /// Low coefficient = Node connects different groups
    /// </para>
    /// </remarks>
    public static Dictionary<string, double> CalculateClusteringCoefficient<T>(
        KnowledgeGraph<T> graph)
    {
        if (graph == null)
            throw new ArgumentNullException(nameof(graph));

        var nodes = graph.GetAllNodes().ToList();
        var coefficients = new Dictionary<string, double>();

        foreach (var node in nodes)
        {
            var neighbors = new HashSet<string>();

            // Collect all neighbors (treat as undirected)
            foreach (var edge in graph.GetOutgoingEdges(node.Id))
                neighbors.Add(edge.TargetId);
            foreach (var edge in graph.GetIncomingEdges(node.Id))
                neighbors.Add(edge.SourceId);

            if (neighbors.Count < 2)
            {
                coefficients[node.Id] = 0.0;
                continue;
            }

            // Count connections between neighbors
            int connectedPairs = 0;
            var neighborList = neighbors.ToList();

            for (int i = 0; i < neighborList.Count; i++)
            {
                for (int j = i + 1; j < neighborList.Count; j++)
                {
                    var n1 = neighborList[i];
                    var n2 = neighborList[j];

                    // Check if n1 and n2 are connected
                    bool connected = graph.GetOutgoingEdges(n1).Any(e => e.TargetId == n2) ||
                                   graph.GetOutgoingEdges(n2).Any(e => e.TargetId == n1);

                    if (connected)
                        connectedPairs++;
                }
            }

            // Clustering coefficient = actual connections / possible connections
            int possiblePairs = neighbors.Count * (neighbors.Count - 1) / 2;
            coefficients[node.Id] = (double)connectedPairs / possiblePairs;
        }

        return coefficients;
    }

    /// <summary>
    /// Calculates the average clustering coefficient for the entire graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used for vector operations.</typeparam>
    /// <param name="graph">The knowledge graph to analyze.</param>
    /// <returns>The average clustering coefficient (0 to 1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how "clustered" the entire graph is.
    ///
    /// - Close to 1: Graph has many tight groups (like friend circles)
    /// - Close to 0: Graph is sparse, few triangles (like a tree)
    ///
    /// Compare to random graphs:
    /// - Random graph: Low clustering (~0.01 for large graphs)
    /// - Real social networks: High clustering (~0.3-0.6)
    /// - Small-world networks: High clustering + short paths
    ///
    /// This is one measure of graph structure used in network science.
    /// </para>
    /// </remarks>
    public static double CalculateAverageClusteringCoefficient<T>(KnowledgeGraph<T> graph)
    {
        var coefficients = CalculateClusteringCoefficient(graph);
        return coefficients.Count > 0 ? coefficients.Values.Average() : 0.0;
    }
}
