using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Simple pattern matching for graph queries (inspired by Cypher/SPARQL but simplified).
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// Supports basic graph pattern matching queries like:
/// - (Person)-[KNOWS]->(Person)
/// - (Person {name: "Alice"})-[WORKS_AT]->(Company)
/// - (a:Person)-[r:KNOWS]->(b:Person)
/// </para>
/// <para><b>For Beginners:</b> Pattern matching is like SQL for graphs.
///
/// SQL Example:
/// ```sql
/// SELECT * FROM persons WHERE name = 'Alice'
/// ```
///
/// Graph Pattern Example:
/// ```
/// (Person {name: "Alice"})-[KNOWS]->(Person)
/// ```
/// Meaning: Find all people that Alice knows
///
/// Another Example:
/// ```
/// (Person)-[WORKS_AT]->(Company {name: "Google"})
/// ```
/// Meaning: Find all people who work at Google
///
/// This is much more natural for relationship-heavy data!
/// </para>
/// </remarks>
public class GraphQueryMatcher<T>
{
    private readonly KnowledgeGraph<T> _graph;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphQueryMatcher{T}"/> class.
    /// </summary>
    /// <param name="graph">The knowledge graph to query.</param>
    public GraphQueryMatcher(KnowledgeGraph<T> graph)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
    }

    /// <summary>
    /// Finds nodes matching a label and optional property filters.
    /// </summary>
    /// <param name="label">The node label to match.</param>
    /// <param name="properties">Optional property filters.</param>
    /// <returns>List of matching nodes.</returns>
    /// <example>
    /// FindNodes("Person", new Dictionary&lt;string, object&gt; { { "name", "Alice" } })
    /// </example>
    public List<GraphNode<T>> FindNodes(string label, Dictionary<string, object>? properties = null)
    {
        if (string.IsNullOrWhiteSpace(label))
            throw new ArgumentException("Label cannot be null or whitespace", nameof(label));

        var nodes = _graph.GetNodesByLabel(label).ToList();

        if (properties == null || properties.Count == 0)
            return nodes;

        // Filter by properties
        return nodes.Where(node =>
        {
            foreach (var (key, value) in properties)
            {
                if (!node.Properties.TryGetValue(key, out var nodeValue))
                    return false;

                // Simple equality check
                if (!AreEqual(nodeValue, value))
                    return false;
            }
            return true;
        }).ToList();
    }

    /// <summary>
    /// Finds paths matching a pattern: (source label)-[relationship type]->(target label).
    /// </summary>
    /// <param name="sourceLabel">The source node label.</param>
    /// <param name="relationshipType">The relationship type.</param>
    /// <param name="targetLabel">The target node label.</param>
    /// <param name="sourceProperties">Optional source node property filters.</param>
    /// <param name="targetProperties">Optional target node property filters.</param>
    /// <returns>List of matching paths.</returns>
    /// <example>
    /// FindPaths("Person", "KNOWS", "Person") // Find all KNOWS relationships
    /// FindPaths("Person", "WORKS_AT", "Company",
    ///     new Dictionary&lt;string, object&gt; { { "name", "Alice" } },
    ///     new Dictionary&lt;string, object&gt; { { "name", "Google" } })
    /// </example>
    public List<GraphPath<T>> FindPaths(
        string sourceLabel,
        string relationshipType,
        string targetLabel,
        Dictionary<string, object>? sourceProperties = null,
        Dictionary<string, object>? targetProperties = null)
    {
        if (string.IsNullOrWhiteSpace(sourceLabel))
            throw new ArgumentException("Source label cannot be null or whitespace", nameof(sourceLabel));
        if (string.IsNullOrWhiteSpace(relationshipType))
            throw new ArgumentException("Relationship type cannot be null or whitespace", nameof(relationshipType));
        if (string.IsNullOrWhiteSpace(targetLabel))
            throw new ArgumentException("Target label cannot be null or whitespace", nameof(targetLabel));

        var results = new List<GraphPath<T>>();

        // Find matching source nodes
        var sourceNodes = FindNodes(sourceLabel, sourceProperties);

        foreach (var sourceNode in sourceNodes)
        {
            // Get outgoing edges
            var edges = _graph.GetOutgoingEdges(sourceNode.Id)
                .Where(e => e.RelationType == relationshipType);

            foreach (var edge in edges)
            {
                var targetNode = _graph.GetNode(edge.TargetId);
                if (targetNode == null
                    || targetNode.Label != targetLabel
                    || (targetProperties != null && targetProperties.Count > 0 && !MatchesProperties(targetNode, targetProperties)))
                {
                    continue;
                }

                // Found a match!
                results.Add(new GraphPath<T>
                {
                    SourceNode = sourceNode,
                    Edge = edge,
                    TargetNode = targetNode
                });
            }
        }

        return results;
    }

    /// <summary>
    /// Finds all paths of specified length from a source node.
    /// </summary>
    /// <param name="sourceId">The source node ID.</param>
    /// <param name="pathLength">The path length (number of hops).</param>
    /// <param name="relationshipType">Optional relationship type filter.</param>
    /// <returns>List of paths.</returns>
    public List<List<GraphNode<T>>> FindPathsOfLength(
        string sourceId,
        int pathLength,
        string? relationshipType = null)
    {
        if (string.IsNullOrWhiteSpace(sourceId))
            throw new ArgumentException("Source ID cannot be null or whitespace", nameof(sourceId));
        if (pathLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(pathLength), "Path length must be positive");

        var results = new List<List<GraphNode<T>>>();
        var currentPaths = new List<List<GraphNode<T>>>();

        var sourceNode = _graph.GetNode(sourceId);
        if (sourceNode == null)
            return results;

        // Initialize with source node
        currentPaths.Add(new List<GraphNode<T>> { sourceNode });

        // BFS expansion
        for (int depth = 0; depth < pathLength; depth++)
        {
            var nextPaths = new List<List<GraphNode<T>>>();

            foreach (var path in currentPaths)
            {
                var lastNode = path[^1];
                var edges = _graph.GetOutgoingEdges(lastNode.Id);

                // Filter by relationship type if specified
                if (!string.IsNullOrWhiteSpace(relationshipType))
                {
                    edges = edges.Where(e => e.RelationType == relationshipType);
                }

                // Map edges to target nodes, filter nulls and cycles, then create new paths
                var newPaths = edges
                    .Select(edge => _graph.GetNode(edge.TargetId))
                    .OfType<GraphNode<T>>()
                    .Where(targetNode => !path.Any(n => n.Id == targetNode.Id))
                    .Select(targetNode => new List<GraphNode<T>>(path) { targetNode });
                nextPaths.AddRange(newPaths);
            }

            currentPaths = nextPaths;
        }

        return currentPaths;
    }

    /// <summary>
    /// Finds all shortest paths between two nodes.
    /// </summary>
    /// <param name="sourceId">The source node ID.</param>
    /// <param name="targetId">The target node ID.</param>
    /// <param name="maxDepth">Maximum depth to search (prevents infinite loops).</param>
    /// <returns>List of shortest paths.</returns>
    public List<List<GraphNode<T>>> FindShortestPaths(
        string sourceId,
        string targetId,
        int maxDepth = 10)
    {
        if (string.IsNullOrWhiteSpace(sourceId))
            throw new ArgumentException("Source ID cannot be null or whitespace", nameof(sourceId));
        if (string.IsNullOrWhiteSpace(targetId))
            throw new ArgumentException("Target ID cannot be null or whitespace", nameof(targetId));

        var sourceNode = _graph.GetNode(sourceId);
        var targetNode = _graph.GetNode(targetId);

        if (sourceNode == null || targetNode == null)
            return new List<List<GraphNode<T>>>();

        if (sourceId == targetId)
            return new List<List<GraphNode<T>>> { new List<GraphNode<T>> { sourceNode } };

        // BFS to find shortest paths
        var queue = new Queue<List<GraphNode<T>>>();
        var visitedAtDepth = new Dictionary<string, int>();
        var results = new List<List<GraphNode<T>>>();
        var shortestLength = int.MaxValue;

        queue.Enqueue(new List<GraphNode<T>> { sourceNode });
        visitedAtDepth[sourceId] = 0;

        while (queue.Count > 0)
        {
            var path = queue.Dequeue();
            var currentNode = path[^1];
            var currentDepth = path.Count - 1;

            // Check if we've exceeded max depth
            if (path.Count > maxDepth)
                continue;

            // If we found longer paths than shortest, stop processing this path
            if (path.Count > shortestLength)
                continue;

            // Get neighbors - map edges to target nodes and filter nulls
            var neighbors = _graph.GetOutgoingEdges(currentNode.Id)
                .Select(edge => _graph.GetNode(edge.TargetId))
                .OfType<GraphNode<T>>();

            foreach (var neighbor in neighbors)
            {
                // Check if we found target
                if (neighbor.Id == targetId)
                {
                    var newPath = new List<GraphNode<T>>(path) { neighbor };
                    results.Add(newPath);
                    shortestLength = Math.Min(shortestLength, newPath.Count);
                    continue;
                }

                // Avoid cycles
                if (path.Any(n => n.Id == neighbor.Id))
                    continue;

                // Continue exploring - allow visiting at same or earlier depth for shortest paths
                var newDepth = currentDepth + 1;
                if (!visitedAtDepth.TryGetValue(neighbor.Id, out var prevDepth) || newDepth <= prevDepth)
                {
                    visitedAtDepth[neighbor.Id] = newDepth;
                    var newPath = new List<GraphNode<T>>(path) { neighbor };
                    queue.Enqueue(newPath);
                }
            }
        }

        return results.Where(p => p.Count == shortestLength).ToList();
    }

    /// <summary>
    /// Executes a simple pattern query string.
    /// </summary>
    /// <param name="pattern">The pattern string (simplified Cypher-like syntax).</param>
    /// <returns>List of matching paths.</returns>
    /// <example>
    /// ExecutePattern("(Person)-[KNOWS]->(Person)")
    /// ExecutePattern("(Person {name: \"Alice\"})-[WORKS_AT]->(Company)")
    /// </example>
    public List<GraphPath<T>> ExecutePattern(string pattern)
    {
        if (string.IsNullOrWhiteSpace(pattern))
            throw new ArgumentException("Pattern cannot be null or whitespace", nameof(pattern));

        // Simple regex-based pattern parser
        // Pattern: (SourceLabel {prop: "value"})-[RELATIONSHIP]->(TargetLabel {prop: "value"})
        var regex = RegexHelper.Create(@"\((\w+)(?:\s*\{([^}]+)\})?\)-\[(\w+)\]->\((\w+)(?:\s*\{([^}]+)\})?\)", RegexOptions.None);
        var match = regex.Match(pattern);

        if (!match.Success)
            throw new ArgumentException($"Invalid pattern format: {pattern}. Expected format: (SourceLabel)-[RELATIONSHIP]->(TargetLabel)", nameof(pattern));

        var sourceLabel = match.Groups[1].Value;
        var sourcePropsStr = match.Groups[2].Value;
        var relationshipType = match.Groups[3].Value;
        var targetLabel = match.Groups[4].Value;
        var targetPropsStr = match.Groups[5].Value;

        var sourceProps = ParseProperties(sourcePropsStr);
        var targetProps = ParseProperties(targetPropsStr);

        return FindPaths(sourceLabel, relationshipType, targetLabel, sourceProps, targetProps);
    }

    /// <summary>
    /// Parses property string into dictionary.
    /// </summary>
    /// <param name="propsString">String like: name: "Alice", age: 30</param>
    private Dictionary<string, object>? ParseProperties(string propsString)
    {
        if (string.IsNullOrWhiteSpace(propsString))
            return null;

        var props = propsString.Split(',')
            .Select(pair => pair.Split(':'))
            .Where(parts => parts.Length == 2)
            .Select(parts =>
            {
                var key = parts[0].Trim();
                var valueStr = parts[1].Trim().Trim('"', '\'');
                object value;
                if (int.TryParse(valueStr, out var intValue))
                    value = intValue;
                else if (double.TryParse(valueStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var doubleValue))
                    value = doubleValue;
                else
                    value = valueStr;
                return new KeyValuePair<string, object>(key, value);
            })
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        return props.Count > 0 ? props : null;
    }

    /// <summary>
    /// Checks if a node matches property filters.
    /// </summary>
    private bool MatchesProperties(GraphNode<T> node, Dictionary<string, object> properties)
    {
        foreach (var kvp in properties)
        {
            if (!node.Properties.TryGetValue(kvp.Key, out var nodeValue))
                return false;

            if (!AreEqual(nodeValue, kvp.Value))
                return false;
        }
        return true;
    }

    /// <summary>
    /// Compares two objects for equality.
    /// </summary>
    private bool AreEqual(object obj1, object obj2)
    {
        if (obj1 == null && obj2 == null)
            return true;
        if (obj1 == null || obj2 == null)
            return false;

        // Handle numeric comparisons with tolerance for floating-point values
        if (IsNumeric(obj1) && IsNumeric(obj2))
        {
            var d1 = Convert.ToDouble(obj1);
            var d2 = Convert.ToDouble(obj2);
            const double tolerance = 1e-10;
            return Math.Abs(d1 - d2) < tolerance;
        }

        return obj1.Equals(obj2);
    }

    /// <summary>
    /// Checks if an object is numeric.
    /// </summary>
    private bool IsNumeric(object obj)
    {
        return obj is int or long or float or double or decimal;
    }
}

/// <summary>
/// Represents a path in the graph: source node -> edge -> target node.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class GraphPath<T>
{
    /// <summary>
    /// Gets or sets the source node.
    /// </summary>
    public GraphNode<T> SourceNode { get; set; } = null!;

    /// <summary>
    /// Gets or sets the edge connecting source to target.
    /// </summary>
    public GraphEdge<T> Edge { get; set; } = null!;

    /// <summary>
    /// Gets or sets the target node.
    /// </summary>
    public GraphNode<T> TargetNode { get; set; } = null!;

    /// <summary>
    /// Returns a string representation of the path.
    /// </summary>
    public override string ToString()
    {
        return $"({SourceNode.Label}:{SourceNode.Id})-[{Edge.RelationType}]->({TargetNode.Label}:{TargetNode.Id})";
    }
}



