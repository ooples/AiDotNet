using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Tree-of-Thoughts retriever that explores multiple reasoning paths in a tree structure.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This advanced retrieval pattern builds upon Chain-of-Thought by creating a tree of
/// possible reasoning paths. Instead of following a single linear chain, it explores
/// multiple branches of reasoning at each step, evaluates them, and can backtrack to
/// explore alternative paths. This enables more comprehensive exploration of complex
/// problem spaces.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a chess player considering multiple moves.
///
/// Chain-of-Thought (linear):
/// - Question: "Impact of AI on healthcare?"
/// - Path: AI → Diagnosis → Treatment → Outcomes
///
/// Tree-of-Thoughts (branching):
/// - Question: "Impact of AI on healthcare?"
/// - Level 1: [AI in Diagnosis, AI in Treatment, AI in Research]
/// - Level 2 (from Diagnosis): [Image Analysis, Patient Records, Early Detection]
/// - Level 2 (from Treatment): [Drug Discovery, Personalized Medicine, Surgery Assistance]
/// - Explores all promising paths and selects best documents
///
/// This is especially useful when:
/// - Multiple valid reasoning approaches exist
/// - The problem requires exploring alternatives
/// - You want comprehensive coverage of a topic
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// var generator = new StubGenerator&lt;double&gt;();
/// var baseRetriever = new DenseRetriever&lt;double&gt;(embeddingModel, documentStore);
///
/// // Create tree-of-thoughts retriever
/// var totRetriever = new TreeOfThoughtsRetriever&lt;double&gt;(
///     generator,
///     baseRetriever,
///     maxDepth: 3,           // Explore 3 levels deep
///     branchingFactor: 3     // Generate 3 alternatives at each level
/// );
///
/// // Retrieve using breadth-first search
/// var documents = totRetriever.Retrieve(
///     "What are the applications of quantum computing?",
///     topK: 15,
///     searchStrategy: TreeSearchStrategy.BreadthFirst
/// );
///
/// // Retrieve using best-first search (prioritizes highest-scored paths)
/// var documentsBestFirst = totRetriever.Retrieve(
///     "What are the applications of quantum computing?",
///     topK: 15,
///     searchStrategy: TreeSearchStrategy.BestFirst
/// );
/// </code>
/// </para>
/// </remarks>
public class TreeOfThoughtsRetriever<T>
{
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _baseRetriever;
    private readonly int _maxDepth;
    private readonly int _branchingFactor;

    /// <summary>
    /// Represents a node in the reasoning tree.
    /// </summary>
    private class ThoughtNode
    {
        public string Thought { get; set; } = string.Empty;
        public List<ThoughtNode> Children { get; set; } = new List<ThoughtNode>();
        public double EvaluationScore { get; set; }
        public List<Document<T>> RetrievedDocuments { get; set; } = new List<Document<T>>();
        public int Depth { get; set; }
        public ThoughtNode? Parent { get; set; }
    }

    /// <summary>
    /// Tree search strategies for exploring the reasoning space.
    /// </summary>
    public enum TreeSearchStrategy
    {
        /// <summary>
        /// Explores all nodes at each depth level before going deeper.
        /// Good for comprehensive shallow exploration.
        /// </summary>
        BreadthFirst,

        /// <summary>
        /// Explores one branch fully before backtracking.
        /// Good for deep reasoning along specific paths.
        /// </summary>
        DepthFirst,

        /// <summary>
        /// Always explores the highest-scored node next.
        /// Good for efficient exploration of promising paths.
        /// </summary>
        BestFirst
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="TreeOfThoughtsRetriever{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator for reasoning.</param>
    /// <param name="baseRetriever">The underlying retriever to use.</param>
    /// <param name="maxDepth">Maximum depth of the reasoning tree (default: 3).</param>
    /// <param name="branchingFactor">Number of alternative thoughts at each level (default: 3).</param>
    public TreeOfThoughtsRetriever(
        IGenerator<T> generator,
        RetrieverBase<T> baseRetriever,
        int maxDepth = 3,
        int branchingFactor = 3)
    {
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _baseRetriever = baseRetriever ?? throw new ArgumentNullException(nameof(baseRetriever));

        if (maxDepth <= 0 || maxDepth > 10)
            throw new ArgumentOutOfRangeException(nameof(maxDepth), "maxDepth must be between 1 and 10");

        if (branchingFactor <= 0 || branchingFactor > 10)
            throw new ArgumentOutOfRangeException(nameof(branchingFactor), "branchingFactor must be between 1 and 10");

        _maxDepth = maxDepth;
        _branchingFactor = branchingFactor;
    }

    /// <summary>
    /// Retrieves documents using tree-of-thoughts reasoning.
    /// </summary>
    /// <param name="query">The query to retrieve documents for.</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <param name="searchStrategy">The tree search strategy to use.</param>
    /// <param name="metadataFilters">Metadata filters to apply during retrieval.</param>
    /// <returns>Retrieved documents from exploring the reasoning tree.</returns>
    public IEnumerable<Document<T>> Retrieve(
        string query,
        int topK,
        TreeSearchStrategy searchStrategy = TreeSearchStrategy.BestFirst,
        Dictionary<string, object>? metadataFilters = null)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        metadataFilters ??= new Dictionary<string, object>();

        // Build the reasoning tree
        var rootNode = new ThoughtNode { Thought = query, Depth = 0 };
        ExpandTree(rootNode, searchStrategy, metadataFilters);

        // Collect all documents from the tree
        var allDocuments = new Dictionary<string, (Document<T> doc, double maxScore)>();
        CollectDocuments(rootNode, allDocuments);

        // Return top-K documents by score
        return allDocuments.Values
            .OrderByDescending(item => item.maxScore)
            .Select(item => item.doc)
            .Take(topK);
    }

    /// <summary>
    /// Expands the reasoning tree using the specified search strategy.
    /// </summary>
    private void ExpandTree(ThoughtNode root, TreeSearchStrategy strategy, Dictionary<string, object> metadataFilters)
    {
        switch (strategy)
        {
            case TreeSearchStrategy.BreadthFirst:
                ExpandBreadthFirst(root, metadataFilters);
                break;
            case TreeSearchStrategy.DepthFirst:
                ExpandDepthFirst(root, metadataFilters);
                break;
            case TreeSearchStrategy.BestFirst:
                ExpandBestFirst(root, metadataFilters);
                break;
        }
    }

    /// <summary>
    /// Breadth-first tree expansion: explores all nodes at each level.
    /// </summary>
    private void ExpandBreadthFirst(ThoughtNode root, Dictionary<string, object> metadataFilters)
    {
        var queue = new Queue<ThoughtNode>();
        queue.Enqueue(root);

        while (queue.Count > 0)
        {
            var node = queue.Dequeue();

            if (node.Depth >= _maxDepth)
                continue;

            // Generate and evaluate child thoughts
            var children = GenerateChildThoughts(node);
            foreach (var child in children)
            {
                EvaluateAndRetrieve(child, metadataFilters);
                node.Children.Add(child);
                queue.Enqueue(child);
            }
        }
    }

    /// <summary>
    /// Depth-first tree expansion: explores one branch fully before backtracking.
    /// </summary>
    private void ExpandDepthFirst(ThoughtNode root, Dictionary<string, object> metadataFilters)
    {
        var stack = new Stack<ThoughtNode>();
        stack.Push(root);

        while (stack.Count > 0)
        {
            var node = stack.Pop();

            if (node.Depth >= _maxDepth)
                continue;

            // Generate and evaluate child thoughts
            var children = GenerateChildThoughts(node);
            foreach (var child in children.Reverse<ThoughtNode>()) // Reverse to maintain left-to-right order
            {
                EvaluateAndRetrieve(child, metadataFilters);
                node.Children.Add(child);
                stack.Push(child);
            }
        }
    }

    /// <summary>
    /// Best-first tree expansion: always explores the highest-scored node next.
    /// </summary>
    private void ExpandBestFirst(ThoughtNode root, Dictionary<string, object> metadataFilters)
    {
        // Priority queue based on evaluation score
        var priorityQueue = new SortedSet<(double score, ThoughtNode node, int id)>(
            Comparer<(double score, ThoughtNode node, int id)>.Create((a, b) =>
            {
                var scoreComparison = b.score.CompareTo(a.score); // Higher score first
                return scoreComparison != 0 ? scoreComparison : a.id.CompareTo(b.id); // Use id for tie-breaking
            })
        );

        int nodeIdCounter = 0;
        priorityQueue.Add((1.0, root, nodeIdCounter++)); // Start with root (score 1.0)

        while (priorityQueue.Count > 0)
        {
            var (_, node, _) = priorityQueue.Min;
            priorityQueue.Remove(priorityQueue.Min);

            if (node.Depth >= _maxDepth)
                continue;

            // Generate and evaluate child thoughts
            var children = GenerateChildThoughts(node);
            foreach (var child in children)
            {
                EvaluateAndRetrieve(child, metadataFilters);
                node.Children.Add(child);
                priorityQueue.Add((child.EvaluationScore, child, nodeIdCounter++));
            }
        }
    }

    /// <summary>
    /// Generates alternative child thoughts for a given node.
    /// </summary>
    private List<ThoughtNode> GenerateChildThoughts(ThoughtNode parent)
    {
        var children = new List<ThoughtNode>();

        // Build context from parent chain
        var context = BuildThoughtContext(parent);

        var prompt = $@"{context}

Based on the reasoning so far, generate {_branchingFactor} different next steps or perspectives to explore.
Each should be a specific, focused direction for further investigation.

Format your response as a numbered list:
1. [First alternative thought]
2. [Second alternative thought]
3. [Third alternative thought]";

        var response = _generator.Generate(prompt);
        var thoughts = ParseThoughts(response);

        foreach (var thought in thoughts.Take(_branchingFactor))
        {
            children.Add(new ThoughtNode
            {
                Thought = thought,
                Depth = parent.Depth + 1,
                Parent = parent
            });
        }

        return children;
    }

    /// <summary>
    /// Evaluates a thought node and retrieves relevant documents.
    /// </summary>
    private void EvaluateAndRetrieve(ThoughtNode node, Dictionary<string, object> metadataFilters)
    {
        // Retrieve documents for this thought
        var documents = _baseRetriever.Retrieve(node.Thought, topK: 5, metadataFilters).ToList();
        node.RetrievedDocuments = documents;

        // Evaluate the quality of this thought
        node.EvaluationScore = EvaluateThought(node);
    }

    /// <summary>
    /// Evaluates the quality of a thought based on retrieved documents and coherence.
    /// </summary>
    private double EvaluateThought(ThoughtNode node)
    {
        // Evaluation criteria:
        // 1. Number of relevant documents found (0-1 normalized)
        // 2. Average relevance score of documents
        // 3. Coherence with parent chain (evaluated by LLM - simplified here)

        double documentScore = Math.Min(node.RetrievedDocuments.Count / 5.0, 1.0);

        double relevanceScore = 0.0;
        if (node.RetrievedDocuments.Count > 0 && node.RetrievedDocuments.Any(d => d.HasRelevanceScore))
        {
            var scores = node.RetrievedDocuments
                .Where(d => d.HasRelevanceScore)
                .Select(d => Convert.ToDouble(d.RelevanceScore))
                .ToList();

            if (scores.Count > 0)
            {
                relevanceScore = scores.Average();
            }
        }

        // Depth penalty: slightly prefer shallower nodes to avoid going too deep
        double depthPenalty = 1.0 - (node.Depth * 0.1);

        return (documentScore * 0.3 + relevanceScore * 0.6 + depthPenalty * 0.1);
    }

    /// <summary>
    /// Builds a context string from the parent chain of thoughts.
    /// </summary>
    private string BuildThoughtContext(ThoughtNode node)
    {
        var chain = new List<string>();
        var current = node;

        while (current != null)
        {
            chain.Insert(0, current.Thought);
            current = current.Parent;
        }

        if (chain.Count == 0)
            return string.Empty;

        var context = "Question: " + chain[0];
        for (int i = 1; i < chain.Count; i++)
        {
            context += $"\nStep {i}: {chain[i]}";
        }

        return context;
    }

    /// <summary>
    /// Parses thoughts from LLM response.
    /// </summary>
    private List<string> ParseThoughts(string response)
    {
        var thoughts = new List<string>();
        var lines = response.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var trimmed = line.Trim();

            // Match numbered list items like "1. ", "1) ", "- ", etc.
            var match = System.Text.RegularExpressions.Regex.Match(
                trimmed,
                @"^(?:\d+[\.\)]\s*|[-\*]\s*)(.+)$"
            );

            if (match.Success && match.Groups[1].Value.Length > 10)
            {
                thoughts.Add(match.Groups[1].Value.Trim());
            }
        }

        return thoughts;
    }

    /// <summary>
    /// Collects all documents from the tree, keeping track of the best score for each.
    /// </summary>
    private void CollectDocuments(ThoughtNode node, Dictionary<string, (Document<T> doc, double maxScore)> allDocuments)
    {
        // Add documents from this node
        foreach (var doc in node.RetrievedDocuments)
        {
            double score = node.EvaluationScore;

            if (allDocuments.ContainsKey(doc.Id))
            {
                // Keep the higher score
                if (score > allDocuments[doc.Id].maxScore)
                {
                    allDocuments[doc.Id] = (doc, score);
                }
            }
            else
            {
                allDocuments[doc.Id] = (doc, score);
            }
        }

        // Recursively collect from children
        foreach (var child in node.Children)
        {
            CollectDocuments(child, allDocuments);
        }
    }
}
