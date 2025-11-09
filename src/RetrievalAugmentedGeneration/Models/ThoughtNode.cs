using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Models;

/// <summary>
/// Represents a node in the Tree-of-Thoughts reasoning tree.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class ThoughtNode<T>
{
    /// <summary>
    /// The reasoning thought or statement at this node.
    /// </summary>
    public string Thought { get; set; } = string.Empty;

    /// <summary>
    /// Child nodes branching from this thought.
    /// </summary>
    public List<ThoughtNode<T>> Children { get; set; } = new List<ThoughtNode<T>>();

    /// <summary>
    /// Evaluation score for this thought (0-1, higher is better).
    /// </summary>
    public double EvaluationScore { get; set; }

    /// <summary>
    /// Documents retrieved for this thought.
    /// </summary>
    public List<Document<T>> RetrievedDocuments { get; set; } = new List<Document<T>>();

    /// <summary>
    /// Depth of this node in the tree (0 = root).
    /// </summary>
    public int Depth { get; set; }

    /// <summary>
    /// Parent node in the tree (null for root).
    /// </summary>
    public ThoughtNode<T>? Parent { get; set; }
}
