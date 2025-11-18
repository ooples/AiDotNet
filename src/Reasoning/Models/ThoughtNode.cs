using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Reasoning.Models;

/// <summary>
/// Represents a node in a tree of thoughts, used for exploring multiple reasoning paths.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring and calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Imagine you're solving a complex problem and at each step, you could go
/// in several different directions. A ThoughtNode represents one possible "thought" or direction
/// you might explore.
///
/// Think of it like a choose-your-own-adventure book:
/// - Each page (node) presents a situation (the thought)
/// - Each page might have several choices that lead to different pages (children)
/// - You can trace back through your choices (parent links)
/// - Some paths lead to good endings (high scores), others to bad ones (low scores)
///
/// Tree-of-Thoughts reasoning builds a tree of these nodes, exploring different paths
/// and choosing the best ones. It's more sophisticated than just following one path
/// (Chain-of-Thought) because it can compare alternatives and backtrack if needed.
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Root node: the original problem
/// var root = new ThoughtNode&lt;double&gt;
/// {
///     Thought = "How can we reduce carbon emissions?",
///     Depth = 0
/// };
///
/// // Generate child nodes exploring different approaches
/// var child1 = new ThoughtNode&lt;double&gt;
/// {
///     Thought = "Focus on renewable energy adoption",
///     Parent = root,
///     Depth = 1,
///     EvaluationScore = 0.85
/// };
///
/// var child2 = new ThoughtNode&lt;double&gt;
/// {
///     Thought = "Improve transportation efficiency",
///     Parent = root,
///     Depth = 1,
///     EvaluationScore = 0.78
/// };
///
/// root.Children.Add(child1);
/// root.Children.Add(child2);
///
/// // Continue exploring the highest-scored path
/// var bestChild = root.Children.MaxBy(c => c.EvaluationScore);
/// </code>
/// </para>
/// </remarks>
public class ThoughtNode<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ThoughtNode{T}"/> class.
    /// </summary>
    public ThoughtNode()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        EvaluationScore = _numOps.Zero;
    }

    /// <summary>
    /// The thought or reasoning content at this node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the actual idea or reasoning step that this node represents.
    /// It should be clear and specific. For example:
    /// - "Convert 15% to decimal form"
    /// - "Consider using recursion to solve this problem"
    /// - "Look for patterns in the data"
    /// </para>
    /// </remarks>
    public string Thought { get; set; } = string.Empty;

    /// <summary>
    /// The parent node that led to this thought (null for root node).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This links back to the previous step in the reasoning path.
    /// By following parent links, you can trace back to see how you arrived at this thought.
    /// The root node (the original problem) has no parent.
    /// </para>
    /// </remarks>
    public ThoughtNode<T>? Parent { get; set; }

    /// <summary>
    /// Child nodes representing alternative next steps from this thought.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the different directions you could explore next
    /// from this point. If this is a dead end or final answer, there are no children.
    ///
    /// Multiple children allow the system to explore different approaches simultaneously
    /// and compare their effectiveness.
    /// </para>
    /// </remarks>
    public List<ThoughtNode<T>> Children { get; set; } = new();

    /// <summary>
    /// Depth level of this node in the tree (root = 0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many steps deep you are from the original problem:
    /// - Depth 0: The root (original problem)
    /// - Depth 1: First level of exploration
    /// - Depth 2: Second level
    /// - And so on...
    ///
    /// Depth is useful for:
    /// - Limiting how deep to search
    /// - Understanding complexity
    /// - Visualizing the tree structure
    /// </para>
    /// </remarks>
    public int Depth { get; set; }

    /// <summary>
    /// Quality score evaluating how promising this thought is (typically 0.0 to 1.0).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This score tells you how good or promising this thought is.
    /// Higher scores mean this thought is more likely to lead to a good solution.
    ///
    /// The score might consider:
    /// - How relevant this thought is to solving the problem
    /// - How logical or well-reasoned it is
    /// - How much progress it makes toward a solution
    /// - How well-supported it is by evidence or calculations
    ///
    /// Scores help the system decide which paths to explore further and which to abandon.
    /// </para>
    /// </remarks>
    public T EvaluationScore { get; set; }

    /// <summary>
    /// Whether this node has been visited during tree exploration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tracks whether we've already explored this node.
    /// It prevents wasting time re-exploring the same thought multiple times and helps
    /// detect cycles (going in circles).
    /// </para>
    /// </remarks>
    public bool IsVisited { get; set; }

    /// <summary>
    /// Whether this node represents a complete solution or terminal state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some nodes represent final answers rather than intermediate steps.
    /// Setting this to true indicates "we're done - this is the solution" and prevents
    /// further expansion from this node.
    ///
    /// For example, in a math problem, a terminal node might contain the final answer: "36"
    /// </para>
    /// </remarks>
    public bool IsTerminal { get; set; }

    /// <summary>
    /// The complete path from root to this node as a Vector of scores.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This gives you all the evaluation scores along the path
    /// from the original problem (root) to this current thought, stored as a Vector.
    ///
    /// For example, if you followed a path through nodes with scores [0.9, 0.85, 0.92],
    /// this vector would contain those three values.
    ///
    /// Useful for:
    /// - Analyzing the quality of the reasoning path
    /// - Identifying weak steps (low scores)
    /// - Calculating path statistics (average confidence, minimum score, etc.)
    /// </para>
    /// </remarks>
    public Vector<T> PathScores
    {
        get
        {
            var scores = new List<T>();
            var current = this;

            // Trace back to root, collecting scores
            while (current != null)
            {
                scores.Insert(0, current.EvaluationScore); // Insert at beginning to maintain order
                current = current.Parent;
            }

            return scores.Count > 0 ? new Vector<T>(scores) : new Vector<T>(0);
        }
    }

    /// <summary>
    /// Additional context or metadata specific to this thought.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A flexible container for any extra information about this thought,
    /// such as:
    /// - Which documents or sources support this thought
    /// - Alternative phrasings considered
    /// - Why this direction was chosen
    /// - Domain-specific data
    /// </para>
    /// </remarks>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Gets the complete path of thoughts from root to this node as strings.
    /// </summary>
    /// <returns>List of thought strings from root to current node.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This reconstructs the complete reasoning path that led to this node.
    /// It's like showing your complete work from problem to current step.
    ///
    /// Example output might be:
    /// ["What is 15% of 240?", "Convert 15% to decimal", "0.15 × 240", "Result is 36"]
    /// </para>
    /// </remarks>
    public List<string> GetPathFromRoot()
    {
        var path = new List<string>();
        var current = this;

        while (current != null)
        {
            path.Insert(0, current.Thought); // Insert at beginning to maintain order
            current = current.Parent;
        }

        return path;
    }

    /// <summary>
    /// Checks if this node is a leaf (has no children).
    /// </summary>
    /// <returns>True if this node has no children, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A "leaf" node is like the end of a branch on a tree -
    /// it has no further branches growing from it. In reasoning, leaf nodes are either:
    /// - Final answers (terminal nodes)
    /// - Dead ends that weren't worth exploring further
    /// - Nodes that haven't been expanded yet
    /// </para>
    /// </remarks>
    public bool IsLeaf() => Children.Count == 0;

    /// <summary>
    /// Checks if this node is the root (has no parent).
    /// </summary>
    /// <returns>True if this is the root node, false otherwise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The root is the starting point - usually the original
    /// problem or question. There's only one root in a tree.
    /// </para>
    /// </remarks>
    public bool IsRoot() => Parent == null;

    /// <summary>
    /// Gets the number of nodes in the complete path from root to this node.
    /// </summary>
    /// <returns>The path length (equal to depth + 1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This tells you how many reasoning steps it took to get here
    /// from the original problem, including the problem itself.
    ///
    /// For example, if Depth is 3, PathLength is 4 (root + 3 steps).
    /// </para>
    /// </remarks>
    public int PathLength => Depth + 1;

    /// <summary>
    /// Checks if this node appears to be terminal based on heuristics.
    /// </summary>
    /// <returns>True if the node is marked terminal or contains terminal indicators.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method uses simple heuristics to detect if a thought
    /// represents a final answer rather than an intermediate reasoning step.
    ///
    /// It checks for:
    /// - The IsTerminal flag being explicitly set
    /// - Keywords that typically indicate conclusions ("final answer", "conclusion", etc.)
    ///
    /// This is useful for search algorithms that need to identify when they've reached
    /// a complete solution and should stop expanding further.
    /// </para>
    /// </remarks>
    public bool CheckIsTerminalByHeuristic()
    {
        if (IsTerminal)
            return true;

        string thought = Thought.ToLowerInvariant();
        return thought.Contains("final answer") ||
               thought.Contains("conclusion") ||
               thought.Contains("therefore") ||
               thought.Contains("the answer is");
    }

    /// <summary>
    /// Returns a string representation of this thought node.
    /// </summary>
    /// <returns>A formatted string with depth and thought content.</returns>
    public override string ToString()
    {
        var prefix = new string(' ', Depth * 2); // Indent based on depth
        var terminal = IsTerminal ? " [TERMINAL]" : "";
        return $"{prefix}[Depth {Depth}, Score {EvaluationScore}]{terminal} {Thought}";
    }
}
