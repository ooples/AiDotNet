using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Node in the speculation tree.
/// </summary>
/// <typeparam name="T">The numeric type for probabilities.</typeparam>
internal class TreeNode<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a new tree node with default probability of zero.
    /// </summary>
    public TreeNode()
    {
        Probability = NumOps.Zero;
        Children = new List<TreeNode<T>>();
    }

    /// <summary>
    /// Creates a new tree node with the specified probability.
    /// </summary>
    /// <param name="probability">The probability of this token.</param>
    public TreeNode(T probability)
    {
        Probability = probability;
        Children = new List<TreeNode<T>>();
    }

    /// <summary>
    /// The token at this node.
    /// </summary>
    public int Token { get; set; }

    /// <summary>
    /// Probability of this token.
    /// </summary>
    public T Probability { get; set; }

    /// <summary>
    /// Depth in the tree (0 = root).
    /// </summary>
    public int Depth { get; set; }

    /// <summary>
    /// Parent node (null for root).
    /// </summary>
    public TreeNode<T>? Parent { get; set; }

    /// <summary>
    /// Child nodes.
    /// </summary>
    public List<TreeNode<T>> Children { get; }

    /// <summary>
    /// Context tokens up to this node.
    /// </summary>
    public Vector<int>? Context { get; set; }
}
