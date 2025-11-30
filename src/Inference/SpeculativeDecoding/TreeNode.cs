namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Node in the speculation tree.
/// </summary>
internal class TreeNode
{
    public int Token { get; set; }
    public float Probability { get; set; }
    public int Depth { get; set; }
    public TreeNode? Parent { get; set; }
    public List<TreeNode> Children { get; } = [];
    public int[]? Context { get; set; }
}
