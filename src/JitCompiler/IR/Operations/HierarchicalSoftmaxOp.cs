namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Hierarchical Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Organizes classes into a tree structure for efficient computation.
/// Reduces complexity from O(V) to O(log V) for large vocabularies.
/// </para>
/// </remarks>
public class HierarchicalSoftmaxOp : IROp
{
    /// <summary>
    /// The hierarchy tree structure (encoded).
    /// </summary>
    public int[] TreeStructure { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
