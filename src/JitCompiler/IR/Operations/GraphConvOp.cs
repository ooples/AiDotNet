namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents graph convolution in the IR.
/// </summary>
public class GraphConvOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // features, adjacency_matrix, weights
        if (InputIds.Length != 3) return false;
        return true;
    }
}
