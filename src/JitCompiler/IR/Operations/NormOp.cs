namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents L2 norm operation in the IR.
/// </summary>
public class NormOp : IROp
{
    /// <summary>
    /// The axis along which to compute the norm.
    /// </summary>
    public int Axis { get; set; } = -1;

    /// <summary>
    /// Whether to keep the reduced dimension.
    /// </summary>
    public bool KeepDims { get; set; } = false;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
