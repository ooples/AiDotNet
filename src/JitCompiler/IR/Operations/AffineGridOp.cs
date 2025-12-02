namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents affine grid generation for spatial transformer in the IR.
/// </summary>
public class AffineGridOp : IROp
{
    public int[] OutputSize { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;  // theta (affine transformation matrix)
        return true;
    }
}
