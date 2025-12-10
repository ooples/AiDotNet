namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents RBF (Radial Basis Function) kernel computation in the IR.
/// </summary>
public class RBFKernelOp : IROp
{
    public double Gamma { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;  // x, centers
        return true;
    }
}
