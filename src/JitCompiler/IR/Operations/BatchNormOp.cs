namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents batch normalization in the IR.
/// </summary>
public class BatchNormOp : IROp
{
    public double Epsilon { get; set; } = 1e-5;
    public double Momentum { get; set; } = 0.1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Input, gamma, beta, running_mean, running_var
        if (InputIds.Length != 5) return false;
        return true;
    }
}
