namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents SoftPlus activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes SoftPlus(x) = ln(1 + exp(x)).
/// Smooth approximation of ReLU.
/// </para>
/// </remarks>
public class SoftPlusOp : IROp
{
    /// <summary>
    /// Scaling factor. Default is 1.0.
    /// </summary>
    public double Beta { get; set; } = 1.0;

    /// <summary>
    /// Threshold for switching to linear. Default is 20.0.
    /// </summary>
    public double Threshold { get; set; } = 20.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
