namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Hard Tanh activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes HardTanh(x) = clip(x, -1, 1).
/// Faster piecewise linear approximation of tanh.
/// </para>
/// </remarks>
public class HardTanhOp : IROp
{
    /// <summary>
    /// Minimum value. Default is -1.0.
    /// </summary>
    public double MinVal { get; set; } = -1.0;

    /// <summary>
    /// Maximum value. Default is 1.0.
    /// </summary>
    public double MaxVal { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
