namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Leaky ReLU activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LeakyReLU(x) = max(alpha * x, x) where alpha is typically 0.01.
/// Allows small gradients for negative inputs.
/// </para>
/// </remarks>
public class LeakyReLUOp : IROp
{
    /// <summary>
    /// The negative slope. Default is 0.01.
    /// </summary>
    public double Alpha { get; set; } = 0.01;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LeakyReLU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
