namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for DivideOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a / b
/// Backward: grad_a = grad_c / b, grad_b = -grad_c * a / (b^2)
/// </para>
/// </remarks>
public class GradDivideOp : BackwardOp
{
    /// <summary>Which input: 0 = numerator, 1 = denominator.</summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Needs grad_output and original inputs
        return InputIds.Length >= 2;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradDivide[input={InputIndex}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
