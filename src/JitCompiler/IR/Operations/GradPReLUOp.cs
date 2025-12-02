namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for PReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = max(0, x) + alpha * min(0, x)
/// Backward for x: grad_x = grad_y if x > 0, grad_y * alpha otherwise
/// Backward for alpha: grad_alpha = grad_y * min(0, x)
/// </para>
/// </remarks>
public class GradPReLUOp : BackwardOp
{
    /// <summary>Which input: 0 = input x, 1 = alpha parameter.</summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false; // grad_output, forward input, alpha
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradPReLU[input={InputIndex}](t{InputIds[0]}, t{InputIds[1]}, t{InputIds[2]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
