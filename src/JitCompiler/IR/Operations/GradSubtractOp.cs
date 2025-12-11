namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SubtractOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a - b
/// Backward: grad_a = grad_c, grad_b = -grad_c
/// </para>
/// </remarks>
public class GradSubtractOp : BackwardOp
{
    /// <summary>
    /// Which input are we computing the gradient for? (0 = left, 1 = right)
    /// </summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSubtract[input={InputIndex}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
