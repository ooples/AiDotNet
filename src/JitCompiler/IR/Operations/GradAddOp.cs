namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for AddOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: c = a + b
/// Backward: grad_a = grad_c, grad_b = grad_c
/// (gradient flows equally to both inputs)
/// </para>
/// </remarks>
public class GradAddOp : BackwardOp
{
    /// <summary>
    /// Which input are we computing the gradient for? (0 = left, 1 = right)
    /// </summary>
    public int InputIndex { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false; // Takes output gradient
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradAdd[input={InputIndex}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
