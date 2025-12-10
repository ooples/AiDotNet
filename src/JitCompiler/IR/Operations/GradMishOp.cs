namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for MishOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x * tanh(softplus(x))
/// Backward: Complex derivative involving sech^2 and other terms
/// </para>
/// </remarks>
public class GradMishOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMish(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
