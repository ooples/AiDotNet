namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for MatMulOp (left input).
/// </summary>
/// <remarks>
/// <para>
/// Forward: C = A @ B (matrix multiplication)
/// Backward for A: grad_A = grad_C @ B^T
/// </para>
/// </remarks>
public class GradMatMulLeftOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and right input (B)
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMatMulLeft(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
