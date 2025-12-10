namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for MatMulOp (right input).
/// </summary>
/// <remarks>
/// <para>
/// Forward: C = A @ B (matrix multiplication)
/// Backward for B: grad_B = A^T @ grad_C
/// </para>
/// </remarks>
public class GradMatMulRightOp : BackwardOp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // left input (A) and grad_output
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradMatMulRight(t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
