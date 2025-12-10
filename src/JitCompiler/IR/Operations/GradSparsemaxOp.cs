namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SparsemaxOp.
/// </summary>
/// <remarks>
/// <para>
/// Sparsemax gradient is computed using the support set (non-zero outputs).
/// More complex than softmax gradient due to sparsity.
/// </para>
/// </remarks>
public class GradSparsemaxOp : BackwardOp
{
    /// <summary>Axis used in forward.</summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward output
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSparsemax[axis={Axis}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
