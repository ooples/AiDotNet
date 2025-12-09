namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for TransposeOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = transpose(x) or permute(x, axes)
/// Backward: grad_x = transpose(grad_y, inverse_axes)
/// </para>
/// </remarks>
public class GradTransposeOp : BackwardOp
{
    /// <summary>Axes used in forward transpose.</summary>
    public int[]? Axes { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        var axesStr = Axes != null ? string.Join(",", Axes) : "default";
        return $"t{OutputId} = GradTranspose[axes={axesStr}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
