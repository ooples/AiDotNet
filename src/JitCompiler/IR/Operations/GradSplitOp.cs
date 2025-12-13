namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for SplitOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: [y1, y2, ...] = split(x, sizes, axis)
/// Backward: grad_x = concat([grad_y1, grad_y2, ...], axis)
/// </para>
/// </remarks>
public class GradSplitOp : BackwardOp
{
    /// <summary>Split axis.</summary>
    public int Axis { get; set; }

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length < 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradSplit[axis={Axis}]({string.Join(", ", InputIds.Select(id => $"t{id}"))}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
