namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ReshapeOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = reshape(x, new_shape)
/// Backward: grad_x = reshape(grad_y, original_shape)
/// Reshape doesn't change data, just view, so gradient just reshapes back.
/// </para>
/// </remarks>
public class GradReshapeOp : BackwardOp
{
    /// <summary>Original shape before reshape.</summary>
    public int[] OriginalShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (OriginalShape.Length == 0) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradReshape[shape={string.Join(",", OriginalShape)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
