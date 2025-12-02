namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for PadOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = pad(x, padding)
/// Backward: grad_x = slice(grad_y, unpad)
/// Gradient comes from the center (unpadded) region.
/// </para>
/// </remarks>
public class GradPadOp : BackwardOp
{
    /// <summary>Padding that was applied.</summary>
    public int[] Padding { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradPad[padding={string.Join(",", Padding)}](t{InputIds[0]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
