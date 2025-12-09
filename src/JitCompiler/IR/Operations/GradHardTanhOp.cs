namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for HardTanhOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = clip(x, min_val, max_val)
/// Backward: grad_x = grad_y if min_val &lt; x &lt; max_val, else 0
/// </para>
/// </remarks>
public class GradHardTanhOp : BackwardOp
{
    /// <summary>Minimum value used in forward.</summary>
    public double MinVal { get; set; } = -1.0;

    /// <summary>Maximum value used in forward.</summary>
    public double MaxVal { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradHardTanh[min={MinVal}, max={MaxVal}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
