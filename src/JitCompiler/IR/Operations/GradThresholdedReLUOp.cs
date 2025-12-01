namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for ThresholdedReLUOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = x if x > threshold, 0 otherwise
/// Backward: grad_x = grad_y if x > threshold, 0 otherwise
/// </para>
/// </remarks>
public class GradThresholdedReLUOp : BackwardOp
{
    /// <summary>Threshold used in forward.</summary>
    public double Threshold { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and forward input
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradThresholdedReLU[threshold={Threshold}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
