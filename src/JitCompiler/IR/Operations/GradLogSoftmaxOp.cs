namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for LogSoftmaxOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = log(softmax(x))
/// Backward: grad_x = grad_y - sum(grad_y) * softmax(x)
/// </para>
/// </remarks>
public class GradLogSoftmaxOp : BackwardOp
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
        return $"t{OutputId} = GradLogSoftmax[axis={Axis}](t{InputIds[0]}, t{InputIds[1]}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
