namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Backward operation for GatherOp.
/// </summary>
/// <remarks>
/// <para>
/// Forward: y = gather(x, indices, axis)
/// Backward: grad_x = scatter(grad_y, indices, axis, shape)
/// </para>
/// </remarks>
public class GradGatherOp : BackwardOp
{
    /// <summary>Gather axis.</summary>
    public int Axis { get; set; }

    /// <summary>Original input shape.</summary>
    public int[] InputShape { get; set; } = Array.Empty<int>();

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false; // grad_output and indices
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = GradGather[axis={Axis}](...) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
