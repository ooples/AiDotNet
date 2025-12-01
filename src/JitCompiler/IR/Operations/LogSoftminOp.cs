namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Log Softmin activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LogSoftmin(x) = log(softmin(x)).
/// Numerically stable version of softmin in log space.
/// </para>
/// </remarks>
public class LogSoftminOp : IROp
{
    /// <summary>
    /// The axis along which to compute log softmin. Default is -1 (last axis).
    /// </summary>
    public int Axis { get; set; } = -1;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = LogSoftmin(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
