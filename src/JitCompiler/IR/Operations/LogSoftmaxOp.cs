namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents LogSoftmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes LogSoftmax(x) = log(softmax(x)).
/// Numerically stable for cross-entropy loss.
/// </para>
/// </remarks>
public class LogSoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute log softmax. Default is -1 (last axis).
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
        return $"t{OutputId} = LogSoftmax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
