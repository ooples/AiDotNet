namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Softmin activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes Softmin(x) = softmax(-x).
/// Similar to softmax but emphasizes smaller values.
/// </para>
/// </remarks>
public class SoftminOp : IROp
{
    /// <summary>
    /// The axis along which to compute softmin. Default is -1 (last axis).
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
        return $"t{OutputId} = Softmin(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
