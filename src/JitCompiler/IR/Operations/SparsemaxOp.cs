namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Sparsemax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Like softmax but produces sparse outputs (some outputs exactly zero).
/// Useful when you want a hard-ish attention mechanism.
/// </para>
/// </remarks>
public class SparsemaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute sparsemax. Default is -1.
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
        return $"t{OutputId} = Sparsemax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
