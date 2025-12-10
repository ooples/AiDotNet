namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents ISRU (Inverse Square Root Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes ISRU(x) = x / sqrt(1 + alpha * x^2).
/// Self-regularizing activation with bounded output.
/// </para>
/// </remarks>
public class ISRUOp : IROp
{
    /// <summary>
    /// The alpha parameter. Default is 1.0.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = ISRU(t{InputIds[0]}, alpha={Alpha}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
