namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents dropout operation in the IR.
/// </summary>
public class DropoutOp : IROp
{
    /// <summary>
    /// Dropout probability.
    /// </summary>
    public double Probability { get; set; } = 0.5;

    /// <summary>
    /// Whether in training mode.
    /// </summary>
    public bool Training { get; set; } = true;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
