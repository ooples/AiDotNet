namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Thresholded ReLU activation in the IR.
/// </summary>
public class ThresholdedReLUOp : IROp
{
    /// <summary>
    /// The threshold value. Default is 1.0.
    /// </summary>
    public double Threshold { get; set; } = 1.0;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
