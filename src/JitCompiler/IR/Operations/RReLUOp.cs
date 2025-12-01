namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents RReLU (Randomized Leaky ReLU) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Computes RReLU(x) = x if x >= 0, else alpha * x where alpha is randomly
/// sampled from uniform(lower, upper) during training.
/// </para>
/// </remarks>
public class RReLUOp : IROp
{
    /// <summary>
    /// Lower bound for random negative slope. Default is 0.125.
    /// </summary>
    public double Lower { get; set; } = 0.125;

    /// <summary>
    /// Upper bound for random negative slope. Default is 0.333.
    /// </summary>
    public double Upper { get; set; } = 0.333;

    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        if (Lower > Upper) return false;
        return true;
    }

    public override string ToString()
    {
        return $"t{OutputId} = RReLU(t{InputIds[0]}, lower={Lower}, upper={Upper}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
