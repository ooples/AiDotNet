namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Tanh (hyperbolic tangent) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Tanh().
/// Computes tanh function: result[i] = (exp(a[i]) - exp(-a[i])) / (exp(a[i]) + exp(-a[i])).
/// Output range is (-1, 1).
/// </para>
/// <para><b>For Beginners:</b> Squashes values to between -1 and 1.
///
/// Example:
/// Tanh([-∞, -2, 0, 2, ∞]) ≈ [-1, -0.96, 0, 0.96, 1]
///
/// Similar to sigmoid but centered at zero, often works better than sigmoid.
/// </para>
/// </remarks>
public class TanhOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
