namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the gradient of octonion multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// For octonion product c = a * b, the gradients are computed using:
/// - d_a = d_c * b^* (right multiplication by conjugate of b)
/// - d_b = a^* * d_c (left multiplication by conjugate of a)
/// where ^* denotes octonion conjugate.
/// </para>
/// <para><b>For Beginners:</b> When training with octonions, we need to compute
/// how changes in inputs affect the output. The gradient tells us this information
/// for backpropagation through the octonion multiplication.
/// </para>
/// </remarks>
public class GradOctonionMultiplyOp : IROp
{
    /// <summary>
    /// Expected number of outputs: d_a and d_b.
    /// </summary>
    public const int ExpectedOutputCount = 2;

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: a, b, gradient of output (d_c)
        if (InputIds.Length != 3) return false;
        // Outputs: d_a (gradient w.r.t. a), d_b (gradient w.r.t. b)
        if (OutputIds.Length != ExpectedOutputCount) return false;
        return true;
    }
}
