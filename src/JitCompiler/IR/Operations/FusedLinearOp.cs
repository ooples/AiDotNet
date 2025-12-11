namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused linear operation (MatMul + Add bias).
/// </summary>
/// <remarks>
/// <para>
/// Combines matrix multiplication and bias addition into a single operation.
/// This is the fundamental operation of a neural network dense/linear layer.
/// </para>
/// <para><b>For Beginners:</b> This combines two operations into one.
///
/// Instead of:
///   t1 = MatMul(input, weights)  // Matrix multiply
///   t2 = Add(t1, bias)           // Add bias
///
/// We do:
///   t2 = Linear(input, weights, bias)  // One operation!
///
/// Benefits:
/// - Fewer memory reads/writes
/// - Better cache utilization
/// - Less overhead
/// - Typically 1.5-2x faster
/// </para>
/// </remarks>
public class FusedLinearOp : IROp
{
    /// <summary>
    /// Validates that this operation has correct inputs (3 inputs: input, weights, bias).
    /// </summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 3) return false;  // input, weights, bias
        return true;
    }
}
