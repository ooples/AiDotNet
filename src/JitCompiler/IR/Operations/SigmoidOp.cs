namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Sigmoid activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Sigmoid().
/// Computes sigmoid function: result[i] = 1 / (1 + exp(-a[i])).
/// Output range is (0, 1).
/// </para>
/// <para><b>For Beginners:</b> Squashes values to between 0 and 1.
///
/// Example:
/// Sigmoid([-∞, -2, 0, 2, ∞]) ≈ [0, 0.12, 0.5, 0.88, 1]
///
/// Used for binary classification (outputs can be interpreted as probabilities).
/// </para>
/// </remarks>
public class SigmoidOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
