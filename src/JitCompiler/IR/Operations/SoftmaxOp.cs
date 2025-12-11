namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Softmax activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Softmax().
/// Computes softmax along specified axis. Converts logits to probabilities.
/// </para>
/// <para><b>For Beginners:</b> Converts scores to probabilities that sum to 1.
///
/// Example:
/// Softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
/// (notice they sum to 1.0)
///
/// Used for multi-class classification - outputs can be interpreted as
/// class probabilities.
/// </para>
/// </remarks>
public class SoftmaxOp : IROp
{
    /// <summary>
    /// The axis along which to compute softmax. Default is -1 (last axis).
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
        return $"t{OutputId} = Softmax(t{InputIds[0]}, axis={Axis}) : {OutputType} {OutputShape.ShapeToString()}";
    }
}
