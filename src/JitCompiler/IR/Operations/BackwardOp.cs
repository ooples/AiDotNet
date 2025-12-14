namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Base class for backward (gradient) operations in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Backward operations compute gradients during backpropagation for training.
/// Each forward operation has corresponding backward operation(s) that compute
/// the gradient with respect to its inputs.
/// </para>
/// <para><b>For Beginners:</b> These operations compute gradients for training.
///
/// In neural network training:
/// - Forward pass: Compute outputs from inputs
/// - Backward pass: Compute how to adjust weights to reduce error
///
/// Backward operations implement the chain rule of calculus to flow
/// gradients backward through the network.
/// </para>
/// </remarks>
public abstract class BackwardOp : IROp
{
    /// <summary>
    /// The tensor ID from the forward pass that may be needed for gradient computation.
    /// Many backward operations need the forward pass output or inputs.
    /// </summary>
    public int? SavedForwardTensorId { get; set; }
}
