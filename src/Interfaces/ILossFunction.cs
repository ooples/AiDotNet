using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for loss functions used in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Loss functions measure how far the predictions of a neural network are from the expected outputs.
/// They provide a signal that helps the network learn by adjusting its weights to minimize this "loss" value.
/// 
/// Think of a loss function as a score that tells you how well or poorly your neural network is performing.
/// A higher loss value means worse performance, while a lower loss value indicates better performance.
/// 
/// Different types of problems require different loss functions. For example:
/// - Mean Squared Error is often used for regression problems (predicting numeric values)
/// - Cross Entropy is commonly used for classification problems (categorizing inputs)
///
/// Training also needs the loss's <i>derivative</i> — the direction to adjust weights in. You do not
/// write that derivative yourself. Implement <see cref="ComputeTapeLoss"/> with engine operations and
/// the gradient tape differentiates it for you, exactly as PyTorch differentiates an
/// <c>nn.Module.forward</c>. A loss defines its forward math once; the tape supplies the backward.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("LossFunction")]
public interface ILossFunction<T>
{
    /// <summary>
    /// Calculates the loss between predicted and actual values.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>The loss value.</returns>
    T CalculateLoss(Vector<T> predicted, Vector<T> actual);

    /// <summary>
    /// Computes the loss as a scalar tensor using tape-differentiable engine operations, so that
    /// gradients flow through it automatically via <c>GradientTape</c>.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the forward pass.</param>
    /// <param name="target">The target tensor.</param>
    /// <returns>A rank-0 scalar tensor containing the loss value.</returns>
    /// <remarks>
    /// <para>
    /// This is the <b>only</b> place a loss function defines its math. There is deliberately no
    /// member on this interface for a hand-written derivative: gradients come from the tape, which
    /// makes it impossible for an analytic backward to drift out of step with the forward it is
    /// supposed to differentiate. Callers obtain gradients either by recording this call on their
    /// own training tape — the preferred form, since one tape then spans forward and loss together —
    /// or via <see cref="LossFunctionExtensions.ComputeGradient{T}"/> for a standalone gradient.
    /// </para>
    /// <para>
    /// Implementations must return a <b>rank-0</b> tensor. A rank-1 <c>[1]</c> result leaves the tape
    /// without a scalar to seed the backward pass from, and silently produces no gradients.
    /// </para>
    /// <para><b>For Beginners:</b> Write the formula for your loss using tensor operations and stop
    /// there. The library works out the calculus for you.
    /// </para>
    /// </remarks>
    Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target);

    /// <summary>
    /// Calculates both loss and gradient on GPU in a single pass.
    /// </summary>
    /// <param name="predicted">The predicted GPU tensor from the model.</param>
    /// <param name="actual">The actual (target) GPU tensor.</param>
    /// <returns>A tuple containing the loss value and gradient tensor.</returns>
    /// <remarks>
    /// This method is more efficient than calling separate loss and gradient calculations
    /// as it can compute both in a single GPU kernel invocation.
    /// </remarks>
    (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual);
}
