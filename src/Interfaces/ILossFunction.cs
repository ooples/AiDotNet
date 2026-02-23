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
/// The derivative of a loss function is equally important, as it tells the network which direction to adjust
/// its weights during training to reduce the loss.
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
    /// Calculates the derivative (gradient) of the loss function.
    /// </summary>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="actual">The actual (target) values.</param>
    /// <returns>A vector containing the derivatives of the loss with respect to each prediction.</returns>
    Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual);

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
    (T Loss, IGpuTensor<T> Gradient) CalculateLossAndGradientGpu(IGpuTensor<T> predicted, IGpuTensor<T> actual);
}
