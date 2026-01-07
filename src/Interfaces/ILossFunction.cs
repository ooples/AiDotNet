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
    /// Calculates the loss between predicted and actual tensors on GPU.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the model (on GPU).</param>
    /// <param name="actual">The actual (target) tensor (on GPU).</param>
    /// <returns>The loss value.</returns>
    T CalculateLossGpu(Tensor<T> predicted, Tensor<T> actual);

    /// <summary>
    /// Calculates the derivative (gradient) of the loss function on GPU.
    /// </summary>
    /// <param name="predicted">The predicted tensor from the model (on GPU).</param>
    /// <param name="actual">The actual (target) tensor (on GPU).</param>
    /// <returns>A tensor containing the derivatives of the loss with respect to each prediction (on GPU).</returns>
    Tensor<T> CalculateDerivativeGpu(Tensor<T> predicted, Tensor<T> actual);
}
