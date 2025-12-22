using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that support computing gradients with respect to input data.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface enables models to compute how their output changes with respect to input modifications.
/// Unlike <see cref="IGradientComputable{T, TInput, TOutput}"/> which computes gradients for model parameters,
/// this interface computes gradients for the input itself.
/// </para>
/// <para><b>Use Cases:</b></para>
/// <list type="bullet">
/// <item><description><b>Adversarial Examples:</b> Generate minimal perturbations that cause misclassification</description></item>
/// <item><description><b>Saliency Maps:</b> Visualize which input features most affect the output</description></item>
/// <item><description><b>Input Attribution:</b> Understand model predictions through input sensitivity</description></item>
/// <item><description><b>Gradient Penalties:</b> WGAN-GP and other regularization techniques</description></item>
/// </list>
/// <para><b>For Beginners:</b>
/// When training a model, we compute gradients to adjust the model's internal parameters (weights).
/// This interface instead computes how sensitive the output is to changes in the input data.
///
/// For example, if you have an image classifier:
/// - Parameter gradients tell us how to adjust weights to improve accuracy
/// - Input gradients tell us which pixels, if changed, would most affect the prediction
///
/// This is essential for adversarial robustness testing - we can find the smallest image change
/// that fools the classifier.
/// </para>
/// </remarks>
public interface IInputGradientComputable<T>
{
    /// <summary>
    /// Computes the gradient of the model output with respect to the input.
    /// </summary>
    /// <param name="input">The input for which to compute gradients.</param>
    /// <param name="outputGradient">The gradient with respect to the output (typically from a loss function).</param>
    /// <returns>The gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// This method performs backpropagation through the model to compute input gradients.
    /// The <paramref name="outputGradient"/> represents how much we "care" about each output dimension,
    /// typically derived from a loss function.
    /// </para>
    /// <para><b>For Adversarial Attacks:</b>
    /// Set <paramref name="outputGradient"/> to emphasize the target class (for targeted attacks)
    /// or the true class (for untargeted attacks), then use the returned input gradient
    /// to perturb the input in a direction that maximizes misclassification.
    /// </para>
    /// </remarks>
    Vector<T> ComputeInputGradient(Vector<T> input, Vector<T> outputGradient);

    /// <summary>
    /// Computes the gradient of the model output with respect to the input using tensor format.
    /// </summary>
    /// <param name="input">The input tensor for which to compute gradients.</param>
    /// <param name="outputGradient">The gradient tensor with respect to the output.</param>
    /// <returns>The gradient tensor with respect to the input.</returns>
    Tensor<T> ComputeInputGradient(Tensor<T> input, Tensor<T> outputGradient);
}
