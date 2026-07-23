using AiDotNet.Tensors;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for activation functions that can compute their derivative
/// given the post-activation output value rather than the pre-activation input.
/// This avoids the common bug where the derivative re-applies the activation
/// (e.g., sigmoid(sigmoid(x)) instead of sigmoid(x)*(1-sigmoid(x))).
/// </summary>
public interface IOutputDerivative<T>
{
    /// <summary>
    /// Computes the activation derivative given the post-activation output value.
    /// For sigmoid: output * (1 - output)
    /// For tanh: 1 - output²
    /// </summary>
    Tensor<T> DerivativeFromOutput(Tensor<T> output);
}
