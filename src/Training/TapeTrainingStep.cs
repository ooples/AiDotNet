using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Provides PyTorch-style training step using tape-based automatic differentiation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This class implements the standard training loop pattern from PyTorch:
/// <code>
/// // PyTorch equivalent:
/// optimizer.zero_grad()
/// output = model(input)
/// loss = criterion(output, target)
/// loss.backward()
/// optimizer.step()
///
/// // AiDotNet equivalent:
/// TapeTrainingStep.Step(layers, input, target, learningRate, forwardFn, lossFn);
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> This helper automates the training loop:
/// 1. Clears old gradients (zero_grad)
/// 2. Records the forward pass on a gradient tape
/// 3. Computes the loss
/// 4. Automatically computes gradients via reverse-mode AD (loss.backward)
/// 5. Updates parameters using SGD (optimizer.step)
/// </para>
/// </remarks>
public static class TapeTrainingStep<T>
{
    /// <summary>
    /// Executes a single training step using tape-based autodiff.
    /// </summary>
    /// <param name="layers">The trainable layers of the model.</param>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <param name="learningRate">The learning rate for SGD parameter updates.</param>
    /// <param name="forward">Function that runs the forward pass given the input.</param>
    /// <param name="computeLoss">Function that computes a scalar loss tensor from predicted and target.</param>
    /// <returns>The loss value as a scalar.</returns>
    public static T Step(
        IReadOnlyList<ITrainableLayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> computeLoss)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        // 1. Zero gradients (PyTorch: optimizer.zero_grad())
        foreach (var layer in layers)
        {
            layer.ZeroGrad();
        }

        // 2. Collect all trainable parameters
        var allParams = new List<Tensor<T>>();
        foreach (var layer in layers)
        {
            allParams.AddRange(layer.GetTrainableParameters());
        }
        var paramArray = allParams.ToArray();

        // 3. Forward pass + loss computation under tape recording
        Tensor<T> loss;
        Dictionary<Tensor<T>, Tensor<T>> grads;

        using (var tape = new GradientTape<T>())
        {
            var predicted = forward(input);
            loss = computeLoss(predicted, target);

            // 4. Compute gradients (PyTorch: loss.backward())
            grads = tape.ComputeGradients(loss, paramArray);
        }

        // 5. Update parameters with SGD (PyTorch: optimizer.step())
        foreach (var param in paramArray)
        {
            if (grads.TryGetValue(param, out var grad))
            {
                // param -= lr * grad (in-place SGD)
                for (int i = 0; i < param.Length && i < grad.Length; i++)
                {
                    param[i] = numOps.Subtract(
                        param[i],
                        numOps.Multiply(learningRate, grad[i]));
                }
            }
        }

        return loss.Length > 0 ? loss[0] : numOps.Zero;
    }

    /// <summary>
    /// Collects all trainable parameters from a sequence of layers.
    /// Equivalent to PyTorch's <c>model.parameters()</c>.
    /// </summary>
    /// <param name="layers">Layers to collect parameters from. Only layers
    /// implementing <see cref="ITrainableLayer{T}"/> contribute parameters.</param>
    /// <returns>Array of all trainable parameter tensors.</returns>
    public static Tensor<T>[] CollectParameters(IEnumerable<ILayer<T>> layers)
    {
        var parameters = new List<Tensor<T>>();
        foreach (var layer in layers)
        {
            if (layer is ITrainableLayer<T> trainable)
            {
                parameters.AddRange(trainable.GetTrainableParameters());
            }
        }
        return parameters.ToArray();
    }

    /// <summary>
    /// Zeros gradients for all trainable layers in a sequence.
    /// Equivalent to PyTorch's <c>optimizer.zero_grad()</c>.
    /// </summary>
    /// <param name="layers">Layers to zero gradients for.</param>
    public static void ZeroGradAll(IEnumerable<ILayer<T>> layers)
    {
        foreach (var layer in layers)
        {
            if (layer is ITrainableLayer<T> trainable)
            {
                trainable.ZeroGrad();
            }
        }
    }
}
