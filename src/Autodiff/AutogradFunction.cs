using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff;

/// <summary>
/// Base class for custom autograd functions with user-defined forward and backward passes.
/// Equivalent to PyTorch's <c>torch.autograd.Function</c>.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Subclass this to define operations with custom gradient computation. The forward method
/// performs the computation and saves any tensors needed for backward. The backward method
/// receives the output gradient and returns input gradients.
/// </para>
/// <para><b>Usage:</b>
/// <code>
/// public class MyCustomOp : AutogradFunction&lt;double&gt;
/// {
///     private Tensor&lt;double&gt;? _savedInput;
///
///     public override Tensor&lt;double&gt; Forward(AutogradContext ctx, params Tensor&lt;double&gt;[] inputs)
///     {
///         ctx.SaveForBackward(inputs[0]);
///         return Engine.TensorMultiply(inputs[0], inputs[0]); // x^2
///     }
///
///     public override Tensor&lt;double&gt;[] Backward(AutogradContext ctx, Tensor&lt;double&gt; gradOutput)
///     {
///         var x = ctx.GetSaved(0);
///         return [Engine.TensorMultiplyScalar(
///             Engine.TensorMultiply(gradOutput, x),
///             MathHelper.GetNumericOperations&lt;double&gt;().FromDouble(2.0))];
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public abstract class AutogradFunction<T>
{
    /// <summary>
    /// Performs the forward computation. Save tensors needed for backward using <paramref name="ctx"/>.
    /// </summary>
    /// <param name="ctx">Context for saving tensors needed during backward.</param>
    /// <param name="inputs">Input tensors.</param>
    /// <returns>Output tensor.</returns>
    public abstract Tensor<T> Forward(AutogradContext ctx, params Tensor<T>[] inputs);

    /// <summary>
    /// Computes gradients w.r.t. inputs given the gradient of the output.
    /// </summary>
    /// <param name="ctx">Context containing tensors saved during forward.</param>
    /// <param name="gradOutput">Gradient of the loss w.r.t. this function's output.</param>
    /// <returns>Gradients w.r.t. each input (same order as Forward inputs). Use null for non-differentiable inputs.</returns>
    public abstract Tensor<T>[] Backward(AutogradContext ctx, Tensor<T> gradOutput);

    /// <summary>
    /// Applies this custom function: runs forward and registers backward on the active tape.
    /// </summary>
    /// <param name="inputs">Input tensors.</param>
    /// <returns>Output tensor.</returns>
    public Tensor<T> Apply(params Tensor<T>[] inputs)
    {
        var ctx = new AutogradContext();
        var output = Forward(ctx, inputs);

        var tape = GradientTape<T>.Current;
        if (tape is not null)
        {
            var entry = new TapeEntry<T>
            {
                OperationName = GetType().Name,
                Output = output,
                Backward = (gradOutput, inputTensors, _, _, eng, grads) =>
                {
                    var inputGrads = Backward(ctx, gradOutput);
                    for (int i = 0; i < inputTensors.Length && i < inputGrads.Length; i++)
                    {
                        if (inputGrads[i] is not null)
                        {
                            if (grads.TryGetValue(inputTensors[i], out var existing))
                                eng.TensorAddInPlace(existing, inputGrads[i]);
                            else
                                grads[inputTensors[i]] = inputGrads[i];
                        }
                    }
                },
                InputCount = 0xFF,
                InputsOverflow = inputs,
            };
            if (inputs.Length > 0) entry.Input0 = inputs[0];
            if (inputs.Length > 1) entry.Input1 = inputs[1];
            if (inputs.Length > 2) entry.Input2 = inputs[2];
            tape.Record(entry);
        }

        return output;
    }
}

/// <summary>
/// Context object passed to <see cref="AutogradFunction{T}"/> for saving tensors between forward and backward.
/// Equivalent to PyTorch's <c>ctx</c> in autograd.Function.
/// </summary>
public sealed class AutogradContext
{
    private readonly List<object> _saved = [];

    /// <summary>
    /// Saves a tensor for use in the backward pass.
    /// </summary>
    public void SaveForBackward<T>(Tensor<T> tensor)
    {
        _saved.Add(tensor);
    }

    /// <summary>
    /// Saves multiple tensors for use in the backward pass.
    /// </summary>
    public void SaveForBackward<T>(params Tensor<T>[] tensors)
    {
        foreach (var t in tensors) _saved.Add(t);
    }

    /// <summary>
    /// Retrieves a saved tensor by index.
    /// </summary>
    public Tensor<T> GetSaved<T>(int index)
    {
        return (Tensor<T>)_saved[index];
    }

    /// <summary>
    /// Gets the number of saved tensors.
    /// </summary>
    public int SavedCount => _saved.Count;
}
