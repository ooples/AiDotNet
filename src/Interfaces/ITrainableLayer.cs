namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a neural network layer with trainable parameters that can be used with
/// tape-based automatic differentiation (autodiff).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This interface bridges layers with <see cref="AiDotNet.Autodiff.GradientTape{T}"/>
/// by exposing the exact <see cref="Tensor{T}"/> instances that the layer uses during
/// its forward pass. The gradient tape tracks operations by tensor reference identity,
/// so the tensors returned here must be the same objects passed to engine operations
/// in <see cref="ILayer{T}.Forward"/>.
/// </para>
/// <para>
/// <b>PyTorch equivalent:</b> This combines the role of <c>nn.Module.parameters()</c>
/// (which yields the same Parameter objects used in forward) and
/// <c>optimizer.zero_grad()</c> (which clears .grad on each parameter).
/// </para>
/// <para>
/// <b>For Beginners:</b> When training a neural network, we need two things:
/// <list type="number">
/// <item>
/// <description>
/// Access to the layer's learnable values (weights and biases) so the optimizer
/// can update them after computing gradients.
/// </description>
/// </item>
/// <item>
/// <description>
/// A way to clear old gradient information before each training step, so gradients
/// from different batches don't accumulate incorrectly.
/// </description>
/// </item>
/// </list>
/// This interface provides both capabilities, enabling the PyTorch-style training loop:
/// <code>
/// tape.Watch(layer.GetTrainableParameters());
/// var output = layer.Forward(input);
/// var loss = computeLoss(output, target);
/// var grads = tape.Gradient(loss);
/// optimizer.Step(layer.GetTrainableParameters(), grads);
/// layer.ZeroGrad();
/// </code>
/// </para>
/// </remarks>
public interface ITrainableLayer<T> : ILayer<T>
{
    /// <summary>
    /// Returns the trainable parameter tensors used by this layer during forward execution.
    /// </summary>
    /// <returns>
    /// An array of <see cref="Tensor{T}"/> instances that are the layer's trainable parameters.
    /// These must be the exact same object instances that participate in engine operations
    /// during <see cref="ILayer{T}.Forward"/>. An empty array if the layer has no trainable parameters.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>Critical contract:</b> The returned tensors must be reference-identical to the tensors
    /// used in the forward pass. The gradient tape keys gradients by <c>object.ReferenceEquals</c>,
    /// so returning copies or reconstructed tensors will prevent gradient lookup.
    /// </para>
    /// <para>
    /// For a typical dense layer with weights W and biases b, this returns <c>[W, b]</c> where
    /// W and b are the same fields used in <c>Engine.TensorMatMul(input, W)</c> and
    /// <c>Engine.TensorAdd(result, b)</c> during Forward.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This gives you direct access to the layer's learned values.
    /// Think of it like getting the actual knobs on a machine — not copies of the knob positions,
    /// but the real knobs themselves. When the optimizer adjusts these tensors, the layer
    /// immediately uses the updated values in its next forward pass.
    /// </para>
    /// </remarks>
    IReadOnlyList<Tensor<T>> GetTrainableParameters();

    /// <summary>
    /// Replaces this layer's trainable parameter tensors with the provided tensors.
    /// Used by <see cref="AiDotNet.Tensors.Engines.Autodiff.ParameterBuffer{T}"/> to replace
    /// independently-allocated tensors with views into a contiguous buffer.
    /// </summary>
    /// <param name="parameters">Replacement tensors, same count and shapes as
    /// <see cref="GetTrainableParameters"/> returns. These become the layer's actual parameters —
    /// the layer uses them in subsequent Forward calls.</param>
    /// <remarks>
    /// <para>
    /// The replacement tensors must have the same shapes as the originals. They are typically
    /// views into a <see cref="AiDotNet.Tensors.Engines.Autodiff.ParameterBuffer{T}"/>,
    /// enabling zero-copy flat parameter access for second-order optimizers.
    /// </para>
    /// <para><b>For Beginners:</b> This swaps out the layer's weight and bias tensors for new ones
    /// that share memory with a large contiguous buffer. The layer doesn't notice the difference —
    /// it uses the new tensors exactly like the old ones — but advanced optimizers can now access
    /// all parameters across all layers as a single flat vector without copying.</para>
    /// </remarks>
    void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters);

    /// <summary>
    /// Clears all accumulated gradients on this layer's trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This should be called at the start of each training step (before the forward pass)
    /// to prevent gradient accumulation across batches. This is equivalent to PyTorch's
    /// <c>optimizer.zero_grad()</c> or <c>model.zero_grad()</c>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Before each training step, we need to erase the gradient
    /// information from the previous step. Without this, gradients would pile up and
    /// confuse the optimizer about which direction to adjust the parameters.
    /// It's like erasing a whiteboard before solving a new problem.
    /// </para>
    /// </remarks>
    void ZeroGrad();
}
