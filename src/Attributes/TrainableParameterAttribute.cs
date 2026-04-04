using AiDotNet.Tensors.Engines;

namespace AiDotNet.Attributes;

/// <summary>
/// Marks a <see cref="AiDotNet.Tensors.LinearAlgebra.Tensor{T}"/> field as a trainable parameter
/// that should be registered with the gradient tape training system.
/// </summary>
/// <remarks>
/// <para>
/// The <c>TrainableParameterGenerator</c> source generator discovers fields marked with this
/// attribute and automatically emits:
/// <list type="bullet">
/// <item><c>GetTrainableParameters()</c> — returns all marked fields in declaration order</item>
/// <item><c>SetTrainableParameters(Tensor&lt;T&gt;[])</c> — updates each marked field from the array</item>
/// <item><c>ZeroGrad()</c> — zeros gradient fields discovered by convention ({fieldName}Gradient)</item>
/// </list>
/// </para>
/// <para>
/// This is the equivalent of PyTorch's <c>nn.Parameter</c> — marking a tensor as trainable
/// makes it automatically visible to the optimizer and gradient tape with zero manual boilerplate.
/// </para>
/// <para><b>Convention for gradient fields:</b> For a parameter field named <c>_weights</c>,
/// the generator looks for <c>_weightsGradient</c> (nullable <c>Tensor&lt;T&gt;?</c>).
/// If found, <c>ZeroGrad()</c> will zero or null it automatically.</para>
/// <para><b>For Beginners:</b> Put this attribute on any tensor field that the network should
/// learn during training. The framework handles everything else — registering it, exposing it
/// to the optimizer, and clearing gradients between training steps.</para>
/// </remarks>
/// <example>
/// <code>
/// public partial class MyLayer&lt;T&gt; : LayerBase&lt;T&gt;
/// {
///     [TrainableParameter(Role = PersistentTensorRole.Weights)]
///     private Tensor&lt;T&gt; _weights;
///
///     [TrainableParameter(Role = PersistentTensorRole.Biases)]
///     private Tensor&lt;T&gt; _biases;
///
///     private Tensor&lt;T&gt;? _weightsGradient;  // auto-discovered by convention
///     private Tensor&lt;T&gt;? _biasesGradient;   // auto-discovered by convention
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class TrainableParameterAttribute : Attribute
{
    /// <summary>
    /// Gets or sets the role of this parameter for GPU memory management hints.
    /// Defaults to <see cref="PersistentTensorRole.Weights"/>.
    /// </summary>
    public PersistentTensorRole Role { get; set; } = PersistentTensorRole.Weights;

    /// <summary>
    /// Gets or sets the display order of this parameter in <c>GetTrainableParameters()</c>.
    /// Parameters are sorted by Order, then by declaration order.
    /// Defaults to 0 (declaration order).
    /// </summary>
    public int Order { get; set; }
}
