using AiDotNet.Tensors.Engines;

namespace AiDotNet.Attributes;

/// <summary>
/// Marks a tensor field as persistent NON-TRAINABLE state: saved with the model, never touched by
/// the optimizer.
/// </summary>
/// <remarks>
/// <para>
/// This is the equivalent of PyTorch's <c>register_buffer</c>, and it exists for the same reason:
/// some state must survive a save/load round trip without ever receiving a gradient.
/// BatchNormalization's running mean and variance, ALiBi's slope table, and an Echo State Network
/// reservoir's fixed weights are all buffers — a reloaded model that lost them predicts differently,
/// and nothing fails to say so.
/// </para>
/// <para>
/// A field marked this way is counted by <c>ParameterCount</c> and appears in
/// <c>GetParameters()</c>/<c>SetParameters()</c>, but is excluded from
/// <c>GetTrainableParameters()</c>, which is the set the optimizer and the gradient tape walk. That
/// separation is what makes it impossible to train a buffer by accident.
/// </para>
/// <para><b>For Beginners:</b> use this for numbers the layer needs to remember but should never
/// learn — running averages, lookup tables, fixed random weights. Use
/// <see cref="TrainableParameterAttribute"/> for weights the model should learn, and
/// <see cref="ScratchAttribute"/> for temporary values that do not need saving at all.</para>
/// <example>
/// <code>
/// public partial class MyNorm&lt;T&gt; : LayerBase&lt;T&gt;
/// {
///     private Tensor&lt;T&gt; _gamma;         // learned (the default)
///     [Buffer] private Tensor&lt;T&gt; _runningMean;   // saved, never learned
///     [Scratch] private Tensor&lt;T&gt; _batchScratch; // neither saved nor learned
/// }
/// </code>
/// </example>
/// </remarks>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class BufferAttribute : Attribute
{
    /// <summary>
    /// Name this buffer is stored under. Defaults to the field name with a leading underscore
    /// stripped.
    /// </summary>
    /// <remarks>
    /// Buffers are name-keyed (unlike trainable parameters, which are positional), so a stable name
    /// keeps a checkpoint readable when fields are reordered.
    /// </remarks>
    public string? Name { get; set; }

    /// <summary>
    /// GPU-residency / persistence role. Defaults to <c>Constant</c>, which is correct for state
    /// that is read every forward pass but never written by an optimizer.
    /// </summary>
    public PersistentTensorRole Role { get; set; } = PersistentTensorRole.Constant;
}
