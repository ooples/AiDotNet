namespace AiDotNet.Attributes;

/// <summary>
/// Opts a layer into automatic parameter discovery: every non-nullable tensor field is a trainable
/// parameter unless it says otherwise.
/// </summary>
/// <remarks>
/// <para>
/// Without this, discovery is opt-IN — a field only becomes a parameter if the author remembers
/// <see cref="TrainableParameterAttribute"/> or a <c>RegisterTrainableParameter</c> call. Opting in
/// has failed in practice. <c>RWKVLayer</c> registers its eight weight matrices and silently omits
/// ten more learned tensors: the time- and channel-mixing coefficients that give RWKV its name, the
/// first-token bonus, and both LayerNorm affine pairs. The optimizer never updates them, so the
/// layer only partly trains and nothing reports it.
/// </para>
/// <para>
/// With this attribute the default inverts, so forgetting is no longer possible — the exceptions
/// declare themselves instead:
/// </para>
/// <list type="bullet">
///   <item><description><c>Tensor&lt;T&gt;?</c> (nullable) — excluded. This is the cache shape
///     (<c>_lastInput</c>, <c>_lastOutput</c>), and it already could not be registered.</description></item>
///   <item><description><c>*Gradient</c> — excluded by the existing naming convention.</description></item>
///   <item><description><c>static</c> — excluded.</description></item>
///   <item><description><see cref="ScratchAttribute"/> — excluded explicitly.</description></item>
///   <item><description><see cref="BufferAttribute"/> — persistent but never trained.</description></item>
/// </list>
/// <para>
/// Explicit always beats the default: a field carrying <see cref="TrainableParameterAttribute"/> or
/// registered through <c>RegisterTrainableParameter</c> keeps exactly the role it was given,
/// including the handful of genuinely nullable parameters that exist
/// (<c>DeformableConvolutionalLayer._maskWeights</c>, <c>MessagePassingLayer._edgeWeights</c>).
/// </para>
/// <para>
/// It is per-class so the inversion can be adopted and verified one layer at a time rather than
/// flipped across the whole library at once. Once every layer carries it, the attribute becomes the
/// default and disappears.
/// </para>
/// <para><b>For Beginners:</b> add this to your layer and you never write parameter plumbing —
/// declare a tensor field and it is learned, saved and restored automatically.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class AutoParametersAttribute : Attribute
{
}
