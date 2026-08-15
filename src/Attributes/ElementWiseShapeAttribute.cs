using System;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares that a layer changes VALUES but never SHAPE: whatever tensor goes in, a tensor of exactly
/// the same dimensions comes out, at any rank.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A "tensor" is just a box of numbers with a shape — for example
/// <c>[32, 3, 64, 64]</c> might be 32 images, 3 colour channels, 64 pixels tall, 64 wide. Most layers
/// change that shape: a pooling layer might turn <c>64x64</c> into <c>32x32</c>, and a dense layer might
/// turn 128 features into 10.
/// </para>
/// <para>
/// Some layers do not. Dropout randomly zeroes some numbers, layer normalization rescales them, a
/// stop-gradient layer passes them straight through. Every one of those touches the NUMBERS and leaves
/// the SHAPE alone. Put this attribute on a layer like that, and the shape system will know its output
/// shape always equals its input shape — without you writing anything else.
/// </para>
/// <para>
/// <b>Why this exists instead of using <c>[TensorLayout]</c>.</b> <c>[TensorLayout]</c> works by NAMING
/// each axis — Batch, Channels, Height, and so on — which is how the shape system reasons about
/// relationships like "the output height is the input height divided by two". That naming is essential
/// for a pooling or convolution layer. But a dropout layer genuinely does not care what its axes mean:
/// it works the same on <c>[128]</c>, on <c>[32, 128]</c>, and on <c>[8, 3, 64, 64]</c>. Forcing it to
/// name axes would mean inventing meanings it does not have, and inventing one rank when it accepts
/// many. Both are worse than saying nothing, because a wrong declaration is trusted just as much as a
/// right one.
/// </para>
/// <para>
/// <b>When NOT to use this.</b> If the layer changes ANY dimension — pooling, striding, flattening,
/// projecting to a different feature count, concatenating, reducing over an axis — this attribute is
/// wrong. Use <c>[TensorLayout]</c> and name the axes, so the system can work out how each output axis
/// relates to the input. If you are unsure, run the discovery sweep
/// (<c>LayerShapeDiscoverySweepTests</c>): it probes the layer with several shapes and reports what the
/// output actually does, which is more reliable than reasoning about it.
/// </para>
/// <example>
/// <code>
/// [ElementWiseShape]
/// public partial class DropoutLayer&lt;T&gt; : LayerBase&lt;T&gt;, IShapeContract
/// {
///     // Nothing else to write. OutputAxesFor is generated: every axis carried through, any rank.
/// }
/// </code>
/// </example>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ElementWiseShapeAttribute : Attribute
{
    /// <summary>
    /// Optional note explaining anything unusual, surfaced in diagnostics and documentation.
    /// </summary>
    public string? Note { get; set; }

    /// <summary>
    /// Highest rank the generated contract answers for. Defaults to 6, which covers every shape this
    /// library produces (batch + frames + channels + depth + height + width).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> "Rank" means how many dimensions a tensor has — <c>[128]</c> is rank 1,
    /// <c>[32, 128]</c> is rank 2. This is only an upper bound for the generated lookup; you will almost
    /// never need to change it.
    /// </remarks>
    public int MaxRank { get; set; } = 6;
}
