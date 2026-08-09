using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// Attaches UniVST's attention-level steering to a frozen attention block: query blending at every
/// timestep, and key/value AdaIN on the beta ramp (Song et al., arXiv:2410.20084, TPAMI 2025).
/// </summary>
/// <remarks>
/// <para>
/// This is the attention half of UniVST's stylization. It has to act on the PROJECTED Q/K/V rather
/// than on the attention output, which is why it implements <see cref="IQkvTransform{T}"/> instead of
/// <see cref="IAttentionBlockDecorator{T}"/> — the latter only sees the post-softmax result, where a
/// Q/K/V-shaped operation would be transforming a different tensor entirely.
/// </para>
/// <para>
/// The reference tensors change every denoising step, because they come from the content and style
/// branches running alongside the editing branch. The driver therefore sets
/// <see cref="ContentQuery"/>, <see cref="StyleKey"/>, <see cref="StyleValue"/> and
/// <see cref="TimestepFraction"/> before each UNet call. A reference left null means "nothing to
/// blend against", and that projection passes through untouched rather than being blended against a
/// stale tensor from the previous step.
/// </para>
/// <para>
/// <b>No gradient tape.</b> UniVST is training-free — it steers a frozen model and never
/// backpropagates — so these transforms build their results directly instead of composing
/// <c>Engine</c> operations. Anything that needs gradients through attention must not use this.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class UniVSTQkvTransform<T> : IQkvTransform<T>
{
    private readonly UniVSTAdaInStylization<T> _stylization;

    /// <summary>Creates the transform.</summary>
    /// <param name="stylization">The AdaIN helper holding the schedule; defaults to paper settings.</param>
    public UniVSTQkvTransform(UniVSTAdaInStylization<T>? stylization = null)
    {
        _stylization = stylization ?? new UniVSTAdaInStylization<T>();
    }

    /// <summary>
    /// Gets or sets the current progress through the schedule, as a fraction of T. Drives the beta
    /// ramp and the key/value window.
    /// </summary>
    public double TimestepFraction { get; set; }

    /// <summary>
    /// Gets or sets the content branch's projected query for this step. When null, queries pass
    /// through unblended.
    /// </summary>
    public Tensor<T>? ContentQuery { get; set; }

    /// <summary>Gets or sets the style branch's projected key. When null, keys pass through.</summary>
    public Tensor<T>? StyleKey { get; set; }

    /// <summary>Gets or sets the style branch's projected value. When null, values pass through.</summary>
    public Tensor<T>? StyleValue { get; set; }

    /// <inheritdoc />
    /// <remarks>
    /// Applied at EVERY timestep — unlike the key/value path there is no window, because the content
    /// query has to keep influencing attention throughout or the stylized branch stops attending to
    /// the original structure.
    /// </remarks>
    public Tensor<T> TransformQuery(Tensor<T> query)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        var content = ContentQuery;
        if (content == null || content.Length != query.Length) return query;

        return _stylization.BlendQuery(query, content);
    }

    /// <inheritdoc />
    public Tensor<T> TransformKey(Tensor<T> key)
    {
        if (key == null) throw new ArgumentNullException(nameof(key));
        return BlendAgainst(key, StyleKey);
    }

    /// <inheritdoc />
    public Tensor<T> TransformValue(Tensor<T> value)
    {
        if (value == null) throw new ArgumentNullException(nameof(value));
        return BlendAgainst(value, StyleValue);
    }

    private Tensor<T> BlendAgainst(Tensor<T> projection, Tensor<T>? reference)
    {
        // Outside the ramp window the paper applies no key/value alignment at all, so the projection
        // must pass through rather than being blended at the clamped end-point beta.
        if (!_stylization.IsKeyValueAdaInActive(TimestepFraction)) return projection;
        if (reference == null || reference.Length != projection.Length) return projection;

        return _stylization.BlendKeyValue(projection, reference, TimestepFraction);
    }
}
