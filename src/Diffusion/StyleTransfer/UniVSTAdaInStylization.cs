using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.StyleTransfer;

/// <summary>
/// UniVST's training-free AdaIN-guided localized stylization (Song et al., arXiv:2410.20084,
/// TPAMI 2025), operating at BOTH the latent level and the attention level.
/// </summary>
/// <remarks>
/// <para>
/// Acting at only one of the two levels is the obvious simplification and the paper argues against
/// it: latent-level AdaIN alone injects colour statistics but loses localized detail, while
/// attention-level alignment alone does not move the global appearance far enough.
/// </para>
/// <para>
/// The three operations run on different schedules, which is the substance of the method:
/// latent AdaIN inside a narrow late window, query blending at EVERY timestep, and key/value AdaIN
/// on a linear ramp.
/// </para>
/// <para><b>For Beginners:</b> AdaIN restyles content by rescaling it to the style's per-channel mean
/// and spread. Doing it to the attention keys and values as well as the latent is what carries style
/// into fine detail rather than just recolouring the whole frame.</para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class UniVSTAdaInStylization<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Guards the division when a channel is constant, so a flat channel passes through.</summary>
    public const double Epsilon = 1e-8;

    private readonly UniVSTOptions _options;

    /// <summary>Creates the stylization helper.</summary>
    public UniVSTAdaInStylization(UniVSTOptions? options = null)
    {
        _options = options ?? new UniVSTOptions();
        _options.Validate();
    }

    /// <summary>
    /// Adaptive instance normalization: rescales <paramref name="content"/> to
    /// <paramref name="style"/>'s per-channel statistics.
    /// <c>sigma(s) * (c - mu(c)) / sigma(c) + mu(s)</c>.
    /// </summary>
    /// <param name="content">Content tensor, channel-first: [channels, ...].</param>
    /// <param name="style">Style tensor with the same channel count.</param>
    /// <remarks>
    /// Statistics are per CHANNEL over every remaining axis. A single global mean and variance would
    /// collapse the channels together and lose the colour relationships that carry the style.
    /// </remarks>
    public Tensor<T> AdaIn(Tensor<T> content, Tensor<T> style)
    {
        if (content == null) throw new ArgumentNullException(nameof(content));
        if (style == null) throw new ArgumentNullException(nameof(style));
        if (content.Shape.Length == 0 || style.Shape.Length == 0)
            throw new ArgumentException("AdaIN needs a channel axis; got a scalar tensor.");
        if (content.Shape[0] != style.Shape[0])
            throw new ArgumentException(
                $"Channel counts must match: content has {content.Shape[0]}, style has {style.Shape[0]}.",
                nameof(style));

        int channels = content.Shape[0];
        int contentPer = content.Length / channels;
        int stylePer = style.Length / channels;
        if (contentPer == 0 || stylePer == 0)
            throw new ArgumentException("Each channel must contain at least one element.");

        var result = new Tensor<T>(content.Shape.ToArray());

        // Per-channel reductions run once per denoising step over a latent, not per element inside a
        // training loop, so a direct pass is the clearer expression here.
        for (int c = 0; c < channels; c++)
        {
            int co = c * contentPer, so = c * stylePer;

            double cMean = 0.0;
            for (int i = 0; i < contentPer; i++) cMean += NumOps.ToDouble(content[co + i]);
            cMean /= contentPer;

            double cVar = 0.0;
            for (int i = 0; i < contentPer; i++)
            {
                double d = NumOps.ToDouble(content[co + i]) - cMean;
                cVar += d * d;
            }
            double cStd = Math.Sqrt(cVar / contentPer);

            double sMean = 0.0;
            for (int i = 0; i < stylePer; i++) sMean += NumOps.ToDouble(style[so + i]);
            sMean /= stylePer;

            double sVar = 0.0;
            for (int i = 0; i < stylePer; i++)
            {
                double d = NumOps.ToDouble(style[so + i]) - sMean;
                sVar += d * d;
            }
            double sStd = Math.Sqrt(sVar / stylePer);

            double scale = sStd / (cStd + Epsilon);
            for (int i = 0; i < contentPer; i++)
            {
                double v = ((NumOps.ToDouble(content[co + i]) - cMean) * scale) + sMean;
                result[co + i] = NumOps.FromDouble(v);
            }
        }

        return result;
    }

    /// <summary>
    /// True when latent-level AdaIN applies at the given progress through the schedule, expressed as
    /// a fraction of T.
    /// </summary>
    public bool IsLatentAdaInActive(double timestepFraction) =>
        timestepFraction >= _options.LatentAdaInStartFraction &&
        timestepFraction <= _options.LatentAdaInEndFraction;

    /// <summary>True when the key/value AdaIN ramp applies at the given fraction of T.</summary>
    public bool IsKeyValueAdaInActive(double timestepFraction) =>
        timestepFraction >= _options.KeyValueAdaInStartFraction &&
        timestepFraction <= _options.KeyValueAdaInEndFraction;

    /// <summary>
    /// beta at the given fraction of T: linear from <c>BetaAtRampStart</c> at the ramp start to
    /// <c>BetaAtRampEnd</c> at the ramp end, clamped to those values outside the ramp.
    /// </summary>
    /// <remarks>
    /// A constant blend is the tempting simplification and it defeats the design: early steps need
    /// mostly the raw style key/value to establish appearance, later steps mostly the AdaIN-aligned
    /// one to hold content.
    /// </remarks>
    public double BetaAt(double timestepFraction)
    {
        double t0 = _options.KeyValueAdaInStartFraction;
        double t1 = _options.KeyValueAdaInEndFraction;
        double b0 = _options.BetaAtRampStart;
        double b1 = _options.BetaAtRampEnd;

        if (timestepFraction <= t0) return b0;
        if (timestepFraction >= t1) return b1;

        // Degenerate ramp: a zero-width interval has no slope, so report the end value rather than
        // dividing by zero.
        double span = t1 - t0;
        if (span <= 0.0) return b1;

        return (((b1 - b0) / span) * (timestepFraction - t0)) + b0;
    }

    /// <summary>
    /// Query blending, applied at EVERY timestep:
    /// <c>gamma * editQuery + (1 - gamma) * contentQuery</c>.
    /// </summary>
    /// <remarks>
    /// Unlike the other two operations this has no window. The content query has to keep influencing
    /// attention throughout, or the stylized branch stops attending to the original structure.
    /// </remarks>
    public Tensor<T> BlendQuery(Tensor<T> editQuery, Tensor<T> contentQuery)
    {
        if (editQuery == null) throw new ArgumentNullException(nameof(editQuery));
        if (contentQuery == null) throw new ArgumentNullException(nameof(contentQuery));
        RequireSameShape(editQuery, contentQuery, nameof(contentQuery));

        double gamma = _options.QueryBlendGamma;
        var result = new Tensor<T>(editQuery.Shape.ToArray());
        for (int i = 0; i < editQuery.Length; i++)
        {
            double v = (gamma * NumOps.ToDouble(editQuery[i])) +
                       ((1.0 - gamma) * NumOps.ToDouble(contentQuery[i]));
            result[i] = NumOps.FromDouble(v);
        }
        return result;
    }

    /// <summary>
    /// Key/value blending on the ramp:
    /// <c>beta_t * AdaIN(editKv, styleKv) + (1 - beta_t) * styleKv</c>.
    /// </summary>
    /// <remarks>
    /// Note that the low-beta end leans on the RAW style key/value, not on the edit branch — the
    /// blend interpolates between "aligned to style" and "style itself".
    /// </remarks>
    public Tensor<T> BlendKeyValue(Tensor<T> editKv, Tensor<T> styleKv, double timestepFraction)
    {
        if (editKv == null) throw new ArgumentNullException(nameof(editKv));
        if (styleKv == null) throw new ArgumentNullException(nameof(styleKv));
        RequireSameShape(editKv, styleKv, nameof(styleKv));

        double beta = BetaAt(timestepFraction);
        var aligned = AdaIn(editKv, styleKv);

        var result = new Tensor<T>(editKv.Shape.ToArray());
        for (int i = 0; i < result.Length; i++)
        {
            double v = (beta * NumOps.ToDouble(aligned[i])) +
                       ((1.0 - beta) * NumOps.ToDouble(styleKv[i]));
            result[i] = NumOps.FromDouble(v);
        }
        return result;
    }

    /// <summary>
    /// Mask gating: <c>mask * edited + (1 - mask) * content</c>, keeping the stylization local.
    /// </summary>
    /// <param name="edited">The editing branch's latent, channel-first [channels, height, width].</param>
    /// <param name="content">The content branch's latent, same shape.</param>
    /// <param name="mask">A [height, width] mask, broadcast across channels.</param>
    public Tensor<T> ApplyMask(Tensor<T> edited, Tensor<T> content, Tensor<T> mask)
    {
        if (edited == null) throw new ArgumentNullException(nameof(edited));
        if (content == null) throw new ArgumentNullException(nameof(content));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        RequireSameShape(edited, content, nameof(content));

        int spatial = mask.Length;
        if (spatial == 0 || edited.Length % spatial != 0)
            throw new ArgumentException(
                $"Mask of {spatial} elements does not tile a latent of {edited.Length} elements.", nameof(mask));

        int channels = edited.Length / spatial;
        var result = new Tensor<T>(edited.Shape.ToArray());
        for (int c = 0; c < channels; c++)
        {
            int offset = c * spatial;
            for (int p = 0; p < spatial; p++)
            {
                double m = NumOps.ToDouble(mask[p]);
                double v = (m * NumOps.ToDouble(edited[offset + p])) +
                           ((1.0 - m) * NumOps.ToDouble(content[offset + p]));
                result[offset + p] = NumOps.FromDouble(v);
            }
        }
        return result;
    }

    private static void RequireSameShape(Tensor<T> a, Tensor<T> b, string paramName)
    {
        if (a.Shape.Length != b.Shape.Length)
            throw new ArgumentException(
                $"Rank mismatch: {a.Shape.Length} vs {b.Shape.Length}.", paramName);
        for (int i = 0; i < a.Shape.Length; i++)
            if (a.Shape[i] != b.Shape[i])
                throw new ArgumentException(
                    $"Shape mismatch at axis {i}: {a.Shape[i]} vs {b.Shape[i]}.", paramName);
    }
}
