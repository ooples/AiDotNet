using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// FIGAN's multi-scale perceptual objective: an L1 + VGG content loss evaluated at every scale, and
/// combined with a very small adversarial term.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// van Amersfoort et al., "Frame Interpolation with Multi-Scale Deep Loss Functions and Generative
/// Adversarial Networks" (arXiv:1711.06045).
/// </para>
/// <para><b>The coefficients are specified by the paper, not free parameters.</b></para>
/// <code>
///   Eq. 12   L = l_syn,x1 + 0.5 [ l_syn,x2 + l_syn,x3 ] + l_syn_refine + 0.0001 * l_GAN
///   Eq. 13   tau(a, b) = ||a - b||_1 + lambda_VGG ||gamma(a) - gamma(b)||_2^2
///                        with lambda_VGG = 0.001, gamma = VGG features at layer 5_4
/// </code>
/// <para>
/// Three details are easy to get wrong and all three matter:
/// </para>
/// <list type="number">
/// <item><description>The content term is <b>L1</b> plus a squared VGG distance — NOT MSE on pixels.
/// Substituting MSE changes what the network optimises for and is the usual default a reimplementation
/// falls into.</description></item>
/// <item><description>The finest scale carries weight <b>1.0</b> while the two coarser scales carry
/// <b>0.5</b>. Weighting all scales equally over-emphasises coarse structure.</description></item>
/// <item><description>The adversarial weight is <b>1e-4</b> — three orders of magnitude below the
/// content terms. It is a light corrective for realism, not a co-equal objective, and treating it as
/// one destabilises training.</description></item>
/// </list>
/// <para><b>For Beginners:</b> The model is graded on how close its invented frame is to the real one,
/// measured both pixel by pixel and through the eyes of a pretrained image network, at three levels of
/// detail at once — plus a small nudge from a critic trying to spot fakes.</para>
/// </remarks>
public static class FiganLoss<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>The paper's VGG weight in the content loss (Eq. 13).</summary>
    public const double VggWeight = 0.001;

    /// <summary>The paper's adversarial weight in the total loss (Eq. 12).</summary>
    public const double AdversarialWeight = 0.0001;

    /// <summary>The paper's weight on the two COARSER synthesis scales (Eq. 12).</summary>
    public const double CoarseScaleWeight = 0.5;

    /// <summary>
    /// Eq. 13's content loss between two images: mean absolute difference plus a weighted squared
    /// distance between their VGG features.
    /// </summary>
    /// <param name="predicted">The synthesised image.</param>
    /// <param name="target">The ground-truth image.</param>
    /// <param name="predictedFeatures">
    /// VGG layer-5_4 features of <paramref name="predicted"/>, or <c>null</c> to omit the perceptual
    /// term.
    /// </param>
    /// <param name="targetFeatures">VGG features of <paramref name="target"/>.</param>
    /// <remarks>
    /// The perceptual term is skipped only when features are absent, and the caller is then getting a
    /// PLAIN L1 loss — measurably not the paper's objective. It is optional rather than mandatory so
    /// the module is testable without standing up a VGG, not so it can be quietly left out.
    /// </remarks>
    public static double Content(
        Vector<T> predicted, Vector<T> target,
        Vector<T>? predictedFeatures = null, Vector<T>? targetFeatures = null)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (predicted.Length != target.Length)
            throw new ArgumentException(
                $"Predicted ({predicted.Length}) and target ({target.Length}) must be the same length.",
                nameof(predicted));

        // ||a - b||_1, averaged so the scale does not depend on image size.
        double l1 = 0.0;
        for (int i = 0; i < predicted.Length; i++)
            l1 += Math.Abs(Ops.ToDouble(predicted[i]) - Ops.ToDouble(target[i]));
        l1 /= predicted.Length;

        if (predictedFeatures is null || targetFeatures is null) return l1;

        if (predictedFeatures.Length != targetFeatures.Length)
            throw new ArgumentException(
                $"Feature vectors must match: {predictedFeatures.Length} vs {targetFeatures.Length}.",
                nameof(predictedFeatures));

        // lambda_VGG ||gamma(a) - gamma(b)||_2^2 — SQUARED, unlike the L1 pixel term.
        double squared = 0.0;
        for (int i = 0; i < predictedFeatures.Length; i++)
        {
            double d = Ops.ToDouble(predictedFeatures[i]) - Ops.ToDouble(targetFeatures[i]);
            squared += d * d;
        }
        squared /= predictedFeatures.Length;

        return l1 + (VggWeight * squared);
    }

    /// <summary>
    /// Eq. 12's total: the finest scale at full weight, the coarser scales at 0.5, the refined
    /// synthesis, and the adversarial term at 1e-4.
    /// </summary>
    /// <param name="finestScale">l_syn,x1 — content loss at the full-resolution synthesis.</param>
    /// <param name="coarserScales">
    /// l_syn,x2 and l_syn,x3. The paper uses two; any number is accepted and each is weighted 0.5, so
    /// a configuration with a different scale count stays consistent with the published form.
    /// </param>
    /// <param name="refinedSynthesis">l_syn_refine — the refinement branch's content loss.</param>
    /// <param name="adversarial">l_GAN — the generator's adversarial loss.</param>
    public static double Total(
        double finestScale,
        IReadOnlyList<double> coarserScales,
        double refinedSynthesis,
        double adversarial)
    {
        if (coarserScales is null) throw new ArgumentNullException(nameof(coarserScales));

        double coarse = 0.0;
        for (int i = 0; i < coarserScales.Count; i++) coarse += coarserScales[i];

        return finestScale
             + (CoarseScaleWeight * coarse)
             + refinedSynthesis
             + (AdversarialWeight * adversarial);
    }

    /// <summary>
    /// The generator's adversarial term, <c>log(1 - D(G(I_0, I_1)))</c> (Eq. 15).
    /// </summary>
    /// <param name="discriminatorOnFake">D applied to the synthesised frame, in (0, 1).</param>
    /// <remarks>
    /// The paper's literal form. Note it saturates when the discriminator is confident — the
    /// non-saturating <c>-log D(G(.))</c> variant trains better but is a DIFFERENT objective, so it is
    /// not substituted silently here. Clamped away from 1 so a certain discriminator yields a large
    /// finite value instead of negative infinity.
    /// </remarks>
    public static double GeneratorAdversarial(double discriminatorOnFake)
    {
        double d = Math.Min(Math.Max(discriminatorOnFake, 1e-12), 1.0 - 1e-12);
        return Math.Log(1.0 - d);
    }

    /// <summary>
    /// The discriminator's objective, <c>log D(real) + log(1 - D(fake))</c> (Eq. 16), which it
    /// MAXIMISES.
    /// </summary>
    public static double DiscriminatorObjective(double discriminatorOnReal, double discriminatorOnFake)
    {
        double real = Math.Min(Math.Max(discriminatorOnReal, 1e-12), 1.0 - 1e-12);
        double fake = Math.Min(Math.Max(discriminatorOnFake, 1e-12), 1.0 - 1e-12);
        return Math.Log(real) + Math.Log(1.0 - fake);
    }
}
