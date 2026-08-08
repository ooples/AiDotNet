namespace AiDotNet.MetaLearning.Components;

/// <summary>
/// The architecture constants specified by SImPa's paper, in one non-generic place.
/// </summary>
/// <remarks>
/// <para>
/// <b>NON-GENERIC ON PURPOSE.</b> These constants were previously declared on
/// <see cref="ImplicitPosteriorGenerator{T}"/> and <see cref="CompressionLemmaKLEstimator{T}"/>, which
/// meant every consumer had to name a type argument to reach a plain integer —
/// <c>ImplicitPosteriorGenerator&lt;double&gt;.PaperLatentDimension</c>. The <c>double</c> there carries
/// no meaning: 128 is 128 whatever the numeric type, and the reader has to stop and work out that the
/// type argument was picked arbitrarily rather than because the value depends on it. Worse, an options
/// class generic in its own <c>T</c> would appear to consult a <i>different</i> class's constant, so
/// somebody would eventually "fix" the mismatch by writing
/// <c>ImplicitPosteriorGenerator&lt;T&gt;.PaperLatentDimension</c> — which does not compile as a field
/// initializer default, having made a constant depend on a type parameter for no reason.
/// </para>
/// <para>
/// The generic classes keep their <c>Paper*</c> members as forwarding aliases so existing callers and
/// their default parameter values are unaffected; this type is simply where the numbers now live.
/// </para>
/// <para>
/// Source: Nguyen, Do and Carneiro, "PAC-Bayes meta-learning with implicit task-specific posteriors"
/// (arXiv:2003.02455).
/// </para>
/// </remarks>
public static class SImPaPaperConstants
{
    /// <summary>The paper's latent dimension: <c>z ~ U[0,1]^128</c>.</summary>
    public const int LatentDimension = 128;

    /// <summary>The paper's first generator hidden width.</summary>
    public const int FirstHiddenWidth = 256;

    /// <summary>The paper's second generator hidden width.</summary>
    public const int SecondHiddenWidth = 512;

    /// <summary>Monte Carlo samples the paper draws when training the phi-network per task.</summary>
    public const int MonteCarloSamples = 512;
}
