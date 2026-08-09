namespace AiDotNet.Enums;

/// <summary>
/// Which likelihood MbPA's output head models, i.e. what <c>log p(v | h, theta_x)</c> means.
/// </summary>
/// <remarks>
/// <para>
/// <b>This enum was referenced from four files and declared in none of them.</b>
/// <c>MbPAOptions.OutputDistribution</c>, <c>MbPAOutputNetwork</c> and <c>LFTAlgorithm</c> all named
/// it, so the type was clearly intended; it simply never got written, and the whole project failed
/// to compile on twelve <c>CS0246</c>s pointing at a name that existed nowhere in the tree.
/// </para>
/// <para>
/// <b>The two members are not an arbitrary pair.</b> They are exactly the cases for which a linear
/// head has the SAME closed-form gradient, <c>w (prediction - target) (x) h</c> — softmax composed
/// with cross entropy, and a unit-variance Gaussian composed with squared error. MbPA's local
/// adaptation depends on that closed form, which is why it can take exact gradient steps on the
/// retrieved neighbours without an autodiff graph. Adding a third distribution here without also
/// giving <c>MbPAOutputNetwork.Gradient</c> its residual would silently reuse one of these two.
/// </para>
/// <para>
/// Source: Sprechmann et al., "Memory-based Parameter Adaptation" (arXiv:1802.10542).
/// </para>
/// </remarks>
public enum MbPAOutputDistribution
{
    /// <summary>
    /// Softmax over classes with a cross-entropy log-likelihood. Both of the paper's task families —
    /// image classification and language modelling — are this case.
    /// </summary>
    /// <remarks>Explicit ordinal: this value is serialized, so it must not move if a member is ever inserted above it.</remarks>
    Categorical = 0,

    /// <summary>
    /// A unit-variance Gaussian, whose negative log-likelihood is one-half squared error up to a
    /// constant. Use this for regression targets.
    /// </summary>
    /// <remarks>Explicit ordinal, for the reason given on <see cref="Categorical"/>.</remarks>
    Gaussian = 1,
}
