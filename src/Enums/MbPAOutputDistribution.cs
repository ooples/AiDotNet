namespace AiDotNet.Enums;

/// <summary>
/// The output distribution MbPA's local adaptation fits, i.e. what <c>log p(v | h, theta_x)</c>
/// means in its objective.
/// </summary>
/// <remarks>
/// <para>
/// MbPA's local adaptation maximizes a weighted log-likelihood over the retrieved neighbours
/// (Sprechmann et al., arXiv:1802.10542). Naming the distribution is what turns that expression into
/// a concrete gradient.
/// </para>
/// <para>
/// Both members share the same gradient form on a linear output network — <c>(prediction - target)
/// (x) h</c> — which is why the local step is exact rather than approximated in either case.
/// </para>
/// </remarks>
public enum MbPAOutputDistribution
{
    /// <summary>
    /// Softmax over the output network's logits with a cross-entropy log-likelihood. This is what
    /// both of the paper's task families use — image classification and language modelling.
    /// </summary>
    Categorical = 0,

    /// <summary>
    /// A unit-variance Gaussian, so the log-likelihood is the negative squared error. For regression
    /// targets, where a softmax over outputs would be meaningless.
    /// </summary>
    Gaussian = 1,
}
