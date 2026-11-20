namespace AiDotNet.Enums;

/// <summary>
/// The norm that bounds an adversarial perturbation or a robustness certificate.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A norm measures how big a change to the input is. L-infinity limits the
/// largest change to any single feature; L2 limits the overall (Euclidean) size of the change.</para>
/// </remarks>
public enum AdversarialNormType
{
    /// <summary>
    /// The L-infinity norm: the largest absolute change of any single input feature.
    /// </summary>
    LInfinity,

    /// <summary>
    /// The L2 (Euclidean) norm: the square root of the sum of squared changes.
    /// </summary>
    L2
}
