namespace AiDotNet.Enums;

/// <summary>
/// The method that certifies a prediction against bounded input perturbations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Randomized smoothing gives a probabilistic guarantee for any model; interval bound
/// propagation and CROWN give exact (deterministic) guarantees for networks whose layers they can bound.</para>
/// </remarks>
public enum CertifiedDefenseMethod
{
    /// <summary>
    /// Randomized smoothing with Gaussian noise (Cohen et al. 2019).
    /// </summary>
    RandomizedSmoothing,

    /// <summary>
    /// Interval bound propagation (Gowal et al. 2018).
    /// </summary>
    IntervalBoundPropagation,

    /// <summary>
    /// CROWN linear relaxation bounds (Zhang et al. 2018).
    /// </summary>
    Crown,

    /// <summary>
    /// Randomized smoothing whose noise level is chosen per input by a Gaussian probe step, certified with the
    /// composed Gaussian mechanism.
    /// </summary>
    AdaptiveRandomizedSmoothing
}
