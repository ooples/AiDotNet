namespace AiDotNet.Enums;

/// <summary>
/// Types of loss functions for preference optimization methods.
/// </summary>
/// <remarks>
/// <para>
/// Different preference optimization methods use different loss formulations.
/// Each has tradeoffs in terms of stability, sample efficiency, and robustness.
/// </para>
/// <para><b>For Beginners:</b> This controls how the model learns from preference data.
/// Start with Sigmoid (standard DPO) and only switch if you have specific needs.</para>
/// </remarks>
public enum PreferenceLossType
{
    /// <summary>
    /// Standard sigmoid loss used in DPO.
    /// </summary>
    /// <remarks>
    /// The original DPO loss: -log(sigmoid(beta * (log_p_chosen - log_p_rejected))).
    /// Works well in most cases and is the recommended default.
    /// </remarks>
    Sigmoid = 0,

    /// <summary>
    /// Hinge loss for preference optimization.
    /// </summary>
    /// <remarks>
    /// max(0, margin - (log_p_chosen - log_p_rejected)).
    /// More aggressive margin-based learning, can be unstable.
    /// </remarks>
    Hinge = 1,

    /// <summary>
    /// Identity Preference Optimization (IPO) loss.
    /// </summary>
    /// <remarks>
    /// Squared loss on log probability ratios.
    /// Addresses overfitting issues in DPO with noisy preferences.
    /// </remarks>
    IPO = 2,

    /// <summary>
    /// Robust DPO loss with outlier handling.
    /// </summary>
    /// <remarks>
    /// Modified sigmoid loss that's more robust to noisy preference labels.
    /// Useful when preference data quality is uncertain.
    /// </remarks>
    Robust = 3,

    /// <summary>
    /// Conservative DPO loss (cDPO).
    /// </summary>
    /// <remarks>
    /// Adds a conservative constraint to prevent over-optimization.
    /// Useful when concerned about reward hacking.
    /// </remarks>
    Conservative = 4,

    /// <summary>
    /// Odds ratio preference loss (used in ORPO).
    /// </summary>
    /// <remarks>
    /// Uses odds ratios instead of log probability differences.
    /// Combines SFT and preference optimization in a single objective.
    /// </remarks>
    OddsRatio = 5,

    /// <summary>
    /// Simple preference optimization loss (used in SimPO).
    /// </summary>
    /// <remarks>
    /// Reference-free loss based on length-normalized log probabilities.
    /// More memory efficient as it doesn't require a reference model.
    /// </remarks>
    Simple = 6,

    /// <summary>
    /// Kahneman-Tversky Optimization loss.
    /// </summary>
    /// <remarks>
    /// Based on prospect theory, handles unpaired preference data.
    /// Asymmetric treatment of gains (good responses) and losses (bad responses).
    /// </remarks>
    KTO = 7
}
