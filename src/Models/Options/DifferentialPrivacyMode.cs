namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies where differential privacy noise is applied in the federated learning pipeline.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Differential privacy can be applied at different points:
/// - Local DP: each client adds noise before sending updates (stronger protection vs server).
/// - Central DP: the server adds noise after aggregation (simpler and often higher utility).
/// - Both: apply local and central DP for defense-in-depth.
/// </remarks>
public enum DifferentialPrivacyMode
{
    /// <summary>
    /// No differential privacy is applied.
    /// </summary>
    None = 0,

    /// <summary>
    /// Apply noise on clients before sending updates.
    /// </summary>
    Local = 1,

    /// <summary>
    /// Apply noise on the server after aggregation.
    /// </summary>
    Central = 2,

    /// <summary>
    /// Apply both local and central differential privacy.
    /// </summary>
    LocalAndCentral = 3,

    /// <summary>
    /// Shuffle model DP: clients add local noise, a shuffler permutes updates before the server
    /// sees them, achieving central-DP-level accuracy with local-DP trust. (Balle et al., 2019)
    /// </summary>
    Shuffle = 4
}

