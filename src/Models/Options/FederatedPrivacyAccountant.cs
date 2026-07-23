namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies which privacy accountant to use for reporting privacy spend in federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> A privacy accountant tracks how much privacy budget has been spent over rounds.
/// Different accountants can report tighter (less pessimistic) bounds depending on assumptions.
/// </remarks>
public enum FederatedPrivacyAccountant
{
    /// <summary>
    /// Basic composition of (ε, δ) over rounds.
    /// </summary>
    Basic = 0,

    /// <summary>
    /// Rényi Differential Privacy (RDP) accounting (recommended when using Gaussian mechanisms).
    /// </summary>
    Rdp = 1
}

