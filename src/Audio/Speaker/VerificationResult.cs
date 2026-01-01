namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Result of speaker verification.
/// </summary>
public class VerificationResult
{
    /// <summary>
    /// Gets or sets the claimed speaker ID.
    /// </summary>
    public string ClaimedSpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the verification succeeded.
    /// </summary>
    public bool IsVerified { get; set; }

    /// <summary>
    /// Gets or sets the similarity score.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets the threshold used for verification.
    /// </summary>
    public double Threshold { get; set; }

    /// <summary>
    /// Gets or sets any error message.
    /// </summary>
    public string? ErrorMessage { get; set; }
}
