namespace AiDotNet.Audio.Speaker;

/// <summary>
/// A speaker match with score.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SpeakerMatch provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class SpeakerMatch
{
    /// <summary>
    /// Gets or sets the speaker ID.
    /// </summary>
    public string SpeakerId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the similarity score.
    /// </summary>
    public double Score { get; set; }
}
