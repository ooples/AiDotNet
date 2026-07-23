namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// The verdict from moderating a piece of content: whether it is allowed, and if not, why.
/// </summary>
public sealed class ModerationVerdict
{
    private ModerationVerdict(bool allowed, string? reason)
    {
        Allowed = allowed;
        Reason = reason;
    }

    /// <summary>Gets a value indicating whether the content is permitted.</summary>
    public bool Allowed { get; }

    /// <summary>Gets the reason the content was blocked, or <c>null</c> when allowed.</summary>
    public string? Reason { get; }

    /// <summary>A verdict permitting the content.</summary>
    public static ModerationVerdict Allow() => new(allowed: true, reason: null);

    /// <summary>A verdict blocking the content with a reason.</summary>
    /// <param name="reason">Why the content was blocked.</param>
    public static ModerationVerdict Block(string reason)
    {
        Guard.NotNull(reason);
        return new ModerationVerdict(allowed: false, reason);
    }
}

/// <summary>
/// Checks whether a piece of text is safe/allowed — the seam guardrails use to screen agent inputs and
/// outputs. Implementations range from simple deny-lists to PII/jailbreak detectors or an
/// <c>src/Safety</c>-backed classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A safety checker. Give it some text and it says "fine" or "blocked, because…".
/// The guardrail middleware calls it on what users send in and on what the model sends back.
/// </para>
/// </remarks>
public interface IContentModerator
{
    /// <summary>
    /// Moderates a piece of content.
    /// </summary>
    /// <param name="content">The text to check.</param>
    /// <param name="cancellationToken">Token used to cancel the check.</param>
    /// <returns>The moderation verdict.</returns>
    Task<ModerationVerdict> CheckAsync(string content, CancellationToken cancellationToken = default);
}
