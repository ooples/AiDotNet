namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// Thrown by <see cref="ContentSafetyMiddleware"/> when content is blocked and the middleware is configured to
/// fail hard (<see cref="ContentSafetyOptions.ThrowOnViolation"/>) rather than return a refusal.
/// </summary>
public sealed class ContentSafetyException : Exception
{
    /// <summary>
    /// Initializes a new exception describing why content was blocked.
    /// </summary>
    /// <param name="reason">The moderation reason.</param>
    public ContentSafetyException(string reason)
        : base(reason)
    {
        Reason = reason;
    }

    /// <summary>Gets the moderation reason the content was blocked.</summary>
    public string Reason { get; }
}
