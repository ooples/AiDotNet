namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// Settings for <see cref="ContentSafetyMiddleware"/>: which sides to screen, what to say when blocking, and
/// whether a violation throws or returns a refusal.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The dials for the safety filter — check the user's message, the model's reply,
/// or both; the message to show when something is blocked; and whether a block is a polite refusal (default)
/// or a hard error.
/// </para>
/// </remarks>
public sealed class ContentSafetyOptions
{
    /// <summary>The default message returned when content is blocked.</summary>
    public const string DefaultRefusalMessage = "I'm sorry, but I can't help with that request.";

    /// <summary>Gets or sets whether to screen the user input before calling the model. Default <c>true</c>.</summary>
    public bool CheckInput { get; set; } = true;

    /// <summary>Gets or sets whether to screen the model's response. Default <c>true</c>.</summary>
    public bool CheckOutput { get; set; } = true;

    /// <summary>
    /// Gets or sets the assistant message returned when content is blocked and <see cref="ThrowOnViolation"/>
    /// is <c>false</c>. <c>null</c> or empty uses <see cref="DefaultRefusalMessage"/>.
    /// </summary>
    public string? RefusalMessage { get; set; }

    /// <summary>
    /// Gets or sets whether a violation throws a <see cref="ContentSafetyException"/> (<c>true</c>) instead of
    /// returning a refusal response (<c>false</c>, the default).
    /// </summary>
    public bool ThrowOnViolation { get; set; }
}
