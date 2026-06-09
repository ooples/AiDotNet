namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Settings for <see cref="LocalEngineChatClient{T}"/>: the reported model id, the default generation length,
/// and the default sampling behavior (overridable per request via <see cref="ChatOptions"/>).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are the local model's defaults. The most useful is
/// <see cref="MaxOutputTokens"/> (how long a reply may get before the engine stops). Leave
/// <see cref="Sampling"/> unset for safe, near-greedy behavior, or set it to make replies more creative.
/// </para>
/// </remarks>
public sealed class LocalEngineOptions
{
    /// <summary>The default maximum number of tokens generated per reply when none is specified.</summary>
    public const int DefaultMaxOutputTokens = 256;

    /// <summary>
    /// Gets or sets the model id reported by <see cref="LocalEngineChatClient{T}.ModelId"/>. <c>null</c> or
    /// empty falls back to <c>"local"</c>.
    /// </summary>
    public string? ModelId { get; set; }

    /// <summary>
    /// Gets or sets the default maximum number of tokens to generate per reply. <c>null</c> or a non-positive
    /// value uses <see cref="DefaultMaxOutputTokens"/>. A request's <see cref="ChatOptions.MaxOutputTokens"/>
    /// overrides this.
    /// </summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>
    /// Gets or sets the default sampling settings. <c>null</c> uses near-greedy defaults. Per-request
    /// <see cref="ChatOptions"/> values (temperature, top-k, top-p, seed) override the matching fields.
    /// </summary>
    public LocalSamplingOptions? Sampling { get; set; }
}
