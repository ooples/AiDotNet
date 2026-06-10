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

    /// <summary>
    /// Gets or sets a token constraint applied during generation (constrained decoding). <c>null</c> means
    /// unconstrained generation. Use this to guarantee structured output (a grammar, JSON shape, or a closed
    /// vocabulary) at the logits rather than relying on prompting.
    /// </summary>
    public ITokenConstraint? Constraint { get; set; }

    /// <summary>
    /// Gets or sets the beam width for beam-search decoding. <c>null</c> or a value &lt;= 1 uses ordinary
    /// token-by-token sampling/greedy decoding. A value &gt; 1 explores that many hypotheses in parallel and
    /// returns the highest-scoring (length-normalized) completion — deterministic, and typically higher
    /// quality than greedy for short structured outputs. Beam search applies to non-streaming
    /// <see cref="LocalEngineChatClient{T}.GetResponseAsync"/>; streaming always decodes token-by-token.
    /// </summary>
    public int? BeamWidth { get; set; }
}
