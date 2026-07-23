namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Per-request generation limits enforced at the OpenAI HTTP boundary. All properties are nullable; when a
/// value is null the industry-standard default (below) applies, so the limits are always active even with
/// zero configuration. An operator can register a configured instance in DI to raise or lower any ceiling.
/// </summary>
/// <remarks>
/// These caps protect serving capacity: a single request that asks for an unbounded number of tokens (or a
/// large fan-out via <c>n</c>) can otherwise monopolize the engine. Violations are rejected with HTTP 400
/// rather than silently clamped, so clients learn their request exceeded a limit.
/// </remarks>
public sealed class ServingLimitsOptions
{
    /// <summary>Maximum <c>max_tokens</c> (new tokens) a single request may ask for. Null =&gt; 4096.</summary>
    public int? MaxCompletionTokens { get; set; }

    /// <summary>Maximum <c>n</c> (parallel completions) a single request may ask for. Null =&gt; 16.</summary>
    public int? MaxN { get; set; }

    /// <summary>Maximum prompt-tokens + <c>max_tokens</c> (total context) for a request. Null =&gt; 32768.</summary>
    public int? MaxContextTokens { get; set; }

    /// <summary>Effective completion-token ceiling (applies the default when unset).</summary>
    public int EffectiveMaxCompletionTokens => MaxCompletionTokens is > 0 ? MaxCompletionTokens.Value : 4096;

    /// <summary>Effective <c>n</c> ceiling (applies the default when unset).</summary>
    public int EffectiveMaxN => MaxN is > 0 ? MaxN.Value : 16;

    /// <summary>Effective total-context ceiling (applies the default when unset).</summary>
    public int EffectiveMaxContextTokens => MaxContextTokens is > 0 ? MaxContextTokens.Value : 32768;
}
