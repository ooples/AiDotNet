using System.Net.Http;

namespace AiDotNet.ProgramSynthesis.Serving;

/// <summary>
/// Configuration for calling an AiDotNet.Serving instance for Program Synthesis operations.
/// </summary>
public sealed class ProgramSynthesisServingClientOptions
{
    /// <summary>
    /// Base address of the AiDotNet.Serving instance (e.g., http://localhost:52432/).
    /// </summary>
    public Uri? BaseAddress { get; set; }

    /// <summary>
    /// Optional API key (sent using <see cref="ApiKeyHeaderName"/>).
    /// </summary>
    public string? ApiKey { get; set; }

    /// <summary>
    /// Optional bearer token (sent using the Authorization: Bearer header).
    /// </summary>
    public string? BearerToken { get; set; }

    /// <summary>
    /// Header name used for API key authentication.
    /// </summary>
    public string ApiKeyHeaderName { get; set; } = "X-AiDotNet-Api-Key";

    /// <summary>
    /// Optional HttpClient to use for requests (recommended for re-use).
    /// If null, a new HttpClient is created.
    /// </summary>
    public HttpClient? HttpClient { get; set; }

    /// <summary>
    /// Request timeout in milliseconds.
    /// </summary>
    public int TimeoutMs { get; set; } = 100_000;

    /// <summary>
    /// When true, higher-level APIs prefer Serving when configured.
    /// </summary>
    public bool PreferServing { get; set; } = true;
}

