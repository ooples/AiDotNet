using Microsoft.AspNetCore.Authentication;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Options for API key authentication.
/// </summary>
public sealed class ApiKeyAuthenticationOptions : AuthenticationSchemeOptions
{
    /// <summary>
    /// Gets or sets the HTTP header used to send the API key.
    /// </summary>
    public string HeaderName { get; set; } = "X-AiDotNet-ApiKey";

    /// <summary>
    /// Gets or sets whether query-string authentication is allowed.
    /// </summary>
    /// <remarks>
    /// Query-string API keys are discouraged; keep this disabled unless you have a controlled internal scenario.
    /// </remarks>
    public bool AllowQueryString { get; set; } = false;

    /// <summary>
    /// Gets or sets the query-string parameter name used for the API key when enabled.
    /// </summary>
    public string QueryStringParameterName { get; set; } = "api_key";
}

