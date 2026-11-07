using AiDotNet.Serving.Security.ApiKeys;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http;

namespace AiDotNet.Serving.Security;

/// <summary>
/// The authentication schemes of AiDotNet.Serving and how a request is routed between them.
/// </summary>
/// <remarks>
/// The default scheme is a policy scheme: it forwards each request to the API key scheme or, when JWT
/// authentication is enabled, to the JWT bearer scheme, so both kinds of caller reach the same endpoints.
/// </remarks>
public static class ServingAuthenticationSchemes
{
    /// <summary>
    /// The default (policy) scheme that forwards to <see cref="ApiKeyAuthenticationDefaults.Scheme"/> or
    /// <see cref="JwtBearerDefaults.AuthenticationScheme"/>.
    /// </summary>
    public const string Default = "AiDotNetServing";

    private const string BearerPrefix = "Bearer ";

    /// <summary>
    /// Chooses the scheme that authenticates <paramref name="request"/>.
    /// </summary>
    /// <param name="request">The incoming request.</param>
    /// <param name="apiKeyHeaderName">The header that carries API keys.</param>
    /// <param name="jwtEnabled">Whether JWT bearer authentication is registered.</param>
    /// <returns>The JWT bearer scheme for a bearer token that is not an AiDotNet API key, when JWT is enabled and
    /// no API key header is present; the API key scheme otherwise.</returns>
    public static string Select(HttpRequest request, string apiKeyHeaderName, bool jwtEnabled)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (!jwtEnabled || request.Headers.ContainsKey(apiKeyHeaderName))
        {
            return ApiKeyAuthenticationDefaults.Scheme;
        }

        string authorization = request.Headers.Authorization.ToString();
        if (authorization.StartsWith(BearerPrefix, StringComparison.OrdinalIgnoreCase) &&
            !ApiKeyFormat.HasApiKeyPrefix(authorization.Substring(BearerPrefix.Length).TrimStart()))
        {
            return JwtBearerDefaults.AuthenticationScheme;
        }

        return ApiKeyAuthenticationDefaults.Scheme;
    }
}
