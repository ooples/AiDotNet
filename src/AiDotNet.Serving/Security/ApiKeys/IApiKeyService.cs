namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// API key management and authentication service.
/// </summary>
public interface IApiKeyService
{
    Task<ApiKeyAuthenticationResult?> AuthenticateAsync(string apiKey, CancellationToken cancellationToken = default);

    Task<ApiKeyCreateResponse> CreateAsync(ApiKeyCreateRequest request, CancellationToken cancellationToken = default);

    Task<bool> RevokeAsync(string keyId, CancellationToken cancellationToken = default);
}

