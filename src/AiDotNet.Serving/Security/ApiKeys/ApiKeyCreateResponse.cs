using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Response for API key creation (key is returned once).
/// </summary>
public sealed class ApiKeyCreateResponse
{
    public string KeyId { get; set; } = string.Empty;

    public string ApiKey { get; set; } = string.Empty;

    public SubscriptionTier Tier { get; set; } = SubscriptionTier.Free;

    public ApiKeyScopes Scopes { get; set; } = ApiKeyScopes.None;

    public DateTimeOffset CreatedAt { get; set; }

    public DateTimeOffset? ExpiresAt { get; set; }
}

