using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Result of authenticating an API key.
/// </summary>
public sealed class ApiKeyAuthenticationResult
{
    public ApiKeyAuthenticationResult(string keyId, SubscriptionTier tier, ApiKeyScopes scopes)
    {
        if (string.IsNullOrWhiteSpace(keyId))
        {
            throw new ArgumentException("KeyId cannot be null or empty.", nameof(keyId));
        }

        KeyId = keyId;
        Tier = tier;
        Scopes = scopes;
    }

    public string KeyId { get; }
    public SubscriptionTier Tier { get; }
    public ApiKeyScopes Scopes { get; }
}
