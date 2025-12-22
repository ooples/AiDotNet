namespace AiDotNet.Serving.Security.ApiKeys;

/// <summary>
/// Claim types used for AiDotNet API key authentication.
/// </summary>
public static class ApiKeyClaimTypes
{
    public const string KeyId = "aidn_key_id";
    public const string Tier = "aidn_tier";
    public const string Scope = "aidn_scope";
}

