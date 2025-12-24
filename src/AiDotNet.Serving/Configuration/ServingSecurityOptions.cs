namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Security configuration for AiDotNet.Serving (tier identification, API keys, and admin authorization).
/// </summary>
public sealed class ServingSecurityOptions
{
    public ServingApiKeyOptions ApiKeys { get; set; } = new();

    public ServingJwtAuthenticationOptions JwtAuthentication { get; set; } = new();

    public ServingJwtTierOptions Jwt { get; set; } = new();

    public AdminAuthorizationOptions Admin { get; set; } = new();
}
