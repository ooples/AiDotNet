namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Security configuration for AiDotNet.Serving, bound from the <c>ServingSecurity</c> configuration section.
/// </summary>
/// <remarks>
/// The server binds this section at startup and validates it before it accepts requests: an invalid value stops
/// the host with an error that names the setting.
/// </remarks>
public sealed class ServingSecurityOptions
{
    /// <summary>
    /// The configuration section this class is bound from.
    /// </summary>
    public const string SectionName = "ServingSecurity";

    /// <summary>
    /// Gets or sets the API key settings: the request header, hashing cost and config-backed (static) keys.
    /// </summary>
    public ServingApiKeyOptions ApiKeys { get; set; } = new();

    /// <summary>
    /// Gets or sets the JWT bearer authentication settings. JWT callers are accepted alongside API key callers.
    /// </summary>
    public ServingJwtAuthenticationOptions JwtAuthentication { get; set; } = new();

    /// <summary>
    /// Gets or sets how a JWT caller's subscription tier is read from its claims.
    /// </summary>
    public ServingJwtTierOptions Jwt { get; set; } = new();

    /// <summary>
    /// Gets or sets the claim that grants a JWT caller access to the admin endpoints.
    /// </summary>
    public AdminAuthorizationOptions Admin { get; set; } = new();

    /// <summary>
    /// Validates every subsection and throws <see cref="InvalidOperationException"/> naming the first invalid
    /// setting.
    /// </summary>
    public void Validate()
    {
        ApiKeys.Validate($"{SectionName}:{nameof(ApiKeys)}");
        JwtAuthentication.Validate($"{SectionName}:{nameof(JwtAuthentication)}");
        Jwt.Validate($"{SectionName}:{nameof(Jwt)}");
        Admin.Validate($"{SectionName}:{nameof(Admin)}");
    }
}
