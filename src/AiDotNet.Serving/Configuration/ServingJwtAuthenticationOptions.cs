namespace AiDotNet.Serving.Configuration;

/// <summary>
/// JWT bearer authentication configuration for AiDotNet.Serving.
/// </summary>
/// <remarks>
/// When enabled, a request whose <c>Authorization: Bearer</c> value is not an AiDotNet API key is authenticated as
/// a JWT. Tokens are validated either against an OpenID Connect <see cref="Authority"/> (signing keys and issuer
/// come from its metadata) or against a symmetric <see cref="SigningKey"/> with an explicit <see cref="Issuer"/>.
/// The <see cref="Audience"/> is always validated.
/// </remarks>
public sealed class ServingJwtAuthenticationOptions
{
    /// <summary>
    /// The minimum symmetric signing key length, in bytes, for HMAC-SHA256 tokens.
    /// </summary>
    public const int MinimumSigningKeyBytes = 32;

    /// <summary>
    /// Gets or sets whether JWT bearer authentication is enabled.
    /// </summary>
    public bool Enabled { get; set; }

    /// <summary>
    /// Gets or sets the OpenID Connect authority (recommended for production identity providers).
    /// </summary>
    public string? Authority { get; set; }

    /// <summary>
    /// Gets or sets the audience a token must be issued for. Required when JWT authentication is enabled.
    /// </summary>
    public string? Audience { get; set; }

    /// <summary>
    /// Gets or sets the issuer a token must come from. Required with <see cref="SigningKey"/>; with
    /// <see cref="Authority"/>, it overrides the issuer from the authority's metadata when set.
    /// </summary>
    public string? Issuer { get; set; }

    /// <summary>
    /// Gets or sets a symmetric signing key for self-hosted and development deployments: base64 when it decodes
    /// as base64, UTF-8 text otherwise; at least <see cref="MinimumSigningKeyBytes"/> bytes either way. Prefer
    /// <see cref="Authority"/> in production.
    /// </summary>
    public string? SigningKey { get; set; }

    /// <summary>
    /// Gets or sets whether the authority's metadata must be fetched over HTTPS.
    /// </summary>
    public bool RequireHttpsMetadata { get; set; } = true;

    /// <summary>
    /// Returns the bytes of <see cref="SigningKey"/>, or <see langword="null"/> when no signing key is set.
    /// </summary>
    public byte[]? GetSigningKeyBytes()
    {
        if (string.IsNullOrWhiteSpace(SigningKey))
        {
            return null;
        }

        string key = SigningKey.Trim();
        var buffer = new byte[key.Length];
        return Convert.TryFromBase64String(key, buffer, out int written) && written > 0
            ? buffer.AsSpan(0, written).ToArray()
            : System.Text.Encoding.UTF8.GetBytes(key);
    }

    /// <summary>
    /// Validates these settings and throws <see cref="InvalidOperationException"/> naming the first invalid one.
    /// </summary>
    /// <param name="configurationPath">The configuration path used in error messages.</param>
    public void Validate(string configurationPath)
    {
        if (!Enabled)
        {
            return;
        }

        bool hasAuthority = !string.IsNullOrWhiteSpace(Authority);
        byte[]? signingKey = GetSigningKeyBytes();
        if (!hasAuthority && signingKey is null)
        {
            throw new InvalidOperationException(
                $"{configurationPath}: Enabled is true, so Authority or SigningKey must be set.");
        }

        if (hasAuthority && !Uri.TryCreate(Authority, UriKind.Absolute, out _))
        {
            throw new InvalidOperationException($"{configurationPath}:Authority must be an absolute URI.");
        }

        if (string.IsNullOrWhiteSpace(Audience))
        {
            throw new InvalidOperationException($"{configurationPath}:Audience is required when JWT authentication is enabled.");
        }

        if (signingKey is not null)
        {
            if (signingKey.Length < MinimumSigningKeyBytes)
            {
                throw new InvalidOperationException(
                    $"{configurationPath}:SigningKey must be at least {MinimumSigningKeyBytes} bytes.");
            }

            if (!hasAuthority && string.IsNullOrWhiteSpace(Issuer))
            {
                throw new InvalidOperationException(
                    $"{configurationPath}:Issuer is required when tokens are validated with SigningKey.");
            }
        }
    }
}
