using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.Configuration;

/// <summary>
/// A hashed API key stored in configuration for self-hosted and development deployments.
/// </summary>
/// <remarks>
/// The key itself is <c>aidn.&lt;Prefix&gt;.&lt;secret&gt;</c>. Configuration holds only the PBKDF2-SHA256 hash
/// of the base64url-decoded secret, never the secret.
/// </remarks>
public sealed class StaticApiKeyOptions
{
    /// <summary>
    /// Gets or sets the key id: the middle part of <c>aidn.&lt;keyId&gt;.&lt;secret&gt;</c>.
    /// </summary>
    public string Prefix { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the base64 HMAC-SHA256 of <see cref="Prefix"/> (UTF-8) under one of
    /// <see cref="ServingApiKeyOptions.HmacSecrets"/> (UTF-8). Required exactly when HMAC secrets are configured.
    /// </summary>
    public string HmacIdBase64 { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the base64 PBKDF2 salt.
    /// </summary>
    public string Pbkdf2SaltBase64 { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the base64 PBKDF2-SHA256 hash (32 bytes) of the key's decoded secret.
    /// </summary>
    public string Pbkdf2HashBase64 { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the PBKDF2 iteration count the hash was computed with.
    /// </summary>
    public int Pbkdf2Iterations { get; set; } = 600_000;

    /// <summary>
    /// Gets or sets the tier the key grants. <see cref="ServingTier.Premium"/> grants the Pro subscription tier.
    /// </summary>
    public ServingTier Tier { get; set; } = ServingTier.Free;

    /// <summary>
    /// Gets or sets when the key stops authenticating, or <see langword="null"/> for no expiry.
    /// </summary>
    public DateTimeOffset? ExpiresAt { get; set; }

    /// <summary>
    /// Gets or sets whether the key is revoked. A revoked key never authenticates.
    /// </summary>
    public bool Revoked { get; set; }

    /// <summary>
    /// Validates this entry and throws <see cref="InvalidOperationException"/> naming the first invalid setting.
    /// </summary>
    /// <param name="configurationPath">The configuration path used in error messages.</param>
    /// <param name="requireHmacId">Whether HMAC secrets are configured, which makes <see cref="HmacIdBase64"/>
    /// required; without them it must be empty, since it could not be verified.</param>
    public void Validate(string configurationPath, bool requireHmacId)
    {
        if (string.IsNullOrWhiteSpace(Prefix) || Prefix.IndexOf('.') >= 0)
        {
            throw new InvalidOperationException($"{configurationPath}:Prefix must be a non-empty key id without '.'.");
        }

        RequireBase64(Pbkdf2SaltBase64, $"{configurationPath}:Pbkdf2SaltBase64");
        RequireBase64(Pbkdf2HashBase64, $"{configurationPath}:Pbkdf2HashBase64");

        if (Pbkdf2Iterations < 1)
        {
            throw new InvalidOperationException($"{configurationPath}:Pbkdf2Iterations must be positive.");
        }

        if (!Enum.IsDefined(Tier))
        {
            throw new InvalidOperationException($"{configurationPath}:Tier '{Tier}' is not a defined tier.");
        }

        if (requireHmacId)
        {
            RequireBase64(HmacIdBase64, $"{configurationPath}:HmacIdBase64");
        }
        else if (!string.IsNullOrWhiteSpace(HmacIdBase64))
        {
            throw new InvalidOperationException(
                $"{configurationPath}:HmacIdBase64 is set but ServingSecurity:ApiKeys:HmacSecrets is empty, so it cannot be verified.");
        }
    }

    private static void RequireBase64(string value, string path)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException($"{path} is required.");
        }

        var buffer = new byte[value.Length];
        if (!Convert.TryFromBase64String(value, buffer, out int written) || written == 0)
        {
            throw new InvalidOperationException($"{path} must be non-empty base64.");
        }
    }
}
