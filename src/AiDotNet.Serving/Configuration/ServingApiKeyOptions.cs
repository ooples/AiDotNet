namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration for API key based tier access in AiDotNet.Serving.
/// </summary>
/// <remarks>
/// <para>
/// An API key has the form <c>aidn.&lt;keyId&gt;.&lt;secret&gt;</c>: <c>keyId</c> locates the key and
/// <c>secret</c> (base64url) is verified against its PBKDF2-SHA256 hash. Keys created through the admin API are
/// stored in the database; <see cref="StaticApiKeys"/> holds config-backed keys for self-hosted deployments.
/// </para>
/// <para>
/// When <see cref="HmacSecrets"/> is set, a static key must also carry the HMAC-SHA256 of its key id under one of
/// those secrets (<see cref="StaticApiKeyOptions.HmacIdBase64"/>), so a copy of the configuration without the
/// secret store cannot be used to add a key.
/// </para>
/// </remarks>
public sealed class ServingApiKeyOptions
{
    /// <summary>
    /// The header used when no header is configured: the one existing clients send.
    /// </summary>
    public const string DefaultHeaderName = "X-AiDotNet-ApiKey";

    /// <summary>
    /// Gets or sets the HTTP header that carries the API key. Keys are also accepted as an
    /// <c>Authorization: Bearer</c> value.
    /// </summary>
    /// <remarks>
    /// The legacy <c>ApiKeyAuthenticationOptions:HeaderName</c> setting still overrides this value when it is set,
    /// and the server logs a warning asking to move it here.
    /// </remarks>
    public string HeaderName { get; set; } = DefaultHeaderName;

    /// <summary>
    /// Gets or sets the PBKDF2 iteration count used when hashing newly created API keys.
    /// </summary>
    /// <remarks>
    /// Existing keys keep the count they were created with, so raising this value does not invalidate them. The
    /// default follows OWASP's PBKDF2-HMAC-SHA256 recommendation.
    /// </remarks>
    public int Pbkdf2Iterations { get; set; } = 600_000;

    /// <summary>
    /// Gets or sets the HMAC secrets used to verify the lookup identifiers of static API keys.
    /// </summary>
    /// <remarks>
    /// To rotate, put the new secret first and keep the older ones until every static key's
    /// <see cref="StaticApiKeyOptions.HmacIdBase64"/> has been recomputed. Store secrets in a secure provider
    /// (environment variables, secret manager, key vault), never in source control.
    /// </remarks>
    public List<string> HmacSecrets { get; set; } = new();

    /// <summary>
    /// Gets or sets config-backed API keys for self-hosted and development deployments. Only hashes are stored.
    /// </summary>
    public List<StaticApiKeyOptions> StaticApiKeys { get; set; } = new();

    /// <summary>
    /// Validates these settings and throws <see cref="InvalidOperationException"/> naming the first invalid one.
    /// </summary>
    /// <param name="configurationPath">The configuration path used in error messages.</param>
    public void Validate(string configurationPath)
    {
        if (string.IsNullOrWhiteSpace(HeaderName))
        {
            throw new InvalidOperationException($"{configurationPath}:HeaderName must be a non-empty string.");
        }

        if (Pbkdf2Iterations < 1)
        {
            throw new InvalidOperationException($"{configurationPath}:Pbkdf2Iterations must be positive.");
        }

        for (int i = 0; i < HmacSecrets.Count; i++)
        {
            if (string.IsNullOrWhiteSpace(HmacSecrets[i]))
            {
                throw new InvalidOperationException($"{configurationPath}:HmacSecrets:{i} must be a non-empty string.");
            }
        }

        var keyIds = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < StaticApiKeys.Count; i++)
        {
            string path = $"{configurationPath}:StaticApiKeys:{i}";
            StaticApiKeys[i].Validate(path, requireHmacId: HmacSecrets.Count > 0);
            if (!keyIds.Add(StaticApiKeys[i].Prefix))
            {
                throw new InvalidOperationException($"{path}:Prefix '{StaticApiKeys[i].Prefix}' is used by another static key.");
            }
        }
    }
}
