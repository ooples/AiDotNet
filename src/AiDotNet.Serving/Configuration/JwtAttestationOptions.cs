namespace AiDotNet.Serving.Configuration;

/// <summary>
/// Configuration options for JWT-based attestation verification.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Many real-world attestation providers return a signed token (often a JWT).
/// This configuration tells AiDotNet.Serving how to validate that signature and which issuers/audiences are trusted.
/// </remarks>
public class JwtAttestationOptions
{
    /// <summary>
    /// Gets or sets the set of trusted issuers. If empty, issuer validation is disabled.
    /// </summary>
    public string[] ValidIssuers { get; set; } = [];

    /// <summary>
    /// Gets or sets the set of valid audiences. If empty, audience validation is disabled.
    /// </summary>
    public string[] ValidAudiences { get; set; } = [];

    /// <summary>
    /// Gets or sets trusted signing certificate bytes encoded as Base64 (DER).
    /// </summary>
    /// <remarks>
    /// Use this for environments where you want to ship public certs via configuration.
    /// </remarks>
    public string[] TrustedSigningCertificatesBase64 { get; set; } = [];

    /// <summary>
    /// Gets or sets trusted signing certificate file paths (DER/PEM supported by X509 loader).
    /// </summary>
    public string[] TrustedSigningCertificatePaths { get; set; } = [];

    /// <summary>
    /// Gets or sets the allowed clock skew (in seconds) when validating token lifetimes.
    /// </summary>
    public int ClockSkewSeconds { get; set; } = 60;

    /// <summary>
    /// Gets or sets whether a nonce claim must be present and match the provided evidence nonce.
    /// </summary>
    public bool RequireNonceClaim { get; set; } = true;

    /// <summary>
    /// Gets or sets whether a platform claim must be present and match the provided evidence platform.
    /// </summary>
    public bool RequirePlatformClaimMatch { get; set; } = true;

    /// <summary>
    /// Gets or sets whether a TEE type claim must be present and match the provided evidence tee type.
    /// </summary>
    public bool RequireTeeTypeClaimMatch { get; set; } = true;

    /// <summary>
    /// Gets or sets the claim type used to carry the nonce value.
    /// </summary>
    public string NonceClaimType { get; set; } = "nonce";

    /// <summary>
    /// Gets or sets the claim type used to carry the platform value.
    /// </summary>
    public string PlatformClaimType { get; set; } = "platform";

    /// <summary>
    /// Gets or sets the claim type used to carry the TEE type value.
    /// </summary>
    public string TeeTypeClaimType { get; set; } = "tee_type";
}

