namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TEE remote attestation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Remote attestation lets you verify that a remote computer is truly
/// running inside a secure enclave. These options control how that verification works —
/// what evidence to expect and how long it's valid.</para>
/// </remarks>
public class TeeAttestationOptions
{
    /// <summary>
    /// Gets or sets the attestation quote format. Default is "ECDSA-256" (Intel DCAP).
    /// </summary>
    public string QuoteFormat { get; set; } = "ECDSA-256";

    /// <summary>
    /// Gets or sets the expected enclave measurement hashes.
    /// Each entry is a hex-encoded hash of an approved enclave binary.
    /// </summary>
    public IList<string> ExpectedMeasurements { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the maximum age of an attestation quote in seconds before it's considered stale.
    /// Default is 3600 (1 hour).
    /// </summary>
    public int MaxQuoteAgeSec { get; set; } = 3600;

    /// <summary>
    /// Gets or sets whether to check the enclave signer identity (MRSIGNER for SGX).
    /// Default is true.
    /// </summary>
    public bool VerifySignerIdentity { get; set; } = true;

    /// <summary>
    /// Gets or sets the expected signer identity hash (hex-encoded).
    /// </summary>
    public string ExpectedSignerIdentity { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether to verify the platform firmware is up to date.
    /// Default is true.
    /// </summary>
    public bool VerifyPlatformFirmware { get; set; } = true;
}
