namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Represents the result of verifying a remote attestation quote from a TEE.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a server says "I'm running inside a secure enclave," the client
/// needs proof. Remote attestation produces a hardware-signed "quote" that a verifier checks against
/// expected measurements. This class holds the outcome of that verification.</para>
///
/// <para><b>Key fields:</b></para>
/// <list type="bullet">
/// <item><description><b>IsValid:</b> Did the quote pass all checks?</description></item>
/// <item><description><b>MeasurementHash:</b> The enclave's code identity (MRENCLAVE for SGX).</description></item>
/// <item><description><b>SignerIdentity:</b> The enclave author identity (MRSIGNER for SGX).</description></item>
/// <item><description><b>QuoteTimestamp:</b> When the quote was generated — stale quotes may be rejected.</description></item>
/// </list>
/// </remarks>
public class RemoteAttestationResult
{
    /// <summary>
    /// Gets or sets whether the attestation verification passed all policy checks.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets the enclave measurement hash (MRENCLAVE for SGX, launch digest for TDX/SEV-SNP).
    /// Hex-encoded.
    /// </summary>
    public string MeasurementHash { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the enclave signer identity (MRSIGNER for SGX). Hex-encoded.
    /// </summary>
    public string SignerIdentity { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the attestation policy that was applied.
    /// </summary>
    public AiDotNet.Models.Options.AttestationPolicy PolicyApplied { get; set; }

    /// <summary>
    /// Gets or sets the UTC timestamp when the attestation quote was generated.
    /// </summary>
    public DateTime QuoteTimestamp { get; set; }

    /// <summary>
    /// Gets or sets a human-readable reason if verification failed.
    /// Empty when <see cref="IsValid"/> is true.
    /// </summary>
    public string FailureReason { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the raw attestation quote bytes (platform-specific binary format).
    /// </summary>
    public byte[] RawQuote { get; set; } = Array.Empty<byte>();

    /// <summary>
    /// Gets or sets the TEE provider type that produced this attestation.
    /// </summary>
    public AiDotNet.Models.Options.TeeProviderType ProviderType { get; set; }

    /// <summary>
    /// Gets or sets whether the platform firmware was verified as up-to-date.
    /// </summary>
    public bool FirmwareVerified { get; set; }

    /// <summary>
    /// Gets or sets the platform firmware version string (e.g., TCB level for SGX/TDX).
    /// </summary>
    public string FirmwareVersion { get; set; } = string.Empty;
}
