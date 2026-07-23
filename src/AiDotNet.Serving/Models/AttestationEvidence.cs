namespace AiDotNet.Serving.Models;

/// <summary>
/// Attestation evidence for a client runtime (device/VM TEE).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Attestation is how a server verifies that code is running inside a trusted hardware-backed environment
/// (for example, a secure enclave on a server VM, or a trusted runtime on a managed enterprise device).
/// </remarks>
public class AttestationEvidence
{
    /// <summary>
    /// Gets or sets the platform identifier (e.g., "Windows", "Android", "iOS", "Linux").
    /// </summary>
    public string Platform { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the TEE type (e.g., "SGX", "SEV-SNP", "TDX", "ARM-CCA", "MobileGateway").
    /// </summary>
    public string TeeType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets a server-provided nonce to prevent replay attacks.
    /// </summary>
    public string Nonce { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets a provider-specific attestation token/quote.
    /// </summary>
    public string AttestationToken { get; set; } = string.Empty;
}

