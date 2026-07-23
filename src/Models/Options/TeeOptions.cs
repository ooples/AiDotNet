namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Trusted Execution Environment integration in federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how the FL server uses hardware security
/// features. The defaults use simulated mode for testing; switch to a real provider
/// (Tdx, SevSnp, etc.) for production deployment.</para>
/// </remarks>
public class TeeOptions
{
    /// <summary>
    /// Gets or sets the TEE provider type. Default is <see cref="TeeProviderType.Simulated"/>.
    /// </summary>
    public TeeProviderType Provider { get; set; } = TeeProviderType.Simulated;

    /// <summary>
    /// Gets or sets the attestation policy. Default is <see cref="AttestationPolicy.Strict"/>.
    /// </summary>
    public AttestationPolicy Policy { get; set; } = AttestationPolicy.Strict;

    /// <summary>
    /// Gets or sets the attestation configuration.
    /// </summary>
    public TeeAttestationOptions Attestation { get; set; } = new TeeAttestationOptions();

    /// <summary>
    /// Gets or sets the maximum enclave memory in megabytes. Default is 256 (SGX limit).
    /// TDX and SEV-SNP support much larger values.
    /// </summary>
    public int MaxEnclaveMemoryMb { get; set; } = 256;

    /// <summary>
    /// Gets or sets whether to enable simulation mode for testing. Default is true.
    /// Must be false for production deployments with real hardware.
    /// </summary>
    public bool SimulationMode { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to require remote attestation from clients. Default is true.
    /// </summary>
    public bool RequireAttestation { get; set; } = true;

    /// <summary>
    /// Gets or sets the enclave code identity hash (MRENCLAVE for SGX, measurement for others).
    /// Used during attestation to verify the correct code is running.
    /// </summary>
    public string ExpectedMeasurement { get; set; } = string.Empty;
}
