using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.TEE;

/// <summary>
/// Verifies remote attestation quotes from TEE enclaves.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a client receives a claim "I'm running in a secure enclave,"
/// it needs to verify that claim. The verifier checks the hardware-signed attestation quote
/// against expected measurements, signer identity, and freshness policies.</para>
///
/// <para><b>Verification steps:</b></para>
/// <list type="number">
/// <item><description>Parse the raw quote (format varies by platform).</description></item>
/// <item><description>Verify the hardware signature chain (Intel/AMD/ARM root of trust).</description></item>
/// <item><description>Check that the measurement matches expected enclave code.</description></item>
/// <item><description>Check signer identity if required.</description></item>
/// <item><description>Verify quote freshness (not stale).</description></item>
/// <item><description>Verify platform firmware status.</description></item>
/// </list>
/// </remarks>
public interface IRemoteAttestationVerifier
{
    /// <summary>
    /// Verifies a remote attestation quote against the given policy and options.
    /// </summary>
    /// <param name="quote">Raw attestation quote bytes.</param>
    /// <param name="reportData">Application-specific report data that should be bound in the quote.</param>
    /// <param name="options">Attestation options (expected measurements, quote age, etc.).</param>
    /// <param name="policy">Attestation policy (strict, relaxed, custom).</param>
    /// <returns>Verification result containing pass/fail, measurement, and failure reason.</returns>
    RemoteAttestationResult Verify(
        byte[] quote,
        byte[] reportData,
        TeeAttestationOptions options,
        AttestationPolicy policy);

    /// <summary>
    /// Gets the supported TEE provider type for this verifier.
    /// </summary>
    TeeProviderType SupportedProvider { get; }
}
