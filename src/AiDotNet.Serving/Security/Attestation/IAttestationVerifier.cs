using AiDotNet.Serving.Models;

namespace AiDotNet.Serving.Security.Attestation;

/// <summary>
/// Verifies client attestation evidence.
/// </summary>
public interface IAttestationVerifier
{
    /// <summary>
    /// Verifies the provided attestation evidence.
    /// </summary>
    Task<AttestationVerificationResult> VerifyAsync(AttestationEvidence evidence, CancellationToken cancellationToken = default);
}

