namespace AiDotNet.Serving.Security.Attestation;

/// <summary>
/// Result of attestation verification.
/// </summary>
public sealed class AttestationVerificationResult
{
    public AttestationVerificationResult(bool isSuccess, string? failureReason = null)
    {
        IsSuccess = isSuccess;
        FailureReason = failureReason;
    }

    /// <summary>
    /// Gets whether attestation verification succeeded.
    /// </summary>
    public bool IsSuccess { get; }

    /// <summary>
    /// Gets the failure reason if attestation verification failed.
    /// </summary>
    public string? FailureReason { get; }
}

