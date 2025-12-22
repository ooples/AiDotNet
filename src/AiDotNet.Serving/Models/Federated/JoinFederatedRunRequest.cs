namespace AiDotNet.Serving.Models.Federated;

/// <summary>
/// Request for a client to join a federated training run.
/// </summary>
public class JoinFederatedRunRequest
{
    /// <summary>
    /// Gets or sets an optional client identifier.
    /// </summary>
    /// <remarks>
    /// If omitted, the server assigns a client id.
    /// </remarks>
    public int? ClientId { get; set; } = null;

    /// <summary>
    /// Gets or sets attestation evidence for admission control.
    /// </summary>
    public AttestationEvidence? Attestation { get; set; } = null;
}
