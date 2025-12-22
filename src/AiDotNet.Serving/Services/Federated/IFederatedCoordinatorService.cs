using AiDotNet.Serving.Models.Federated;
using AiDotNet.Serving.Services;

namespace AiDotNet.Serving.Services.Federated;

/// <summary>
/// Coordinator service for HTTP-based federated training runs.
/// </summary>
public interface IFederatedCoordinatorService
{
    CreateFederatedRunResponse CreateRun(CreateFederatedRunRequest request);
    Task<JoinFederatedRunResponse> JoinRunAsync(string runId, JoinFederatedRunRequest request, CancellationToken cancellationToken = default);
    FederatedRunParametersResponse GetParameters(string runId, int clientId);
    SubmitFederatedUpdateResponse SubmitUpdate(string runId, SubmitFederatedUpdateRequest request);
    AggregateFederatedRoundResponse AggregateRound(string runId);
    FederatedRunStatusResponse GetStatus(string runId);

    /// <summary>
    /// Gets the current plaintext run artifact path (updated after each aggregation).
    /// </summary>
    string GetRunArtifactPath(string runId);

    /// <summary>
    /// Gets or creates an encrypted run artifact for Option C deployments.
    /// </summary>
    ProtectedModelArtifact GetOrCreateEncryptedRunArtifact(string runId);
}
