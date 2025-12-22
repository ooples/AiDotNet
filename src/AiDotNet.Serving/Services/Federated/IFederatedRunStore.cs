namespace AiDotNet.Serving.Services.Federated;

/// <summary>
/// Stores federated run state for the serving coordinator.
/// </summary>
public interface IFederatedRunStore
{
    FederatedRunState Create(FederatedRunState state);
    bool TryGet(string runId, out FederatedRunState? state);
    void Update(FederatedRunState state);
    bool TryRemove(string runId);
}

