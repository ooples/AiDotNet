using System.Collections.Concurrent;

namespace AiDotNet.Serving.Services.Federated;

/// <summary>
/// In-memory federated run store for development and single-instance deployments.
/// </summary>
public sealed class InMemoryFederatedRunStore : IFederatedRunStore
{
    private readonly ConcurrentDictionary<string, FederatedRunState> _runs = new(StringComparer.OrdinalIgnoreCase);

    public FederatedRunState Create(FederatedRunState state)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        if (string.IsNullOrWhiteSpace(state.RunId))
        {
            throw new ArgumentException("RunId is required.", nameof(state));
        }

        if (!_runs.TryAdd(state.RunId, state))
        {
            throw new InvalidOperationException($"Federated run '{state.RunId}' already exists.");
        }

        return state;
    }

    public bool TryGet(string runId, out FederatedRunState? state)
    {
        return _runs.TryGetValue(runId, out state);
    }

    public void Update(FederatedRunState state)
    {
        if (state == null)
        {
            throw new ArgumentNullException(nameof(state));
        }

        _runs[state.RunId] = state;
    }

    public bool TryRemove(string runId)
    {
        return _runs.TryRemove(runId, out _);
    }
}

