using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Persists and loads versioned opaque evolution checkpoints.</summary>
public interface IEvolutionCheckpointStore
{
    /// <summary>Atomically saves the latest checkpoint for its run.</summary>
    Task SaveAsync(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken = default);

    /// <summary>Loads the latest valid checkpoint for a run, or <c>null</c> when none exists.</summary>
    Task<EvolutionCheckpoint?> LoadLatestAsync(string runId, CancellationToken cancellationToken = default);
}
