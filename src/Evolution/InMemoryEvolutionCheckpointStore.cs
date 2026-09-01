using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>A thread-safe in-memory checkpoint store intended for tests and short-lived applications.</summary>
public sealed class InMemoryEvolutionCheckpointStore : IEvolutionCheckpointStore
{
    private readonly object _gate = new();
    private readonly Dictionary<string, EvolutionCheckpoint> _checkpoints = new(StringComparer.Ordinal);

    /// <inheritdoc/>
    public Task SaveAsync(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(checkpoint);
        cancellationToken.ThrowIfCancellationRequested();
        checkpoint.Validate();
        lock (_gate)
        {
            if (_checkpoints.TryGetValue(checkpoint.RunId, out EvolutionCheckpoint? existing))
            {
                ValidateSuccessor(existing, checkpoint);
                if (checkpoint.Sequence == existing.Sequence) return Task.CompletedTask;
            }
            _checkpoints[checkpoint.RunId] = checkpoint.Clone();
        }
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public Task<EvolutionCheckpoint?> LoadLatestAsync(string runId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(runId);
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            EvolutionCheckpoint? result = _checkpoints.TryGetValue(runId, out EvolutionCheckpoint? checkpoint)
                ? checkpoint.Clone()
                : null;
            return Task.FromResult(result);
        }
    }

    private static void ValidateSuccessor(EvolutionCheckpoint existing, EvolutionCheckpoint checkpoint)
    {
        if (!string.Equals(existing.CompatibilityHash, checkpoint.CompatibilityHash, StringComparison.Ordinal))
            throw new InvalidOperationException("A checkpoint run cannot change compatibility identity.");
        if (checkpoint.Sequence < existing.Sequence)
            throw new InvalidOperationException("A checkpoint store cannot move a run backwards.");
        if (checkpoint.Sequence == existing.Sequence &&
            !string.Equals(existing.Checksum, checkpoint.Checksum, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("A checkpoint sequence cannot identify two different states.");
        }
    }
}
