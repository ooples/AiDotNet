namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// A process-local <see cref="ITrajectoryStore"/> that keeps captured runs in memory. Ideal for tests and
/// single-process self-improvement loops; contents are lost when the process exits.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The simplest logbook, kept in RAM — fast and zero-config, but not saved to
/// disk. Swap in a durable store when you need trajectories to persist across restarts.
/// </para>
/// </remarks>
public sealed class InMemoryTrajectoryStore : ITrajectoryStore
{
    private readonly object _gate = new();
    private readonly List<AgentTrajectory> _trajectories = new();

    /// <inheritdoc/>
    /// <exception cref="ArgumentException">Thrown when a trajectory with the same id is already stored.</exception>
    public Task<string> AddAsync(AgentTrajectory trajectory, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(trajectory);
        lock (_gate)
        {
            // GetAsync(id) implies a unique key; silently accepting a duplicate
            // id would make lookups and reward annotation nondeterministic.
            if (_trajectories.Any(t => string.Equals(t.Id, trajectory.Id, StringComparison.Ordinal)))
            {
                throw new ArgumentException(
                    $"A trajectory with id '{trajectory.Id}' already exists.", nameof(trajectory));
            }

            _trajectories.Add(trajectory);
        }

        return Task.FromResult(trajectory.Id);
    }

    /// <inheritdoc/>
    public Task<AgentTrajectory?> GetAsync(string id, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(id);
        lock (_gate)
        {
            AgentTrajectory? match = _trajectories.FirstOrDefault(t => string.Equals(t.Id, id, StringComparison.Ordinal));
            return Task.FromResult<AgentTrajectory?>(match);
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<AgentTrajectory>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            IReadOnlyList<AgentTrajectory> all = new List<AgentTrajectory>(_trajectories);
            return Task.FromResult(all);
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<AgentTrajectory>> QueryAsync(
        Func<AgentTrajectory, bool> predicate,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(predicate);
        // Snapshot under the lock, filter outside it: the predicate is
        // caller-supplied code and must not be able to block every
        // add/get/clear on the store (or deadlock by re-entering it).
        List<AgentTrajectory> snapshot;
        lock (_gate)
        {
            snapshot = new List<AgentTrajectory>(_trajectories);
        }

        IReadOnlyList<AgentTrajectory> matches = snapshot.Where(predicate).ToList();
        return Task.FromResult(matches);
    }

    /// <inheritdoc/>
    public Task ClearAsync(CancellationToken cancellationToken = default)
    {
        lock (_gate)
        {
            _trajectories.Clear();
        }

        return Task.CompletedTask;
    }
}
