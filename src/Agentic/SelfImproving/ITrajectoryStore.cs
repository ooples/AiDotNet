namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Stores captured <see cref="AgentTrajectory"/> records so the self-improving layer can replay, evaluate,
/// and learn from past agent runs.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is the logbook of everything the agents have done. Other parts of the
/// system read from it to grade past runs and figure out how to do better next time.
/// </para>
/// </remarks>
public interface ITrajectoryStore
{
    /// <summary>
    /// Adds a trajectory and returns its id.
    /// </summary>
    /// <param name="trajectory">The trajectory to store.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    /// <returns>The stored trajectory's id.</returns>
    Task<string> AddAsync(AgentTrajectory trajectory, CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets a trajectory by id, or <c>null</c> when not found.
    /// </summary>
    /// <param name="id">The trajectory id.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task<AgentTrajectory?> GetAsync(string id, CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns all stored trajectories, oldest first.
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task<IReadOnlyList<AgentTrajectory>> GetAllAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Returns the trajectories matching a predicate, oldest first.
    /// </summary>
    /// <param name="predicate">The filter to apply.</param>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task<IReadOnlyList<AgentTrajectory>> QueryAsync(
        Func<AgentTrajectory, bool> predicate,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Removes all stored trajectories.
    /// </summary>
    /// <param name="cancellationToken">Token used to cancel the operation.</param>
    Task ClearAsync(CancellationToken cancellationToken = default);
}
