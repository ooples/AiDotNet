using AiDotNet.Evolution;

namespace AiDotNet.Interfaces;

/// <summary>Persists and loads versioned opaque evolution checkpoints.</summary>
/// <remarks>
/// <para>
/// The engine treats a store as durable memory for one run identifier: <see cref="SaveAsync"/> receives
/// checkpoints whose <see cref="EvolutionCheckpoint.Sequence"/> never decreases and whose
/// <see cref="EvolutionCheckpoint.CompatibilityHash"/> never changes within a run, and
/// <see cref="LoadLatestAsync"/> is called at start-up when resume is enabled. Implementations should
/// validate incoming checkpoints with <see cref="EvolutionCheckpoint.Validate"/>, write atomically so a
/// crash mid-save can never leave a truncated or half-written latest checkpoint, refuse to move a run
/// backwards in sequence, and refuse to change a run's compatibility identity. The payload is engine-owned
/// and opaque: a store must persist it exactly and never interpret or rewrite it.
/// </para>
/// <para><b>For Beginners:</b> A long evolutionary search can run for hours, so the engine periodically
/// writes a snapshot of everything it needs to continue, such as the archives, the random-number state, and
/// its progress counters. This interface is the place those snapshots are saved to and read back from, much
/// like the save slot of a video game. If the process is restarted with <c>EvolutionEngineOptions.Resume</c>
/// enabled, the engine loads the newest snapshot and carries on where it left off instead of starting from
/// scratch. Use <see cref="InMemoryEvolutionCheckpointStore"/> for tests and short-lived jobs and
/// <see cref="JsonEvolutionCheckpointStore"/> to keep checkpoints on disk; implement the interface yourself
/// to target a database or cloud storage.</para>
/// </remarks>
public interface IEvolutionCheckpointStore
{
    /// <summary>Atomically saves the latest checkpoint for its run.</summary>
    /// <param name="checkpoint">The checkpoint to persist; its run identifier selects the slot that is replaced.</param>
    /// <param name="cancellationToken">A token that cancels the save before it is committed.</param>
    /// <returns>A task that completes once the checkpoint is durably stored.</returns>
    Task SaveAsync(EvolutionCheckpoint checkpoint, CancellationToken cancellationToken = default);

    /// <summary>Loads the latest valid checkpoint for a run, or <c>null</c> when none exists.</summary>
    /// <param name="runId">The stable run identifier whose newest checkpoint is requested.</param>
    /// <param name="cancellationToken">A token that cancels the load.</param>
    /// <returns>The newest checkpoint for the run, or <c>null</c> when the run has never been saved.</returns>
    Task<EvolutionCheckpoint?> LoadLatestAsync(string runId, CancellationToken cancellationToken = default);
}
