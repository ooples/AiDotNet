using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.ArtifactStore;

/// <summary>Persists each evaluation's artifacts so they outlive the run and stay retrievable by genome.</summary>
/// <remarks>
/// <para>
/// The engine keeps artifacts in memory and hands them to exactly one follow-up proposal, after which they are
/// gone. That is the right lifetime for prompt feedback and the wrong one for an audit: once a run ends there is
/// no way to ask why a particular candidate scored as it did. This observer copies each evaluation's artifacts
/// into an <see cref="IProgramArtifactStore"/> as they are committed, where the store's own size threshold decides
/// what stays inline and what is promoted to a file, and its retention policy bounds the total.
/// </para>
/// <para>
/// Failures are swallowed and counted rather than thrown. An observer runs inside the engine's commit path, so an
/// unwritable disk must not be able to end a search that is otherwise progressing; <see cref="FailureCount"/> and
/// <see cref="LastError"/> report what happened. Artifact text is untrusted evaluator output, so it is stored as
/// data and never interpreted.
/// </para>
/// <para><b>For Beginners:</b> Artifacts are the notes an evaluation leaves behind, such as a compiler error or a
/// captured stdout. Without this they vanish when the run ends. With it they are written next to the run, and you
/// can look up everything a given candidate produced afterwards.</para>
/// </remarks>
public sealed class ProgramArtifactStoreObserver : IEvolutionObserver<ProgramGenome>
{
    private readonly IProgramArtifactStore _store;
    private readonly int _purgeEveryStores;
    private readonly Func<DateTimeOffset> _clock;
    private readonly object _gate = new();
    private int _stored;
    private int _failures;
    private int _sweeps;
    private int _removed;
    private int _sinceSweep;
    private string? _lastError;

    /// <summary>Initializes an observer that writes into the supplied store.</summary>
    /// <param name="store">The store that decides inline-versus-file placement and retention.</param>
    /// <param name="purgeEveryStores">
    /// How many stored evaluations pass between retention sweeps; zero sweeps only when the search stops. Defaults
    /// to <see cref="ProgramArtifactStoreOptions.DefaultPurgeEveryStores"/>.
    /// </param>
    /// <param name="clock">
    /// Supplies the time a sweep measures ages against, or <see langword="null"/> for the system clock.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="store"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="purgeEveryStores"/> is negative.</exception>
    public ProgramArtifactStoreObserver(
        IProgramArtifactStore store,
        int purgeEveryStores = ProgramArtifactStoreOptions.DefaultPurgeEveryStores,
        Func<DateTimeOffset>? clock = null)
    {
        Guard.NotNull(store);
        if (purgeEveryStores < 0)
            throw new ArgumentOutOfRangeException(nameof(purgeEveryStores), purgeEveryStores, "Value cannot be negative.");
        _store = store;
        _purgeEveryStores = purgeEveryStores;
        _clock = clock ?? (() => DateTimeOffset.UtcNow);
    }

    /// <summary>Gets the number of evaluations whose artifacts were written.</summary>
    public int StoredCount { get { lock (_gate) { return _stored; } } }

    /// <summary>Gets the number of evaluations whose artifacts could not be written.</summary>
    public int FailureCount { get { lock (_gate) { return _failures; } } }

    /// <summary>Gets how many retention sweeps have run.</summary>
    public int SweepCount { get { lock (_gate) { return _sweeps; } } }

    /// <summary>Gets how many genomes those sweeps removed in total.</summary>
    public int RemovedCount { get { lock (_gate) { return _removed; } } }

    /// <summary>Gets the most recent failure message, or <c>null</c> when nothing has failed.</summary>
    public string? LastError { get { lock (_gate) { return _lastError; } } }

    /// <inheritdoc/>
    public async ValueTask OnEventAsync(
        EvolutionEvent<ProgramGenome> evolutionEvent,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(evolutionEvent);

        // A search that stops having written anything still leaves a directory behind, and it is the last chance to
        // apply the retention the caller configured, so the final sweep runs whatever the cadence.
        if (evolutionEvent.Kind == EvolutionEventKind.Stopped)
        {
            await SweepAsync(cancellationToken).ConfigureAwait(false);
            return;
        }

        if (evolutionEvent.Kind != EvolutionEventKind.Evaluated) return;
        if (evolutionEvent.Evaluation is not { } evaluation) return;
        if (evaluation.Artifacts.Count == 0) return;

        var artifacts = new List<ProgramArtifact>(evaluation.Artifacts.Count);
        foreach (EvolutionArtifact artifact in evaluation.Artifacts)
        {
            artifacts.Add(ProgramArtifact.FromText(artifact.Key, artifact.Text, artifact.IsTruncated));
        }

        bool due;
        try
        {
            await _store.StoreAsync(evaluation.GenomeId, artifacts, cancellationToken).ConfigureAwait(false);
            lock (_gate)
            {
                _stored++;
                _sinceSweep++;
                due = _purgeEveryStores > 0 && _sinceSweep >= _purgeEveryStores;
                if (due) _sinceSweep = 0;
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException
                                              or ArgumentException or InvalidOperationException)
        {
            // A search that is otherwise progressing must not end because a disk filled up.
            lock (_gate)
            {
                _failures++;
                _lastError = exception.Message;
            }

            return;
        }

        if (due) await SweepAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Runs one retention sweep, counting what it removed and swallowing what it cannot.</summary>
    /// <remarks>
    /// The store already knows what should be kept; nothing was ever calling it, so a configured retention period
    /// bounded nothing and a long run's artifact directory grew until somebody swept it by hand. A sweep that fails
    /// is counted like a failed write rather than thrown: retention is housekeeping, and housekeeping must not be
    /// able to end a search.
    /// </remarks>
    private async ValueTask SweepAsync(CancellationToken cancellationToken)
    {
        try
        {
            int removed = await _store.PurgeAsync(_clock(), cancellationToken).ConfigureAwait(false);
            lock (_gate)
            {
                _sweeps++;
                _removed += removed;
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException
                                              or ArgumentException or InvalidOperationException)
        {
            lock (_gate)
            {
                _failures++;
                _lastError = exception.Message;
            }
        }
    }
}
