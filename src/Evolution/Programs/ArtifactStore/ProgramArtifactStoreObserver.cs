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
    private readonly object _gate = new();
    private int _stored;
    private int _failures;
    private string? _lastError;

    /// <summary>Initializes an observer that writes into the supplied store.</summary>
    /// <param name="store">The store that decides inline-versus-file placement and retention.</param>
    /// <exception cref="ArgumentNullException"><paramref name="store"/> is <see langword="null"/>.</exception>
    public ProgramArtifactStoreObserver(IProgramArtifactStore store)
    {
        Guard.NotNull(store);
        _store = store;
    }

    /// <summary>Gets the number of evaluations whose artifacts were written.</summary>
    public int StoredCount { get { lock (_gate) { return _stored; } } }

    /// <summary>Gets the number of evaluations whose artifacts could not be written.</summary>
    public int FailureCount { get { lock (_gate) { return _failures; } } }

    /// <summary>Gets the most recent failure message, or <c>null</c> when nothing has failed.</summary>
    public string? LastError { get { lock (_gate) { return _lastError; } } }

    /// <inheritdoc/>
    public async ValueTask OnEventAsync(
        EvolutionEvent<ProgramGenome> evolutionEvent,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(evolutionEvent);
        if (evolutionEvent.Kind != EvolutionEventKind.Evaluated) return;
        if (evolutionEvent.Evaluation is not { } evaluation) return;
        if (evaluation.Artifacts.Count == 0) return;

        var artifacts = new List<ProgramArtifact>(evaluation.Artifacts.Count);
        foreach (EvolutionArtifact artifact in evaluation.Artifacts)
        {
            artifacts.Add(ProgramArtifact.FromText(artifact.Key, artifact.Text, artifact.IsTruncated));
        }

        try
        {
            await _store.StoreAsync(evaluation.GenomeId, artifacts, cancellationToken).ConfigureAwait(false);
            lock (_gate) { _stored++; }
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
        }
    }
}
