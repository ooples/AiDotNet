using System.Collections.ObjectModel;
using AiDotNet.Configuration;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Outputs;

/// <summary>Writes the best program to disk at every checkpoint and once more when the run stops.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve controller performs these writes inside the engine: <c>_save_checkpoint</c> is called
/// from the evolution loop and <c>_save_best_program</c> from the end of <c>run</c>. Here the same two writes are
/// driven from outside, through the observer stream the engine already publishes - a
/// <see cref="EvolutionEventKind.Checkpointed"/> event triggers the snapshot and a
/// <see cref="EvolutionEventKind.Stopped"/> event triggers the final write - so run output is a component a caller
/// opts into rather than behaviour welded into the search.
/// </para>
/// <para>
/// The observer needs the archives to answer "which program is best", because a checkpoint event carries no
/// candidate. Register them with <see cref="AddArchive"/> from the archive factory the engine is constructed with,
/// or pass them to the constructor when they already exist. The winner is chosen exactly as
/// <see cref="EvolutionRunResult{TGenome}.Best"/> chooses it: best quality in the archive's own optimisation
/// direction, ties broken by canonical genome identity, so the answer does not depend on island ordering or timing.
/// </para>
/// <para>
/// Write failures never stop a run. The engine already isolates observer exceptions, but silently losing the
/// output is worse than losing it loudly, so a failed write is recorded in <see cref="LastError"/> and the run
/// continues; a caller that cares can assert on it after the run. The observer holds no unbounded state: it keeps
/// one record per write, and records never contain program text.
/// </para>
/// <para><b>For Beginners:</b> Hand this to the evolution engine as its observer and your run will leave the
/// winning program on disk instead of only in memory - a copy at every checkpoint under
/// <c>checkpoints/checkpoint_1/</c>, <c>checkpoint_2/</c> and so on, and the final answer under <c>best/</c>. The
/// one wiring step is registering the archives, which you do inside the archive factory you already pass to the
/// engine so the observer knows where to look for the best program.</para>
/// </remarks>
public sealed class ProgramRunOutputObserver : IEvolutionObserver<ProgramGenome>
{
    private const int MaxNoteLength = 256;

    private long _programsWritten;

    private readonly object _gate = new();
    private readonly List<IEvolutionArchiveView<ProgramGenome>> _archives = new();
    private readonly List<ProgramRunOutputRecord> _records = new();
    private readonly ProgramRunOutputWriter _writer;
    private readonly ProgramRunOutputOptions _options;
    private long _checkpointOrdinal;

    /// <summary>Initializes an observer.</summary>
    /// <param name="writer">The writer that produces the files.</param>
    /// <param name="archives">Archives to read the best program from, or <c>null</c> to register them later.</param>
    /// <exception cref="ArgumentNullException"><paramref name="writer"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="archives"/> contains a <c>null</c> entry.</exception>
    public ProgramRunOutputObserver(
        ProgramRunOutputWriter writer,
        IEnumerable<IEvolutionArchiveView<ProgramGenome>>? archives = null)
    {
        Guard.NotNull(writer);
        _writer = writer;
        _options = writer.GetOptions();
        if (archives is null) return;
        foreach (IEvolutionArchiveView<ProgramGenome> archive in archives)
        {
            if (archive is null) throw new ArgumentException("Archives cannot contain null entries.", nameof(archives));
            _archives.Add(archive);
        }
    }

    /// <summary>Gets the records of every write this observer has made, in order.</summary>
    public IReadOnlyList<ProgramRunOutputRecord> Records
    {
        get { lock (_gate) { return new ReadOnlyCollection<ProgramRunOutputRecord>(_records.ToArray()); } }
    }

    /// <summary>Gets the most recent write, or <c>null</c> when nothing has been written.</summary>
    public ProgramRunOutputRecord? LastRecord
    {
        get { lock (_gate) { return _records.Count == 0 ? null : _records[_records.Count - 1]; } }
    }

    /// <summary>Gets a bounded description of the most recent failed write, or <c>null</c> when none has failed.</summary>
    public string? LastError { get; private set; }

    /// <summary>Gets how many per-candidate program files this observer has written.</summary>
    public long ProgramsWritten => Interlocked.Read(ref _programsWritten);

    /// <summary>Registers one archive the observer reads the best program from.</summary>
    /// <param name="archive">The archive view, typically captured inside the engine's archive factory.</param>
    /// <exception cref="ArgumentNullException"><paramref name="archive"/> is <c>null</c>.</exception>
    /// <remarks>Registering the same archive twice is harmless; the winner is chosen deterministically regardless.</remarks>
    public void AddArchive(IEvolutionArchiveView<ProgramGenome> archive)
    {
        Guard.NotNull(archive);
        lock (_gate) { _archives.Add(archive); }
    }

    /// <inheritdoc/>
    public ValueTask OnEventAsync(EvolutionEvent<ProgramGenome> evolutionEvent, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(evolutionEvent);
        cancellationToken.ThrowIfCancellationRequested();
        if (evolutionEvent.Kind == EvolutionEventKind.Evaluated && _options.WriteEveryProgram)
        {
            TryWriteProgram(evolutionEvent);
        }

        if (evolutionEvent.Kind == EvolutionEventKind.Checkpointed && _options.WriteAtCheckpoints)
        {
            long ordinal;
            lock (_gate) { ordinal = ++_checkpointOrdinal; }
            TryWrite(ProgramRunOutputTrigger.Checkpoint, ordinal, evolutionEvent.Message);
        }
        else if (evolutionEvent.Kind == EvolutionEventKind.Stopped && _options.WriteAtRunEnd)
        {
            TryWrite(ProgramRunOutputTrigger.RunEnd, 0, evolutionEvent.Message);
        }

        return default;
    }

    /// <summary>Writes the current best program immediately, outside the observer event stream.</summary>
    /// <param name="note">An optional short note recorded in the info document.</param>
    /// <returns>The record of the write, or <c>null</c> when no archive holds a scored program.</returns>
    /// <exception cref="IOException">The output directory could not be written.</exception>
    public ProgramRunOutputRecord? WriteNow(string? note = null)
    {
        EvolutionArchiveEntry<ProgramGenome>? best = SelectBest();
        if (best is null) return null;
        ProgramRunOutputRecord record = _writer.Write(best, ProgramRunOutputTrigger.Manual, 0, Bound(note));
        lock (_gate) { _records.Add(record); }
        return record;
    }

    /// <summary>Selects the globally best archived program using quality then canonical identity.</summary>
    /// <returns>The best entry across every registered archive, or <c>null</c> when none holds one.</returns>
    public EvolutionArchiveEntry<ProgramGenome>? SelectBest()
    {
        IEvolutionArchiveView<ProgramGenome>[] archives;
        lock (_gate) { archives = _archives.ToArray(); }
        if (archives.Length == 0) return null;
        bool maximize = archives[0].Direction == EvolutionOptimizationDirection.Maximize;
        return archives
            .Select(archive => archive.Best)
            .OfType<EvolutionArchiveEntry<ProgramGenome>>()
            .OrderBy(entry => entry.Evaluation.Quality, maximize
                ? Comparer<double?>.Create((left, right) => Nullable.Compare(right, left))
                : Comparer<double?>.Default)
            .ThenBy(entry => entry.Evaluation.GenomeId, StringComparer.Ordinal)
            .FirstOrDefault();
    }

    /// <summary>Writes one completed candidate to its own file, if the run asked for that.</summary>
    /// <remarks>
    /// Written from the commit event rather than at the end of the run, because by then the archive has discarded
    /// every candidate that lost its cell, which is most of them. A failure to write is recorded and swallowed: a
    /// full disk should cost an audit trail, not a search that was otherwise going fine.
    /// </remarks>
    private void TryWriteProgram(EvolutionEvent<ProgramGenome> evolutionEvent)
    {
        EvolutionCandidate<ProgramGenome>? candidate = evolutionEvent.Candidate;
        EvolutionEvaluation? evaluation = evolutionEvent.Evaluation;
        if (candidate is null || evaluation is null || evaluation.Status != EvolutionEvaluationStatus.Completed) return;

        // The cell is left out rather than guessed: the archives belong to the engine and are not visible until the
        // run ends, and the raw descriptors a cell is derived from are recorded either way.
        try
        {
            string? path = _writer.WriteProgram(candidate, evaluation);
            if (path is not null) Interlocked.Increment(ref _programsWritten);
        }
        catch (IOException exception)
        {
            LastError = Bound(exception.Message);
        }
        catch (UnauthorizedAccessException exception)
        {
            LastError = Bound(exception.Message);
        }
    }

    private void TryWrite(ProgramRunOutputTrigger trigger, long ordinal, string? note)
    {
        EvolutionArchiveEntry<ProgramGenome>? best = SelectBest();
        if (best is null) return;
        try
        {
            ProgramRunOutputRecord record = _writer.Write(best, trigger, ordinal, Bound(note));
            lock (_gate) { _records.Add(record); }
        }
        catch (IOException exception)
        {
            LastError = Bound(exception.Message);
        }
        catch (UnauthorizedAccessException exception)
        {
            LastError = Bound(exception.Message);
        }
    }

    private static string? Bound(string? value)
    {
        if (value is null) return null;
        return value.Length > MaxNoteLength ? value.Substring(0, MaxNoteLength) : value;
    }
}
