using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Evolution.Prompts;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution.Programs.Outputs;

/// <summary>Writes the best program of a run to disk, with an info document describing it.</summary>
/// <remarks>
/// <para>
/// The layout matches the reference OpenEvolve controller
/// (<c>openevolve/controller.py</c> <c>_save_checkpoint</c> and <c>_save_best_program</c>): a checkpoint snapshot
/// goes to <c>checkpoints/checkpoint_&lt;n&gt;/</c> and the final answer to <c>best/</c>, each holding
/// <c>best_program</c> with the extension of its language plus <c>best_program_info.json</c>. The upstream info
/// document carries the program's identity, generation, iteration, metrics, language and timestamps; this one
/// carries all of those and adds what upstream's <c>Program</c> record has no room for - the archive cell and the
/// raw descriptor coordinates behind it, the full lineage including operator, island and inspirations, the
/// evaluation cost and cache status, and the task, evaluator and configuration hashes that say which code and
/// settings produced the score.
/// </para>
/// <para>
/// Both files are written atomically: each is serialized to a temporary file in the destination directory, flushed
/// to disk, and swapped into place, so an interrupted write leaves the previous snapshot rather than a truncated
/// one. Upstream writes both with a plain <c>open(...,"w")</c>, which leaves a half-written program behind if the
/// process dies mid-write - the exact moment a caller most wants the file to be trustworthy.
/// </para>
/// <para>
/// The program text is model-generated and untrusted. It is written as opaque UTF-8 bytes, never executed or
/// interpreted, and cut to <see cref="ProgramRunOutputOptions.MaxSourceBytes"/> on a character boundary so a
/// truncated file is still valid UTF-8; the info document records both the original size and the truncation, so a
/// bounded file never silently passes for a complete one.
/// </para>
/// <para><b>For Beginners:</b> When a search finishes, this is what leaves the winning program on your disk instead
/// of only in memory. Create one with an output directory, then either call it yourself or - more usually - hand it
/// to <see cref="ProgramRunOutputObserver"/>, which calls it at every checkpoint and once more when the run stops.
/// Afterwards, <c>best/best_program.py</c> is the program and <c>best/best_program_info.json</c> tells you how it
/// scored and where it came from.</para>
/// </remarks>
public sealed class ProgramRunOutputWriter
{
    private const int InfoSchemaVersion = 1;
    private const int ProgramSchemaVersion = 1;

    private static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false);

    private readonly HashSet<string> _writtenPrograms = new(StringComparer.Ordinal);
    private readonly object _gate = new();
    private int _writtenProgramCount;
    private readonly string _outputDirectory;
    private readonly ProgramRunOutputOptions _options;

    /// <summary>Initializes a writer, creating the output directory when it does not exist.</summary>
    /// <param name="outputDirectory">The directory that holds the <c>best</c> and <c>checkpoints</c> trees.</param>
    /// <param name="options">The layout and limits; <c>null</c> uses the reference-matching defaults.</param>
    /// <exception cref="ArgumentNullException"><paramref name="outputDirectory"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="outputDirectory"/> is empty, or a name in <paramref name="options"/> is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A limit in <paramref name="options"/> is out of range.</exception>
    public ProgramRunOutputWriter(string outputDirectory, ProgramRunOutputOptions? options = null)
    {
        Guard.NotNullOrWhiteSpace(outputDirectory);
        ProgramRunOutputOptions effective = options is null ? new ProgramRunOutputOptions() : options.Clone();
        effective.Validate();
        _options = effective;
        _outputDirectory = Path.GetFullPath(outputDirectory);
        Directory.CreateDirectory(_outputDirectory);
    }

    /// <summary>Gets the absolute directory that holds the run's output tree.</summary>
    public string OutputDirectory => _outputDirectory;

    /// <summary>Gets an independent copy of the settings this writer was validated with.</summary>
    /// <returns>A copy that a caller may mutate without affecting this instance.</returns>
    public ProgramRunOutputOptions GetOptions() => _options.Clone();

    /// <summary>Writes the run's final answer into the <c>best</c> directory.</summary>
    /// <param name="best">The archive entry holding the winning program and its evaluation.</param>
    /// <param name="note">An optional short note recorded in the info document, such as the stop reason.</param>
    /// <returns>A record naming the files that were written.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="best"/> is <c>null</c>.</exception>
    public ProgramRunOutputRecord WriteFinal(EvolutionArchiveEntry<ProgramGenome> best, string? note = null) =>
        Write(best, ProgramRunOutputTrigger.RunEnd, 0, note);

    /// <summary>Writes a snapshot into a numbered checkpoint directory.</summary>
    /// <param name="best">The archive entry holding the best program so far and its evaluation.</param>
    /// <param name="ordinal">The non-negative number that names the checkpoint directory.</param>
    /// <param name="note">An optional short note recorded in the info document.</param>
    /// <returns>A record naming the files that were written.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="best"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ordinal"/> is negative.</exception>
    public ProgramRunOutputRecord WriteCheckpoint(EvolutionArchiveEntry<ProgramGenome> best, long ordinal, string? note = null) =>
        Write(best, ProgramRunOutputTrigger.Checkpoint, ordinal, note);

    /// <summary>Writes the best program and its info document to the directory the trigger selects.</summary>
    /// <param name="best">The archive entry holding the program and the evaluation that placed it.</param>
    /// <param name="trigger">What caused the write, which chooses the destination directory.</param>
    /// <param name="ordinal">The non-negative checkpoint ordinal; ignored for a final or manual write.</param>
    /// <param name="note">An optional short note recorded in the info document; longer text is truncated.</param>
    /// <returns>A record naming the files that were written.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="best"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ordinal"/> is negative, or <paramref name="trigger"/> is undefined.</exception>
    /// <exception cref="IOException">The output directory could not be written.</exception>
    public ProgramRunOutputRecord Write(
        EvolutionArchiveEntry<ProgramGenome> best,
        ProgramRunOutputTrigger trigger,
        long ordinal = 0,
        string? note = null)
    {
        Guard.NotNull(best);
        if (!Enum.IsDefined(typeof(ProgramRunOutputTrigger), trigger)) throw new ArgumentOutOfRangeException(nameof(trigger));
        if (ordinal < 0) throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "Value cannot be negative.");

        ProgramGenome genome = best.Candidate.CanonicalGenome.Genome;
        EvolutionEvaluation evaluation = best.Evaluation;

        // Cutting on a character boundary keeps a truncated program valid UTF-8 instead of ending it
        // in half an encoded character, which a reader would see as a corrupt file rather than a short one.
        string bounded = PromptTextRedactor.BoundToUtf8Bytes(genome.Source, _options.MaxSourceBytes, out bool truncated);
        byte[] source = Utf8.GetBytes(bounded);

        string directory = GetDirectory(trigger, ordinal);
        string programPath = Path.Combine(directory, _options.ProgramFileNameStem + GetExtension(genome.Language));
        string infoPath = Path.Combine(directory, _options.InfoFileName);
        DateTimeOffset savedAt = DateTimeOffset.UtcNow;
        var document = new InfoDocument
        {
            SchemaVersion = InfoSchemaVersion,
            RunId = _options.RunId,
            Trigger = trigger.ToString(),
            Ordinal = ordinal,
            Note = Bound(note),
            GenomeId = best.Candidate.CanonicalGenome.Id,
            EvaluationId = evaluation.EvaluationId,
            Language = genome.Language.ToString(),
            ProgramFileName = Path.GetFileName(programPath),
            SourceLength = genome.Source.Length,
            SourceLineCount = genome.LineCount,
            SourceSha256 = EvolutionHash.Compute(genome.NormalizedSource),
            IsSourceTruncated = truncated,
            Description = Bound(genome.Description),
            Status = evaluation.Status.ToString(),
            Quality = evaluation.Quality,
            Direction = evaluation.Direction.ToString(),
            CacheStatus = evaluation.CacheStatus.ToString(),
            Cell = best.Cell.Bins.ToList(),
            CellKey = best.Cell.StableKey,
            Descriptors = evaluation.Descriptors
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal),
            Objectives = evaluation.Objectives.ToList(),
            ConstraintViolations = evaluation.ConstraintViolations.ToList(),
            Generation = evaluation.Lineage.Generation,
            Island = evaluation.Lineage.Island,
            ParentIds = evaluation.Lineage.ParentIds.ToList(),
            InspirationIds = evaluation.Lineage.InspirationIds.ToList(),
            VariationOperatorId = evaluation.Lineage.VariationOperatorId,
            RefinerId = evaluation.Lineage.RefinerId,
            SeedStream = evaluation.Lineage.SeedStream,
            ElapsedMilliseconds = evaluation.Cost.Elapsed.TotalMilliseconds,
            AttemptCount = evaluation.Cost.AttemptCount,
            CostUnits = evaluation.Cost.CostUnits,
            TaskVersionHash = evaluation.TaskVersionHash,
            EvaluatorVersionHash = evaluation.EvaluatorVersionHash,
            ConfigurationHash = evaluation.ConfigurationHash,
            SavedAtUtc = savedAt
        };

        lock (_gate)
        {
            Directory.CreateDirectory(directory);
            WriteAtomic(programPath, source);
            WriteAtomic(infoPath, Utf8.GetBytes(JsonConvert.SerializeObject(document, Formatting.Indented)));
        }

        return new ProgramRunOutputRecord(trigger, ordinal, best.Candidate.CanonicalGenome.Id, directory,
            programPath, infoPath, evaluation.Quality, truncated, savedAt);
    }

    /// <summary>Writes one evaluated candidate as its own file, program text and all.</summary>
    /// <param name="entry">The candidate and its evaluation.</param>
    /// <returns>The path written, or <c>null</c> when the retention limit has already been reached.</returns>
    /// <remarks>
    /// <para>
    /// The best-program files say what a run produced; these say how it got there. The archive keeps one candidate
    /// per cell and discards the rest, so by the time a run ends most of the programs that led to the winner exist
    /// nowhere else. One file per candidate is what makes a finished run auditable after the fact and what makes its
    /// trajectories usable as training data, and it is the layout the reference implementation writes to
    /// <c>programs/&lt;id&gt;.json</c>.
    /// </para>
    /// <para>
    /// The file is named from the candidate's canonical identity, so re-writing the same candidate overwrites rather
    /// than duplicates, and the name is reduced to a hash whenever the identity is not a safe file name - an identity
    /// is a value the task chooses, and a value from outside must never decide a path. The write is atomic and the
    /// source is bounded exactly as the best-program file is.
    /// </para>
    /// </remarks>
    /// <param name="candidate">The candidate whose program is being recorded.</param>
    /// <param name="evaluation">Its terminal evaluation.</param>
    /// <param name="cell">
    /// The archive cell it occupied, when that is known. It usually is not: a candidate is recorded as it commits,
    /// and the cell is the archive's to compute. The raw <c>Descriptors</c> are always recorded, and the cell is
    /// derived from them, so nothing is lost by leaving this <c>null</c> rather than guessing.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="candidate"/> or <paramref name="evaluation"/> is <c>null</c>.</exception>
    /// <exception cref="IOException">The output directory could not be written.</exception>
    public string? WriteProgram(
        EvolutionCandidate<ProgramGenome> candidate,
        EvolutionEvaluation evaluation,
        EvolutionCellKey? cell = null)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(evaluation);
        ProgramGenome genome = candidate.CanonicalGenome.Genome;
        string identity = candidate.CanonicalGenome.Id;
        string bounded = PromptTextRedactor.BoundToUtf8Bytes(genome.Source, _options.MaxSourceBytes, out bool truncated);

        var document = new ProgramDocument
        {
            SchemaVersion = ProgramSchemaVersion,
            RunId = _options.RunId,
            GenomeId = identity,
            EvaluationId = evaluation.EvaluationId,
            Language = genome.Language.ToString(),
            Source = bounded,
            IsSourceTruncated = truncated,
            SourceLength = genome.Source.Length,
            SourceSha256 = EvolutionHash.Compute(genome.NormalizedSource),
            Description = Bound(genome.Description),
            IsDescriptionTruncated = genome.Description is { } full && full.Length > MaxDescriptionLength,
            Status = evaluation.Status.ToString(),
            Quality = evaluation.Quality,
            Direction = evaluation.Direction.ToString(),
            CacheStatus = evaluation.CacheStatus.ToString(),
            Diagnostics = evaluation.Diagnostics
                .Select(diagnostic => diagnostic.Code + ": " + diagnostic.Message).ToList(),
            Cell = cell?.Bins.ToList(),
            CellKey = cell?.StableKey,
            Descriptors = evaluation.Descriptors
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal),
            Metrics = evaluation.Metrics
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal),
            Generation = evaluation.Lineage.Generation,
            Island = evaluation.Lineage.Island,
            ParentIds = evaluation.Lineage.ParentIds.ToList(),
            InspirationIds = evaluation.Lineage.InspirationIds.ToList(),
            VariationOperatorId = evaluation.Lineage.VariationOperatorId,
            RefinerId = evaluation.Lineage.RefinerId,
            SeedStream = evaluation.Lineage.SeedStream,
            ElapsedMilliseconds = evaluation.Cost.Elapsed.TotalMilliseconds,
            AttemptCount = evaluation.Cost.AttemptCount,
            CostUnits = evaluation.Cost.CostUnits,
            TaskVersionHash = evaluation.TaskVersionHash,
            EvaluatorVersionHash = evaluation.EvaluatorVersionHash,
            ConfigurationHash = evaluation.ConfigurationHash,
            SavedAtUtc = DateTimeOffset.UtcNow
        };

        string directory = Path.Combine(_outputDirectory, _options.ProgramsDirectoryName);
        string path = Path.Combine(directory, SafeFileName(identity) + ".json");
        lock (_gate)
        {
            // The set of written paths exists only to enforce a limit, so it is kept only when there is one. Without
            // this a long unbounded run accumulates one path string per evaluation for no purpose at all.
            if (_options.MaxRetainedPrograms > 0)
            {
                if (!_writtenPrograms.Contains(path) && _writtenPrograms.Count >= _options.MaxRetainedPrograms)
                    return null;
                _writtenPrograms.Add(path);
            }

            _writtenProgramCount++;
            Directory.CreateDirectory(directory);
            WriteAtomic(path, Utf8.GetBytes(JsonConvert.SerializeObject(document, Formatting.Indented)));
        }
        return path;
    }

    /// <summary>Gets how many per-candidate files this writer has written, rewrites included.</summary>
    public int WrittenProgramCount
    {
        get { lock (_gate) { return _writtenProgramCount; } }
    }

    /// <summary>Reduces a candidate identity to a name that is safe as a file name.</summary>
    /// <param name="identity">The canonical genome identity.</param>
    /// <returns>The identity when it is already safe, otherwise a hash of it.</returns>
    /// <remarks>
    /// A genome identity is whatever the task decided it is, which means it is a value from outside this code, and a
    /// value from outside must never be allowed to decide a path. Anything that is not plainly safe is replaced by a
    /// hash of itself, which stays stable and unique without being able to escape the directory.
    /// </remarks>
    private static string SafeFileName(string identity)
    {
        if (identity.Length > 0 && identity.Length <= 100 && identity != "." && identity != ".." &&
            identity.All(character => char.IsLetterOrDigit(character) || character is '-' or '_'))
        {
            return identity;
        }
        return EvolutionHash.Compute(identity);
    }

    /// <summary>Computes the directory one write would target, whether or not it exists.</summary>
    /// <param name="trigger">What would cause the write.</param>
    /// <param name="ordinal">The non-negative checkpoint ordinal; ignored for a final or manual write.</param>
    /// <returns>An absolute path under <see cref="OutputDirectory"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ordinal"/> is negative, or <paramref name="trigger"/> is undefined.</exception>
    public string GetDirectory(ProgramRunOutputTrigger trigger, long ordinal = 0)
    {
        if (!Enum.IsDefined(typeof(ProgramRunOutputTrigger), trigger)) throw new ArgumentOutOfRangeException(nameof(trigger));
        if (ordinal < 0) throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "Value cannot be negative.");
        if (trigger != ProgramRunOutputTrigger.Checkpoint) return Path.Combine(_outputDirectory, _options.BestDirectoryName);
        return Path.Combine(_outputDirectory, _options.CheckpointsDirectoryName,
            _options.CheckpointDirectoryPrefix + ordinal.ToString(CultureInfo.InvariantCulture));
    }

    private static string GetExtension(ProgramLanguage language) => ProgramLanguageDetector.GetFileExtension(language);

    /// <summary>The longest description recorded verbatim; longer ones are cut and the cut is flagged.</summary>
    private const int MaxDescriptionLength = 512;

    private static string? Bound(string? value)
    {
        if (value is null) return null;
        return value.Length > MaxDescriptionLength ? value.Substring(0, MaxDescriptionLength) : value;
    }

    private static void WriteAtomic(string path, byte[] payload)
    {
        string directory = Path.GetDirectoryName(path) ?? ".";
        string tempPath = Path.Combine(directory, "." + Path.GetFileName(path) + "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(payload, 0, payload.Length);
                stream.Flush(flushToDisk: true);
            }

            if (File.Exists(path)) File.Replace(tempPath, path, destinationBackupFileName: null, ignoreMetadataErrors: true);
            else File.Move(tempPath, path);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }
        }
    }

    /// <summary>Serialization shape of one per-candidate program document.</summary>
    /// <remarks>
    /// The program text is inside the document rather than beside it, because a per-candidate file is read as a
    /// record rather than executed as a program, and one file per candidate is already one file too many to pair up.
    /// </remarks>
    private sealed class ProgramDocument
    {
        /// <summary>Gets or sets the document schema version.</summary>
        public int SchemaVersion { get; set; }
        /// <summary>Gets or sets the optional run identifier.</summary>
        public string? RunId { get; set; }
        /// <summary>Gets or sets the canonical genome identity.</summary>
        public string GenomeId { get; set; } = string.Empty;
        /// <summary>Gets or sets the evaluation identifier that produced the score.</summary>
        public long EvaluationId { get; set; }
        /// <summary>Gets or sets the program language.</summary>
        public string Language { get; set; } = string.Empty;
        /// <summary>Gets or sets the model-generated program source, bounded to the configured size.</summary>
        public string Source { get; set; } = string.Empty;
        /// <summary>Gets or sets whether the recorded source was cut to the configured limit.</summary>
        public bool IsSourceTruncated { get; set; }
        /// <summary>Gets or sets the source length in characters before truncation.</summary>
        public int SourceLength { get; set; }
        /// <summary>Gets or sets the hash of the normalized source.</summary>
        public string SourceSha256 { get; set; } = string.Empty;
        /// <summary>Gets or sets the genome's bounded description.</summary>
        public string? Description { get; set; }
        /// <summary>Gets or sets whether the recorded description was cut to the limit.</summary>
        public bool IsDescriptionTruncated { get; set; }
        /// <summary>Gets or sets the terminal evaluation status.</summary>
        public string Status { get; set; } = string.Empty;
        /// <summary>Gets or sets the scalar quality, absent when the candidate did not complete.</summary>
        public double? Quality { get; set; }
        /// <summary>Gets or sets whether larger or smaller qualities are better.</summary>
        public string Direction { get; set; } = string.Empty;
        /// <summary>Gets or sets whether the score was computed or served from the cache.</summary>
        public string CacheStatus { get; set; } = string.Empty;
        /// <summary>Gets or sets the diagnostics the evaluation reported, which say why a failure failed.</summary>
        public List<string> Diagnostics { get; set; } = new();
        /// <summary>Gets or sets the archive cell bin indices, or <c>null</c> when the cell was not known.</summary>
        public List<int>? Cell { get; set; }
        /// <summary>Gets or sets the culture-independent archive cell key, or <c>null</c> when it was not known.</summary>
        public string? CellKey { get; set; }
        /// <summary>Gets or sets the raw descriptor coordinates a cell is derived from.</summary>
        public Dictionary<string, double> Descriptors { get; set; } = new(StringComparer.Ordinal);
        /// <summary>Gets or sets every metric the evaluation reported.</summary>
        public Dictionary<string, double> Metrics { get; set; } = new(StringComparer.Ordinal);
        /// <summary>Gets or sets the generation the candidate was proposed in.</summary>
        public long Generation { get; set; }
        /// <summary>Gets or sets the island the candidate belonged to.</summary>
        public int Island { get; set; }
        /// <summary>Gets or sets the parents the candidate was derived from.</summary>
        public List<string> ParentIds { get; set; } = new();
        /// <summary>Gets or sets the inspirations shown to the operator.</summary>
        public List<string> InspirationIds { get; set; } = new();
        /// <summary>Gets or sets the variation operator that proposed the candidate.</summary>
        public string VariationOperatorId { get; set; } = string.Empty;
        /// <summary>Gets or sets the refiner applied before evaluation, when there was one.</summary>
        public string? RefinerId { get; set; }
        /// <summary>Gets or sets the deterministic random stream the proposal used.</summary>
        public ulong SeedStream { get; set; }
        /// <summary>Gets or sets how long the evaluation took.</summary>
        public double ElapsedMilliseconds { get; set; }
        /// <summary>Gets or sets how many attempts the evaluation needed.</summary>
        public int AttemptCount { get; set; }
        /// <summary>Gets or sets the cost units the evaluation reported.</summary>
        public double CostUnits { get; set; }
        /// <summary>Gets or sets the hash of the task that produced the score.</summary>
        public string TaskVersionHash { get; set; } = string.Empty;
        /// <summary>Gets or sets the hash of the evaluator that produced the score.</summary>
        public string EvaluatorVersionHash { get; set; } = string.Empty;
        /// <summary>Gets or sets the hash of the configuration the run used.</summary>
        public string ConfigurationHash { get; set; } = string.Empty;
        /// <summary>Gets or sets when the document was written.</summary>
        public DateTimeOffset SavedAtUtc { get; set; }
    }

    /// <summary>Serialization shape of one best-program info document.</summary>
    private sealed class InfoDocument
    {
        /// <summary>Gets or sets the document schema version.</summary>
        public int SchemaVersion { get; set; }
        /// <summary>Gets or sets the optional run identifier.</summary>
        public string? RunId { get; set; }
        /// <summary>Gets or sets what caused the write.</summary>
        public string Trigger { get; set; } = string.Empty;
        /// <summary>Gets or sets the checkpoint ordinal.</summary>
        public long Ordinal { get; set; }
        /// <summary>Gets or sets an optional bounded note such as the stop reason.</summary>
        public string? Note { get; set; }
        /// <summary>Gets or sets the canonical genome identity.</summary>
        public string GenomeId { get; set; } = string.Empty;
        /// <summary>Gets or sets the evaluation identifier that produced the score.</summary>
        public long EvaluationId { get; set; }
        /// <summary>Gets or sets the program language.</summary>
        public string Language { get; set; } = string.Empty;
        /// <summary>Gets or sets the program file name beside this document.</summary>
        public string ProgramFileName { get; set; } = string.Empty;
        /// <summary>Gets or sets the source length in characters before truncation.</summary>
        public int SourceLength { get; set; }
        /// <summary>Gets or sets the source line count before truncation.</summary>
        public int SourceLineCount { get; set; }
        /// <summary>Gets or sets the hash of the normalized source.</summary>
        public string SourceSha256 { get; set; } = string.Empty;
        /// <summary>Gets or sets whether the written program was cut to the configured limit.</summary>
        public bool IsSourceTruncated { get; set; }
        /// <summary>Gets or sets the genome's bounded description.</summary>
        public string? Description { get; set; }
        /// <summary>Gets or sets the terminal evaluation status.</summary>
        public string Status { get; set; } = string.Empty;
        /// <summary>Gets or sets the scalar quality.</summary>
        public double? Quality { get; set; }
        /// <summary>Gets or sets whether larger or smaller qualities are better.</summary>
        public string Direction { get; set; } = string.Empty;
        /// <summary>Gets or sets whether the score was computed or served from the cache.</summary>
        public string CacheStatus { get; set; } = string.Empty;
        /// <summary>Gets or sets the archive cell bin indices.</summary>
        public List<int> Cell { get; set; } = new();
        /// <summary>Gets or sets the culture-independent archive cell key.</summary>
        public string CellKey { get; set; } = string.Empty;
        /// <summary>Gets or sets the raw descriptor coordinates behind the cell.</summary>
        public Dictionary<string, double> Descriptors { get; set; } = new(StringComparer.Ordinal);
        /// <summary>Gets or sets the multi-objective values.</summary>
        public List<double> Objectives { get; set; } = new();
        /// <summary>Gets or sets the constraint violation magnitudes.</summary>
        public List<double> ConstraintViolations { get; set; } = new();
        /// <summary>Gets or sets the logical generation.</summary>
        public long Generation { get; set; }
        /// <summary>Gets or sets the zero-based island index.</summary>
        public int Island { get; set; }
        /// <summary>Gets or sets the canonical identities of the direct parents.</summary>
        public List<string> ParentIds { get; set; } = new();
        /// <summary>Gets or sets the canonical identities supplied as variation inspirations.</summary>
        public List<string> InspirationIds { get; set; } = new();
        /// <summary>Gets or sets the operator that produced the program.</summary>
        public string VariationOperatorId { get; set; } = string.Empty;
        /// <summary>Gets or sets the optional refiner identifier.</summary>
        public string? RefinerId { get; set; }
        /// <summary>Gets or sets the deterministic random stream identifier.</summary>
        public ulong SeedStream { get; set; }
        /// <summary>Gets or sets the wall-clock evaluator time in milliseconds.</summary>
        public double ElapsedMilliseconds { get; set; }
        /// <summary>Gets or sets how many attempts the evaluation needed.</summary>
        public int AttemptCount { get; set; }
        /// <summary>Gets or sets the task-defined resource units the evaluation charged.</summary>
        public double CostUnits { get; set; }
        /// <summary>Gets or sets the task version hash in effect for the evaluation.</summary>
        public string TaskVersionHash { get; set; } = string.Empty;
        /// <summary>Gets or sets the evaluator version hash in effect for the evaluation.</summary>
        public string EvaluatorVersionHash { get; set; } = string.Empty;
        /// <summary>Gets or sets the engine configuration hash in effect for the evaluation.</summary>
        public string ConfigurationHash { get; set; } = string.Empty;
        /// <summary>Gets or sets when the write completed.</summary>
        public DateTimeOffset SavedAtUtc { get; set; }
    }
}
