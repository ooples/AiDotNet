using System.Globalization;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>
/// Writes provenance records as JSON Lines into a directory of crash-safe, immutable segment files.
/// </summary>
/// <remarks>
/// <para>
/// Records are buffered and written in batches. Each batch becomes one new segment, written with the same idiom
/// the checkpoint store uses: serialize into a temporary file in the target directory, flush it all the way to
/// disk, then move it into place under its final name. A crash therefore leaves every finished segment intact and
/// at worst discards the batch that was in flight — never a half-written line that would poison a reader.
/// </para>
/// <para>
/// Segments are never rewritten, which is what makes this affordable. Appending to one growing file cannot be made
/// crash-safe without either rewriting the whole file (quadratic in the number of records) or accepting torn
/// writes; a new immutable file per batch gives durability at linear cost, and a reader simply reads the segments
/// in name order. Segment numbering resumes from the highest existing segment, so restarting a run into the same
/// directory adds to the stream rather than overwriting it.
/// </para>
/// <para>
/// The sink never blocks a search. Writes happen under one lock so a single instance may be shared by concurrent
/// workers, and the operator that feeds it treats a failure here as a recording problem rather than a run-ending
/// one. Call <see cref="FlushAsync"/> or dispose the sink at the end of a run so the final partial batch reaches
/// disk.
/// </para>
/// <para><b>For Beginners:</b> This writes the notes about each AI conversation to disk, one JSON object per line,
/// in small files inside a folder you choose. Small files are used instead of one big one so that a crash or a
/// power cut can never leave you with a corrupted half-line. Point <c>ProposalProvenanceReader</c> at the same
/// folder afterwards to read everything back. Remember to dispose the sink (or <c>await FlushAsync</c>) when the
/// run finishes, or the last few notes stay in memory.</para>
/// </remarks>
public sealed class JsonLinesProposalProvenanceSink : IProposalProvenanceSink, IDisposable
{
    /// <summary>The file-name stem used when none is supplied.</summary>
    public const string DefaultBaseName = "provenance";

    /// <summary>The extension every segment carries.</summary>
    public const string SegmentExtension = ".jsonl";

    private static readonly JsonSerializerSettings SerializerSettings = new()
    {
        Formatting = Formatting.None,
        NullValueHandling = NullValueHandling.Ignore,
        DateParseHandling = DateParseHandling.None
    };

    private readonly object _gate = new();
    private readonly List<string> _buffer = new();
    private readonly ProposalProvenanceOptions _options;
    private readonly UTF8Encoding _encoding = new(encoderShouldEmitUTF8Identifier: false);
    private long _bufferedBytes;
    private long _recordsWritten;
    private int _segmentsWritten;
    private int _nextSegmentIndex;
    private bool _disposed;

    /// <summary>Initializes a segment-writing sink.</summary>
    /// <param name="directoryPath">The directory segments are written into; created when it does not exist.</param>
    /// <param name="options">Buffering and size budgets; <c>null</c> uses the defaults.</param>
    /// <param name="baseName">The file-name stem for segments; <c>null</c> or empty uses <see cref="DefaultBaseName"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="directoryPath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="directoryPath"/> is empty or white space, or <paramref name="baseName"/> contains a path
    /// separator or another character that is not valid in a file name.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">An option value is outside its permitted range.</exception>
    public JsonLinesProposalProvenanceSink(
        string directoryPath,
        ProposalProvenanceOptions? options = null,
        string? baseName = null)
    {
        Guard.NotNullOrWhiteSpace(directoryPath);

        ProposalProvenanceOptions copy = (options ?? new ProposalProvenanceOptions()).Clone();
        copy.Validate();
        _options = copy;

        // Narrowed by an explicit null check rather than string.IsNullOrWhiteSpace, which carries no nullable
        // annotation on .NET Framework and would leave the following Trim() flagged there.
        string trimmedBaseName = baseName is null ? string.Empty : baseName.Trim();
        string stem = trimmedBaseName.Length == 0 ? DefaultBaseName : trimmedBaseName;
        if (stem.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new ArgumentException(
                "A provenance segment base name cannot contain path separators or other characters that are invalid in a file name.",
                nameof(baseName));
        }

        BaseName = stem;
        DirectoryPath = Path.GetFullPath(directoryPath);
        Directory.CreateDirectory(DirectoryPath);
        _nextSegmentIndex = FindNextSegmentIndex();
    }

    /// <summary>Gets the absolute directory segments are written into.</summary>
    public string DirectoryPath { get; }

    /// <summary>Gets the file-name stem shared by every segment.</summary>
    public string BaseName { get; }

    /// <summary>Gets how many records have been flushed to disk.</summary>
    public long RecordsWritten
    {
        get
        {
            lock (_gate) return _recordsWritten;
        }
    }

    /// <summary>Gets how many segment files this instance has written.</summary>
    public int SegmentsWritten
    {
        get
        {
            lock (_gate) return _segmentsWritten;
        }
    }

    /// <summary>Gets how many records are buffered and not yet on disk.</summary>
    public int PendingRecords
    {
        get
        {
            lock (_gate) return _buffer.Count;
        }
    }

    /// <inheritdoc/>
    /// <exception cref="ObjectDisposedException">The sink has been disposed.</exception>
    public Task RecordAsync(ProposalProvenanceRecord record, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(record);
        cancellationToken.ThrowIfCancellationRequested();

        if (!_options.Enabled) return Task.CompletedTask;

        string line = JsonConvert.SerializeObject(ProposalProvenanceDocument.From(record), SerializerSettings);
        lock (_gate)
        {
            ThrowIfDisposed();
            _buffer.Add(line);
            _bufferedBytes += _encoding.GetByteCount(line) + 1L;
            if (_buffer.Count >= _options.FlushEveryRecords || _bufferedBytes >= _options.MaxSegmentBytes)
            {
                FlushLocked(cancellationToken);
            }
        }

        return Task.CompletedTask;
    }

    /// <summary>Writes every buffered record to a new segment.</summary>
    /// <param name="cancellationToken">Token used to cancel the write.</param>
    /// <returns>A task that completes when the segment is on disk; completes synchronously.</returns>
    /// <exception cref="ObjectDisposedException">The sink has been disposed.</exception>
    public Task FlushAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            ThrowIfDisposed();
            FlushLocked(cancellationToken);
        }

        return Task.CompletedTask;
    }

    /// <summary>Gets the segment files this directory currently holds, in read order.</summary>
    /// <returns>Absolute paths sorted by name, which is chronological because segment numbers increase.</returns>
    public IReadOnlyList<string> GetSegmentPaths() => EnumerateSegments(DirectoryPath, BaseName);

    /// <summary>Flushes any buffered records and releases the sink.</summary>
    /// <remarks>
    /// A failure while flushing is swallowed: dispose runs on the way out of a run, often inside a <c>finally</c>,
    /// and losing the last batch of notes must not replace the exception that ended the run.
    /// </remarks>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;

            // A disposal-time write failure must not mask the reason the run is unwinding.
#pragma warning disable CA1031
            try
            {
                FlushLocked(CancellationToken.None);
            }
            catch (Exception)
            {
                // Intentionally swallowed; the buffered batch is dropped rather than escaping from Dispose.
            }
#pragma warning restore CA1031

            _disposed = true;
        }
    }

    /// <summary>Lists the segment files belonging to a base name inside a directory, in read order.</summary>
    /// <param name="directoryPath">The directory to scan.</param>
    /// <param name="baseName">The file-name stem; <c>null</c> or empty uses <see cref="DefaultBaseName"/>.</param>
    /// <returns>Absolute paths sorted ordinally by file name; empty when the directory does not exist.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="directoryPath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="directoryPath"/> is empty or white space.</exception>
    public static IReadOnlyList<string> EnumerateSegments(string directoryPath, string? baseName = null)
    {
        Guard.NotNullOrWhiteSpace(directoryPath);
        string full = Path.GetFullPath(directoryPath);
        if (!Directory.Exists(full)) return new List<string>();

        string trimmedBaseName = baseName is null ? string.Empty : baseName.Trim();
        string stem = trimmedBaseName.Length == 0 ? DefaultBaseName : trimmedBaseName;
        var paths = new List<string>(Directory.GetFiles(full, stem + ".*" + SegmentExtension));
        paths.Sort(StringComparer.Ordinal);
        return paths;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(JsonLinesProposalProvenanceSink));
    }

    private void FlushLocked(CancellationToken cancellationToken)
    {
        if (_buffer.Count == 0) return;
        cancellationToken.ThrowIfCancellationRequested();

        var builder = new StringBuilder();
        foreach (string line in _buffer)
        {
            builder.Append(line).Append('\n');
        }

        byte[] payload = _encoding.GetBytes(builder.ToString());
        string tempPath = Path.Combine(
            DirectoryPath,
            "." + BaseName + "." + Guid.NewGuid().ToString("N") + ".tmp");

        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(payload, 0, payload.Length);
                stream.Flush(flushToDisk: true);
            }

            string segmentPath = ReserveSegmentPath();
            File.Move(tempPath, segmentPath);
            _recordsWritten += _buffer.Count;
            _segmentsWritten++;
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
            }

            _buffer.Clear();
            _bufferedBytes = 0L;
        }
    }

    private string ReserveSegmentPath()
    {
        // A concurrent writer in the same directory can claim the number this instance was about to use, so skip
        // forward until an unused name is found rather than overwriting somebody else's finished segment.
        while (true)
        {
            string candidate = Path.Combine(DirectoryPath, BuildSegmentName(_nextSegmentIndex));
            _nextSegmentIndex++;
            if (!File.Exists(candidate)) return candidate;
        }
    }

    private string BuildSegmentName(int index) =>
        BaseName + "." + index.ToString("D6", CultureInfo.InvariantCulture) + SegmentExtension;

    private int FindNextSegmentIndex()
    {
        int highest = -1;
        foreach (string path in EnumerateSegments(DirectoryPath, BaseName))
        {
            string name = Path.GetFileNameWithoutExtension(path);
            int separator = name.LastIndexOf('.');
            if (separator < 0 || separator == name.Length - 1) continue;
            string suffix = name.Substring(separator + 1);
            if (int.TryParse(suffix, NumberStyles.None, CultureInfo.InvariantCulture, out int index) && index > highest)
            {
                highest = index;
            }
        }

        return highest + 1;
    }
}
