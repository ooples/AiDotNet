using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution.Programs.ArtifactStore;

/// <summary>An artifact store that keeps small content in a per-genome index and large content in its own file.</summary>
/// <remarks>
/// <para>
/// Each genome gets one directory under the store root containing <c>index.json</c> and, for artifacts larger than
/// <see cref="ProgramArtifactStoreOptions.InlineSizeThresholdBytes"/>, one file per artifact. That split is the
/// reference OpenEvolve behaviour (<c>openevolve/database.py</c> <c>store_artifacts</c>): content at or below the
/// threshold is serialized into the record and anything larger is written to a per-program directory. Both tiers
/// are durable here, so retrieval by genome identifier keeps working after the run that produced them has exited -
/// which is the whole point, because that is when the output is read.
/// </para>
/// <para>
/// Three things are deliberately stronger than the reference. Writes are atomic: the index is written to a
/// temporary file, flushed to disk, and swapped into place, so a crash leaves the previous index rather than a torn
/// one. Retention operates on the same root the writes go to, whereas upstream writes artifacts under
/// <c>db_path/artifacts</c> - or the working directory when <c>db_path</c> is unset, which is the default - while
/// its cleanup pass looks under <c>&lt;checkpoint&gt;/artifacts</c>, so by default nothing it wrote is ever expired.
/// And every input is bounded: names are sanitized to a safe file name, oversized content is truncated and flagged,
/// and a genome that would exceed its artifact count or byte budget is rejected rather than silently trimmed.
/// </para>
/// <para>
/// Artifact content is untrusted. Nothing here executes, interprets, or renders it; it is written as opaque bytes
/// under a sanitized name that cannot escape the store root. All operations are serialized through one lock, so a
/// single instance may be shared by concurrent callers, and the asynchronous signatures complete synchronously.
/// </para>
/// <para><b>For Beginners:</b> Create one with a directory, hand it to whatever scores your candidate programs, and
/// everything they print or write is filed under the program's fingerprint. After the run, call
/// <c>GetAsync(genomeId)</c> to read the output of any candidate the run recorded, or <c>ListAsync</c> first to see
/// what is there without loading it. Call <c>PurgeAsync(DateTimeOffset.UtcNow)</c> occasionally to delete anything
/// older than the retention period so a long-running search does not fill the disk.</para>
/// </remarks>
public sealed class FileSystemProgramArtifactStore : IProgramArtifactStore
{
    /// <summary>The name of the per-genome index document.</summary>
    public const string IndexFileName = "index.json";

    private const int IndexSchemaVersion = 1;
    private const int MaxDirectoryNameStemLength = 32;

    private static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false);

    private readonly object _gate = new();
    private readonly string _root;
    private readonly ProgramArtifactStoreOptions _options;

    /// <summary>Initializes a file-backed artifact store, creating the root directory when it does not exist.</summary>
    /// <param name="rootDirectory">The directory that holds one subdirectory per genome.</param>
    /// <param name="options">The limits and retention rules; <c>null</c> uses the reference-matching defaults.</param>
    /// <exception cref="ArgumentNullException"><paramref name="rootDirectory"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="rootDirectory"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A limit in <paramref name="options"/> is out of range.</exception>
    public FileSystemProgramArtifactStore(string rootDirectory, ProgramArtifactStoreOptions? options = null)
    {
        Guard.NotNullOrWhiteSpace(rootDirectory);
        ProgramArtifactStoreOptions effective = options is null ? new ProgramArtifactStoreOptions() : options.Clone();
        effective.Validate();
        _options = effective;
        _root = Path.GetFullPath(rootDirectory);
        Directory.CreateDirectory(_root);
    }

    /// <summary>Gets the absolute path of the directory that holds one subdirectory per genome.</summary>
    public string RootDirectory => _root;

    /// <summary>Gets an independent copy of the limits this store was validated with.</summary>
    /// <returns>A copy that a caller may mutate without affecting this instance.</returns>
    public ProgramArtifactStoreOptions GetOptions() => _options.Clone();

    /// <inheritdoc/>
    public Task<IReadOnlyList<ProgramArtifactDescriptor>> StoreAsync(
        string genomeId,
        IEnumerable<ProgramArtifact> artifacts,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNull(artifacts);
        cancellationToken.ThrowIfCancellationRequested();
        ProgramArtifact[] incoming = artifacts.ToArray();
        if (incoming.Any(item => item is null))
            throw new ArgumentException("Artifacts cannot contain null entries.", nameof(artifacts));

        string trimmedId = genomeId.Trim();
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string directory = GetGenomeDirectory(trimmedId);
            ArtifactIndex index = ReadIndex(directory) ?? new ArtifactIndex { GenomeId = trimmedId };
            index.GenomeId = trimmedId;
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var staged = new List<KeyValuePair<ArtifactEntry, ProgramArtifact>>();
            var removedFiles = new List<string>();

            foreach (ProgramArtifact artifact in incoming)
            {
                ProgramArtifact bounded = artifact.Truncate(_options.MaxArtifactBytes);
                RemoveEntry(index, bounded.Name, removedFiles);
                staged.RemoveAll(pair => string.Equals(pair.Key.Name, bounded.Name, StringComparison.Ordinal));
                bool onDisk = bounded.ByteLength > _options.InlineSizeThresholdBytes;
                var entry = new ArtifactEntry
                {
                    Name = bounded.Name,
                    ByteLength = bounded.ByteLength,
                    IsText = bounded.IsText,
                    IsTruncated = bounded.IsTruncated,
                    OnDisk = onDisk,
                    StoredAtUtc = now,
                    FileName = onDisk ? BuildFileName(index.NextFileOrdinal++, bounded.Name) : null,
                    InlineContent = onDisk ? null : Convert.ToBase64String(bounded.GetContentBuffer())
                };
                index.Artifacts.Add(entry);
                staged.Add(new KeyValuePair<ArtifactEntry, ProgramArtifact>(entry, bounded));
            }

            EnforceBudgets(index, trimmedId);
            index.Artifacts.Sort((left, right) => string.CompareOrdinal(left.Name, right.Name));
            index.SchemaVersion = IndexSchemaVersion;
            index.StoredAtUtc = now;

            Directory.CreateDirectory(directory);
            foreach (KeyValuePair<ArtifactEntry, ProgramArtifact> pair in staged)
            {
                string? fileName = pair.Key.FileName;
                if (pair.Key.OnDisk && fileName is not null && fileName.Length > 0)
                {
                    WriteAtomic(Path.Combine(directory, fileName), pair.Value.GetContentBuffer(), cancellationToken);
                }
            }

            WriteIndex(directory, index, cancellationToken);
            foreach (string fileName in removedFiles) TryDeleteFile(Path.Combine(directory, fileName));
            ProgramArtifactDescriptor[] written = staged
                .OrderBy(pair => pair.Key.Name, StringComparer.Ordinal)
                .Select(pair => ToDescriptor(trimmedId, pair.Key))
                .ToArray();
            return Task.FromResult<IReadOnlyList<ProgramArtifactDescriptor>>(Array.AsReadOnly(written));
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ProgramArtifact>> GetAsync(string genomeId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            string directory = GetGenomeDirectory(genomeId.Trim());
            ArtifactIndex? index = ReadIndex(directory);
            if (index is null) return Task.FromResult<IReadOnlyList<ProgramArtifact>>(Array.Empty<ProgramArtifact>());
            var loaded = new List<ProgramArtifact>();
            foreach (ArtifactEntry entry in index.Artifacts.OrderBy(item => item.Name, StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                ProgramArtifact? artifact = Load(directory, entry);
                if (artifact is not null) loaded.Add(artifact);
            }

            return Task.FromResult<IReadOnlyList<ProgramArtifact>>(Array.AsReadOnly(loaded.ToArray()));
        }
    }

    /// <inheritdoc/>
    public Task<ProgramArtifact?> GetAsync(string genomeId, string name, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNullOrWhiteSpace(name);
        cancellationToken.ThrowIfCancellationRequested();
        string trimmedName = name.Trim();
        lock (_gate)
        {
            string directory = GetGenomeDirectory(genomeId.Trim());
            ArtifactIndex? index = ReadIndex(directory);
            ArtifactEntry? entry = index?.Artifacts
                .FirstOrDefault(item => string.Equals(item.Name, trimmedName, StringComparison.Ordinal));
            return Task.FromResult<ProgramArtifact?>(entry is null ? null : Load(directory, entry));
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ProgramArtifactDescriptor>> ListAsync(string genomeId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        cancellationToken.ThrowIfCancellationRequested();
        string trimmedId = genomeId.Trim();
        lock (_gate)
        {
            ArtifactIndex? index = ReadIndex(GetGenomeDirectory(trimmedId));
            if (index is null)
                return Task.FromResult<IReadOnlyList<ProgramArtifactDescriptor>>(Array.Empty<ProgramArtifactDescriptor>());
            ProgramArtifactDescriptor[] descriptors = index.Artifacts
                .OrderBy(entry => entry.Name, StringComparer.Ordinal)
                .Select(entry => ToDescriptor(trimmedId, entry))
                .ToArray();
            return Task.FromResult<IReadOnlyList<ProgramArtifactDescriptor>>(Array.AsReadOnly(descriptors));
        }
    }

    /// <inheritdoc/>
    public Task<bool> RemoveAsync(string genomeId, CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            string directory = GetGenomeDirectory(genomeId.Trim());
            if (!Directory.Exists(directory)) return Task.FromResult(false);
            return Task.FromResult(TryDeleteDirectory(directory));
        }
    }

    /// <inheritdoc/>
    public Task<int> PurgeAsync(DateTimeOffset utcNow, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        lock (_gate)
        {
            if (!Directory.Exists(_root)) return Task.FromResult(0);
            var surviving = new List<KeyValuePair<string, DateTimeOffset>>();
            int removed = 0;
            foreach (string directory in Directory.EnumerateDirectories(_root).OrderBy(value => value, StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                DateTimeOffset storedAt = ReadIndex(directory)?.StoredAtUtc
                    ?? new DateTimeOffset(Directory.GetLastWriteTimeUtc(directory), TimeSpan.Zero);
                if (_options.RetentionPeriod > TimeSpan.Zero && utcNow - storedAt > _options.RetentionPeriod)
                {
                    if (TryDeleteDirectory(directory)) removed++;
                    continue;
                }

                surviving.Add(new KeyValuePair<string, DateTimeOffset>(directory, storedAt));
            }

            if (_options.MaxRetainedGenomes <= 0 || surviving.Count <= _options.MaxRetainedGenomes)
                return Task.FromResult(removed);

            int excess = surviving.Count - _options.MaxRetainedGenomes;
            foreach (KeyValuePair<string, DateTimeOffset> candidate in surviving
                .OrderBy(pair => pair.Value)
                .ThenBy(pair => pair.Key, StringComparer.Ordinal)
                .Take(excess))
            {
                if (TryDeleteDirectory(candidate.Key)) removed++;
            }

            return Task.FromResult(removed);
        }
    }

    /// <summary>Computes the directory this store uses for one genome, whether or not it exists.</summary>
    /// <param name="genomeId">The canonical genome identifier.</param>
    /// <returns>An absolute path under <see cref="RootDirectory"/>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="genomeId"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="genomeId"/> is empty or white space.</exception>
    /// <remarks>
    /// The name is a sanitized prefix of the identifier joined to a hash of the full identifier, so it is always a
    /// valid file name, can never traverse out of the root, and never collides for two different identifiers.
    /// </remarks>
    public string GetGenomeDirectory(string genomeId)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        string trimmed = genomeId.Trim();
        string stem = Sanitize(trimmed, MaxDirectoryNameStemLength);
        if (stem.Length == 0) stem = "genome";
        return Path.Combine(_root, stem + "_" + EvolutionHash.Compute(trimmed).Substring(0, 16));
    }

    private static void RemoveEntry(ArtifactIndex index, string name, List<string> removedFiles)
    {
        ArtifactEntry? existing = index.Artifacts
            .FirstOrDefault(entry => string.Equals(entry.Name, name, StringComparison.Ordinal));
        if (existing is null) return;
        // string.IsNullOrEmpty carries no null-state annotation on .NET Framework, so narrow explicitly.
        string? fileName = existing.FileName;
        if (fileName is not null && fileName.Length > 0) removedFiles.Add(fileName);
        index.Artifacts.Remove(existing);
    }

    private void EnforceBudgets(ArtifactIndex index, string genomeId)
    {
        if (index.Artifacts.Count > _options.MaxArtifactsPerGenome)
        {
            throw new InvalidOperationException(
                $"Genome '{genomeId}' would hold {index.Artifacts.Count} artifacts, " +
                $"which exceeds the configured limit of {_options.MaxArtifactsPerGenome}.");
        }

        long total = index.Artifacts.Sum(entry => (long)entry.ByteLength);
        if (total > _options.MaxTotalBytesPerGenome)
        {
            throw new InvalidOperationException(
                $"Genome '{genomeId}' would hold {total.ToString(CultureInfo.InvariantCulture)} artifact bytes, " +
                $"which exceeds the configured limit of {_options.MaxTotalBytesPerGenome.ToString(CultureInfo.InvariantCulture)}.");
        }
    }

    private static ProgramArtifactDescriptor ToDescriptor(string genomeId, ArtifactEntry entry) => new(
        genomeId,
        entry.Name,
        entry.ByteLength,
        entry.IsText,
        entry.IsTruncated,
        entry.OnDisk ? ProgramArtifactTier.OnDisk : ProgramArtifactTier.Inline,
        entry.StoredAtUtc);

    private static ProgramArtifact? Load(string directory, ArtifactEntry entry)
    {
        if (entry.OnDisk)
        {
            string? fileName = entry.FileName;
            if (fileName is null || fileName.Length == 0) return null;
            string path = Path.Combine(directory, fileName);
            if (!File.Exists(path)) return null;
            return ProgramArtifact.FromBuffer(entry.Name, File.ReadAllBytes(path), entry.IsText, entry.IsTruncated);
        }

        if (entry.InlineContent is null) return null;
        try
        {
            return ProgramArtifact.FromBuffer(entry.Name, Convert.FromBase64String(entry.InlineContent),
                entry.IsText, entry.IsTruncated);
        }
        catch (FormatException)
        {
            return null;
        }
    }

    private static ArtifactIndex? ReadIndex(string directory)
    {
        string path = Path.Combine(directory, IndexFileName);
        if (!File.Exists(path)) return null;
        try
        {
            string json = File.ReadAllText(path, Encoding.UTF8);
            ArtifactIndex? index = JsonConvert.DeserializeObject<ArtifactIndex>(json);
            return index is null || index.SchemaVersion > IndexSchemaVersion ? null : index;
        }
        catch (JsonException)
        {
            return null;
        }
        catch (IOException)
        {
            return null;
        }
    }

    private static void WriteIndex(string directory, ArtifactIndex index, CancellationToken cancellationToken)
    {
        string json = JsonConvert.SerializeObject(index, Formatting.Indented);
        WriteAtomic(Path.Combine(directory, IndexFileName), Utf8.GetBytes(json), cancellationToken);
    }

    private static void WriteAtomic(string path, byte[] payload, CancellationToken cancellationToken)
    {
        string directory = Path.GetDirectoryName(path) ?? ".";
        string tempPath = Path.Combine(directory, "." + Path.GetFileName(path) + "." + Guid.NewGuid().ToString("N") + ".tmp");
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                cancellationToken.ThrowIfCancellationRequested();
                stream.Write(payload, 0, payload.Length);
                stream.Flush(flushToDisk: true);
            }

            cancellationToken.ThrowIfCancellationRequested();
            if (File.Exists(path)) File.Replace(tempPath, path, destinationBackupFileName: null, ignoreMetadataErrors: true);
            else File.Move(tempPath, path);
        }
        finally
        {
            TryDeleteFile(tempPath);
        }
    }

    private static string BuildFileName(int ordinal, string artifactName)
    {
        string stem = Sanitize(artifactName, ProgramArtifact.MaxNameLength);
        if (stem.Length == 0) stem = "artifact";
        return ordinal.ToString("D4", CultureInfo.InvariantCulture) + "_" + stem + ".bin";
    }

    private static string Sanitize(string value, int maxLength)
    {
        var builder = new StringBuilder(Math.Min(value.Length, maxLength));
        foreach (char character in value)
        {
            if (builder.Length >= maxLength) break;
            bool keep = (character >= '0' && character <= '9')
                || (character >= 'a' && character <= 'z')
                || (character >= 'A' && character <= 'Z')
                || character == '.' || character == '_' || character == '-';
            if (keep) builder.Append(character);
        }

        return builder.ToString().TrimStart('.');
    }

    private static void TryDeleteFile(string path)
    {
        if (!File.Exists(path)) return;
        try { File.Delete(path); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static bool TryDeleteDirectory(string path)
    {
        try
        {
            Directory.Delete(path, recursive: true);
            return true;
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    /// <summary>Serialization shape of one genome's artifact index.</summary>
    private sealed class ArtifactIndex
    {
        /// <summary>Gets or sets the index schema version.</summary>
        public int SchemaVersion { get; set; } = IndexSchemaVersion;
        /// <summary>Gets or sets the genome the artifacts belong to.</summary>
        public string GenomeId { get; set; } = string.Empty;
        /// <summary>Gets or sets when the index was last written.</summary>
        public DateTimeOffset StoredAtUtc { get; set; }
        /// <summary>Gets or sets the next ordinal used to name an on-disk artifact file.</summary>
        public int NextFileOrdinal { get; set; }
        /// <summary>Gets the stored artifact entries.</summary>
        public List<ArtifactEntry> Artifacts { get; } = new();
    }

    /// <summary>Serialization shape of one stored artifact.</summary>
    private sealed class ArtifactEntry
    {
        /// <summary>Gets or sets the artifact label.</summary>
        public string Name { get; set; } = string.Empty;
        /// <summary>Gets or sets the stored size in bytes.</summary>
        public int ByteLength { get; set; }
        /// <summary>Gets or sets whether the content was supplied as text.</summary>
        public bool IsText { get; set; }
        /// <summary>Gets or sets whether the content was cut to fit the configured limit.</summary>
        public bool IsTruncated { get; set; }
        /// <summary>Gets or sets whether the bytes live in their own file.</summary>
        public bool OnDisk { get; set; }
        /// <summary>Gets or sets when the artifact was written.</summary>
        public DateTimeOffset StoredAtUtc { get; set; }
        /// <summary>Gets or sets the file holding the bytes, when the artifact is on disk.</summary>
        public string? FileName { get; set; }
        /// <summary>Gets or sets the base64 content, when the artifact is inline.</summary>
        public string? InlineContent { get; set; }
    }
}
