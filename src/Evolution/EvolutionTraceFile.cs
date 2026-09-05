using System.Globalization;
using System.IO.Compression;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Evolution;

/// <summary>Reads and writes evolution trace files in either supported format, compressed or not.</summary>
/// <remarks>
/// <para>
/// This is the counterpart to <see cref="EvolutionTraceObserver{TGenome}"/>: the observer streams a trace out while a
/// run is in flight, and this type loads it back into <see cref="EvolutionTraceRecord"/> objects for tests, analysis,
/// and offline tooling. <see cref="Read(string)"/> detects both the compression and the format from the file's own
/// content - gzip from its two magic bytes, then JSON Lines against JSON from whether the first line is a complete
/// record object - so a trace still loads when it was given an unconventional name. OpenEvolve decides by file
/// extension first and only falls back to sniffing when the extension is unrecognised
/// (trace_export_utils.py:340-363), so a gzipped file named <c>.jsonl</c> fails outright.
/// </para>
/// <para>
/// Every read tolerates a truncated file. A run killed mid-write leaves a partial final line, a partly written JSON
/// array, or an unterminated gzip member; each is reported as <c>IsComplete = false</c> on the result with every
/// complete record still returned. OpenEvolve's loader raises on the first partial line and returns nothing
/// (trace_export_utils.py:204-210). <see cref="ReadRecords(string)"/> streams a JSON Lines trace lazily, holding one
/// record at a time, so a very large trace can be scanned without materialising it.
/// </para>
/// <para>
/// Numbers are written with round-trip precision and enumerations by name, so a trace stays readable and diffable and
/// survives a renumbering of an enumeration. Absent optional fields are omitted rather than written as nulls, which
/// keeps a trace substantially smaller without losing anything on the way back.
/// </para>
/// <para><b>For Beginners:</b> Use <see cref="Read(string)"/> to load a finished run's diary and
/// <see cref="ReadRecords(string)"/> when the diary is too big to hold in memory and you only want to scan it. Use
/// <see cref="Write(IEnumerable{EvolutionTraceRecord}, string, EvolutionTraceFormat, bool, EvolutionTraceSummary)"/>
/// when you already have records in hand - filtered, merged, or built in a test - and want a trace file out of them.
/// You never have to say whether a file is compressed when reading; it works that out itself.</para>
/// </remarks>
public static class EvolutionTraceFile
{
    private const string RecordsPropertyName = "records";
    private const string SummaryPropertyName = "summary";
    private const char ByteOrderMark = '﻿';
    private static readonly UTF8Encoding TraceEncoding = new(encoderShouldEmitUTF8Identifier: false);

    /// <summary>
    /// Reading settings that keep every value exactly as it was written. Newtonsoft turns any ISO-8601-looking string
    /// into a <see cref="DateTime"/> by default, which converts a UTC trace timestamp into the reader's local zone and
    /// discards its sub-second digits; disabling that keeps a timestamp identical across a round trip.
    /// </summary>
    private static readonly JsonSerializerSettings ReadSettings = new()
    {
        DateParseHandling = DateParseHandling.None,
        FloatParseHandling = FloatParseHandling.Double
    };

    /// <summary>Writes a complete trace file, replacing any existing file at the path.</summary>
    /// <param name="records">The records to write, in the order they should appear.</param>
    /// <param name="path">The non-blank destination path; its parent directory is created if needed.</param>
    /// <param name="format">The on-disk layout to write.</param>
    /// <param name="compress">Whether to gzip the whole file as a single stream.</param>
    /// <param name="summary">An optional summary embedded in a JSON document and written to the sidecar file.</param>
    /// <exception cref="ArgumentNullException"><paramref name="records"/> or <paramref name="path"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty, or a record is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="format"/> is undefined.</exception>
    public static void Write(IEnumerable<EvolutionTraceRecord> records, string path,
        EvolutionTraceFormat format = EvolutionTraceFormat.JsonLines, bool compress = false,
        EvolutionTraceSummary? summary = null)
    {
        Guard.NotNull(records);
        Guard.NotNullOrWhiteSpace(path);
        if (!Enum.IsDefined(typeof(EvolutionTraceFormat), format)) throw new ArgumentOutOfRangeException(nameof(format));

        string fullPath = Path.GetFullPath(path.Trim());
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        using (var file = new FileStream(fullPath, FileMode.Create, FileAccess.Write, FileShare.Read))
        {
            Stream target = compress ? new GZipStream(file, CompressionMode.Compress, leaveOpen: true) : file;
            try
            {
                using var writer = new StreamWriter(target, TraceEncoding, 4096, leaveOpen: true);
                bool first = true;
                if (format == EvolutionTraceFormat.Json) writer.Write(JsonHeader(summary?.RunId));
                foreach (EvolutionTraceRecord record in records)
                {
                    if (record is null) throw new ArgumentException("A trace cannot contain null records.", nameof(records));
                    if (format == EvolutionTraceFormat.Json && !first) writer.Write(",\n");
                    writer.Write(SerializeRecord(record));
                    if (format == EvolutionTraceFormat.JsonLines) writer.Write("\n");
                    first = false;
                }
                if (format == EvolutionTraceFormat.Json) writer.Write(JsonFooter(summary));
                writer.Flush();
            }
            finally
            {
                if (compress) target.Dispose();
            }
        }

        if (summary is not null) WriteSummary(EvolutionOutputLayout.SummaryPathFor(fullPath), summary);
    }

    /// <summary>Reads an entire trace file, detecting its compression and format from its content.</summary>
    /// <param name="path">The non-blank path of an existing trace file.</param>
    /// <returns>The recovered records, whether the file was complete, and the sidecar summary when present.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty or white space.</exception>
    /// <exception cref="FileNotFoundException">No file exists at <paramref name="path"/>.</exception>
    public static EvolutionTraceReadResult Read(string path)
    {
        Guard.NotNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path.Trim());
        if (!File.Exists(fullPath)) throw new FileNotFoundException("The evolution trace file was not found.", fullPath);

        byte[] raw;
        using (var file = new FileStream(fullPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var buffer = new MemoryStream())
        {
            file.CopyTo(buffer);
            raw = buffer.ToArray();
        }
        return BuildResult(raw, fullPath);
    }

    /// <summary>Reads an entire trace file without blocking the calling thread on file input.</summary>
    /// <param name="path">The non-blank path of an existing trace file.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The recovered records, whether the file was complete, and the sidecar summary when present.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty or white space.</exception>
    /// <exception cref="FileNotFoundException">No file exists at <paramref name="path"/>.</exception>
    public static async Task<EvolutionTraceReadResult> ReadAsync(string path,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(path);
        cancellationToken.ThrowIfCancellationRequested();
        string fullPath = Path.GetFullPath(path.Trim());
        if (!File.Exists(fullPath)) throw new FileNotFoundException("The evolution trace file was not found.", fullPath);

        byte[] raw;
        using (var file = new FileStream(fullPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite, 81920,
            useAsync: true))
        using (var buffer = new MemoryStream())
        {
            await file.CopyToAsync(buffer, 81920, cancellationToken).ConfigureAwait(false);
            raw = buffer.ToArray();
        }
        return BuildResult(raw, fullPath);
    }

    /// <summary>Streams a trace file one record at a time, holding a single record in memory.</summary>
    /// <param name="path">The non-blank path of an existing trace file.</param>
    /// <returns>A lazy sequence of records that stops silently where a truncated file ends.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty or white space.</exception>
    /// <exception cref="FileNotFoundException">No file exists at <paramref name="path"/>.</exception>
    /// <remarks>
    /// Enumeration holds the file open, so dispose the enumerator - a <c>foreach</c> loop does that for you - before
    /// deleting or rewriting the file. Use <see cref="Read(string)"/> instead when you also need to know whether the
    /// file was complete or want its sidecar summary. Only <see cref="EvolutionTraceFormat.JsonLines"/> streams a
    /// record at a time; a <see cref="EvolutionTraceFormat.Json"/> document is one JSON value and is necessarily
    /// buffered before its records are yielded, which is the concrete reason to prefer JSON Lines for a long run.
    /// </remarks>
    public static IEnumerable<EvolutionTraceRecord> ReadRecords(string path)
    {
        Guard.NotNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path.Trim());
        if (!File.Exists(fullPath)) throw new FileNotFoundException("The evolution trace file was not found.", fullPath);
        return StreamRecords(fullPath);
    }

    /// <summary>Reads the sidecar summary written beside a trace file.</summary>
    /// <param name="tracePath">The non-blank trace path whose sidecar should be read.</param>
    /// <returns>The summary, or <c>null</c> when no readable sidecar exists.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="tracePath"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="tracePath"/> is empty or white space.</exception>
    public static EvolutionTraceSummary? ReadSummary(string tracePath)
    {
        Guard.NotNullOrWhiteSpace(tracePath);
        string summaryPath = EvolutionOutputLayout.SummaryPathFor(Path.GetFullPath(tracePath.Trim()));
        if (!File.Exists(summaryPath)) return null;
        try
        {
            string json;
            using (var stream = new FileStream(summaryPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
            {
                json = reader.ReadToEnd();
            }
            JObject? document = JsonConvert.DeserializeObject<JObject>(json, ReadSettings);
            return document is null ? null : SummaryFromJson(document);
        }
        catch (Exception exception) when (exception is JsonException or IOException or FormatException or
            ArgumentException or OverflowException or InvalidDataException)
        {
            return null;
        }
    }

    /// <summary>Writes a sidecar summary through a temporary file and a rename, so a reader never sees a partial one.</summary>
    internal static void WriteSummary(string summaryPath, EvolutionTraceSummary summary)
    {
        string fullPath = Path.GetFullPath(summaryPath);
        string directory = Path.GetDirectoryName(fullPath) ?? ".";
        Directory.CreateDirectory(directory);
        string tempPath = Path.Combine(directory,
            string.Format(CultureInfo.InvariantCulture, ".{0}.{1:N}.tmp", Path.GetFileName(fullPath), Guid.NewGuid()));
        byte[] payload = TraceEncoding.GetBytes(SummaryToJson(summary).ToString(Formatting.Indented));
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                stream.Write(payload, 0, payload.Length);
                stream.Flush(flushToDisk: true);
            }
            if (File.Exists(fullPath)) File.Delete(fullPath);
            File.Move(tempPath, fullPath);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                try { File.Delete(tempPath); }
                catch (IOException) { }
            }
        }
    }

    /// <summary>Serializes one record to its compact single-line JSON form.</summary>
    internal static string SerializeRecord(EvolutionTraceRecord record) => ToJson(record).ToString(Formatting.None);

    /// <summary>Returns the opening text of an incrementally written JSON document.</summary>
    internal static string JsonHeader(string? runId)
    {
        var builder = new StringBuilder();
        builder.Append("{\n  \"schemaVersion\": ")
            .Append(EvolutionTraceRecord.CurrentSchemaVersion.ToString(CultureInfo.InvariantCulture))
            .Append(",\n  \"format\": \"").Append(EvolutionTraceFormat.Json.ToString()).Append('"');
        if (runId is not null && !string.IsNullOrWhiteSpace(runId))
            builder.Append(",\n  \"runId\": ").Append(JsonConvert.ToString(runId));
        builder.Append(",\n  \"").Append(RecordsPropertyName).Append("\": [\n");
        return builder.ToString();
    }

    /// <summary>Returns the closing text of an incrementally written JSON document.</summary>
    internal static string JsonFooter(EvolutionTraceSummary? summary)
    {
        var builder = new StringBuilder("\n  ]");
        if (summary is not null)
        {
            builder.Append(",\n  \"").Append(SummaryPropertyName).Append("\": ")
                .Append(SummaryToJson(summary).ToString(Formatting.None));
        }
        builder.Append("\n}\n");
        return builder.ToString();
    }

    /// <summary>Converts a record to its JSON object form, omitting absent optional fields.</summary>
    internal static JObject ToJson(EvolutionTraceRecord record)
    {
        Guard.NotNull(record);
        var json = new JObject
        {
            ["schemaVersion"] = record.SchemaVersion,
            ["sequence"] = record.Sequence,
            ["evaluationId"] = record.EvaluationId,
            ["genomeId"] = record.GenomeId,
            ["status"] = record.Status.ToString(),
            ["direction"] = record.Direction.ToString(),
            ["island"] = record.Island,
            ["generation"] = record.Generation,
            ["cacheStatus"] = record.CacheStatus.ToString(),
            ["recordedAtUtc"] = record.RecordedAtUtc.ToString("o", CultureInfo.InvariantCulture),
            ["taskVersionHash"] = record.TaskVersionHash,
            ["evaluatorVersionHash"] = record.EvaluatorVersionHash,
            ["configurationHash"] = record.ConfigurationHash,
            ["attemptCount"] = record.AttemptCount,
            ["costUnits"] = record.CostUnits,
            ["elapsedTicks"] = record.Elapsed.Ticks,
            ["isImprovement"] = record.IsImprovement
        };

        if (record.Quality.HasValue) json["quality"] = record.Quality.Value;
        if (record.InsertionResult.HasValue) json["insertionResult"] = record.InsertionResult.Value.ToString();
        if (record.Cell is not null) json["cell"] = record.Cell;
        if (record.Descriptors.Count > 0) json["descriptors"] = ValuesToJson(record.Descriptors);
        if (record.Metrics.Count > 0) json["metrics"] = ValuesToJson(record.Metrics);
        if (record.ParentGenomeId is not null) json["parentGenomeId"] = record.ParentGenomeId;
        if (record.ParentQuality.HasValue) json["parentQuality"] = record.ParentQuality.Value;
        if (record.QualityDelta.HasValue) json["qualityDelta"] = record.QualityDelta.Value;
        if (record.MetricDeltas.Count > 0) json["metricDeltas"] = ValuesToJson(record.MetricDeltas);
        if (record.ParentIds.Count > 0) json["parentIds"] = ToJsonArray(record.ParentIds);
        if (record.InspirationIds.Count > 0) json["inspirationIds"] = ToJsonArray(record.InspirationIds);
        if (record.VariationOperatorId is not null) json["variationOperatorId"] = record.VariationOperatorId;
        if (record.RefinerId is not null) json["refinerId"] = record.RefinerId;
        if (record.SeedStream != 0) json["seedStream"] = record.SeedStream.ToString(CultureInfo.InvariantCulture);
        if (record.RejectedStage.HasValue) json["rejectedStage"] = record.RejectedStage.Value;
        if (record.Diagnostics.Count > 0)
        {
            var diagnostics = new JArray();
            foreach (EvolutionDiagnostic diagnostic in record.Diagnostics) diagnostics.Add(DiagnosticToJson(diagnostic));
            json["diagnostics"] = diagnostics;
        }
        return json;
    }

    /// <summary>Rebuilds a record from its JSON object form.</summary>
    /// <exception cref="InvalidDataException">A required field is missing or malformed.</exception>
    internal static EvolutionTraceRecord FromJson(JObject json)
    {
        Guard.NotNull(json);
        try
        {
            return new EvolutionTraceRecord(
                RequireLong(json, "sequence"),
                RequireLong(json, "evaluationId"),
                RequireString(json, "genomeId"),
                RequireEnum<EvolutionEvaluationStatus>(json, "status"),
                RequireEnum<EvolutionOptimizationDirection>(json, "direction"),
                (int)RequireLong(json, "island"),
                RequireLong(json, "generation"),
                RequireEnum<EvolutionCacheStatus>(json, "cacheStatus"),
                ParseTimestamp(RequireString(json, "recordedAtUtc")),
                RequireString(json, "taskVersionHash"),
                RequireString(json, "evaluatorVersionHash"),
                RequireString(json, "configurationHash"))
            {
                Quality = OptionalDouble(json, "quality"),
                InsertionResult = OptionalEnum<EvolutionArchiveInsertionResult>(json, "insertionResult"),
                Cell = OptionalString(json, "cell"),
                Descriptors = ValuesFromJson(json, "descriptors"),
                Metrics = ValuesFromJson(json, "metrics"),
                ParentGenomeId = OptionalString(json, "parentGenomeId"),
                ParentQuality = OptionalDouble(json, "parentQuality"),
                QualityDelta = OptionalDouble(json, "qualityDelta"),
                IsImprovement = Flag(json, "isImprovement"),
                MetricDeltas = ValuesFromJson(json, "metricDeltas"),
                ParentIds = IdentitiesFromJson(json, "parentIds"),
                InspirationIds = IdentitiesFromJson(json, "inspirationIds"),
                VariationOperatorId = OptionalString(json, "variationOperatorId"),
                RefinerId = OptionalString(json, "refinerId"),
                SeedStream = OptionalSeedStream(json),
                AttemptCount = (int)OptionalLong(json, "attemptCount"),
                CostUnits = OptionalDouble(json, "costUnits") ?? 0,
                Elapsed = TimeSpan.FromTicks(OptionalLong(json, "elapsedTicks")),
                RejectedStage = OptionalInt(json, "rejectedStage"),
                Diagnostics = DiagnosticsFromJson(json)
            };
        }
        catch (Exception exception) when (exception is ArgumentException or FormatException or OverflowException or
            InvalidCastException or JsonException)
        {
            throw new InvalidDataException("The evolution trace record is not valid.", exception);
        }
    }

    /// <summary>Converts a summary to its JSON object form.</summary>
    internal static JObject SummaryToJson(EvolutionTraceSummary summary)
    {
        Guard.NotNull(summary);
        var statusCounts = new JObject();
        foreach (KeyValuePair<EvolutionEvaluationStatus, long> pair in summary.StatusCounts)
            statusCounts[pair.Key.ToString()] = pair.Value;

        var json = new JObject
        {
            ["schemaVersion"] = summary.SchemaVersion,
            ["runId"] = summary.RunId,
            ["format"] = summary.Format.ToString(),
            ["compressed"] = summary.Compressed,
            ["traceOptions"] = summary.TraceOptions,
            ["recordsWritten"] = summary.RecordsWritten,
            ["recordsDropped"] = summary.RecordsDropped,
            ["bytesWritten"] = summary.BytesWritten,
            ["isTruncated"] = summary.IsTruncated,
            ["isClosed"] = summary.IsClosed,
            ["observerFailures"] = summary.ObserverFailures,
            ["improvementCount"] = summary.ImprovementCount,
            ["deltaSampleCount"] = summary.DeltaSampleCount,
            ["improvementRate"] = summary.ImprovementRate,
            ["cacheHits"] = summary.CacheHits,
            ["cacheHitRate"] = summary.CacheHitRate,
            ["totalCostUnits"] = summary.TotalCostUnits,
            ["totalElapsedTicks"] = summary.TotalElapsed.Ticks,
            ["meanEvaluatorSeconds"] = summary.MeanEvaluatorSeconds,
            ["statusCounts"] = statusCounts,
            ["totalMetricDeltas"] = ValuesToJson(summary.TotalMetricDeltas),
            ["bestMetricDeltas"] = ValuesToJson(summary.BestMetricDeltas),
            ["worstMetricDeltas"] = ValuesToJson(summary.WorstMetricDeltas)
        };
        if (summary.FirstFailureMessage is not null) json["firstFailureMessage"] = summary.FirstFailureMessage;
        if (summary.BestQualityDelta.HasValue) json["bestQualityDelta"] = summary.BestQualityDelta.Value;
        if (summary.WorstQualityDelta.HasValue) json["worstQualityDelta"] = summary.WorstQualityDelta.Value;
        return json;
    }

    /// <summary>Rebuilds a summary from its JSON object form.</summary>
    /// <exception cref="InvalidDataException">A required field is missing or malformed.</exception>
    internal static EvolutionTraceSummary SummaryFromJson(JObject json)
    {
        Guard.NotNull(json);
        var statusCounts = new Dictionary<EvolutionEvaluationStatus, long>();
        if (json["statusCounts"] is JObject counts)
        {
            foreach (KeyValuePair<string, JToken?> pair in counts)
            {
                if (pair.Value is null || !Enum.IsDefined(typeof(EvolutionEvaluationStatus), pair.Key)) continue;
                var status = (EvolutionEvaluationStatus)Enum.Parse(typeof(EvolutionEvaluationStatus), pair.Key);
                statusCounts[status] = pair.Value.Value<long>();
            }
        }

        return new EvolutionTraceSummary(RequireString(json, "runId"),
            RequireEnum<EvolutionTraceFormat>(json, "format"), Flag(json, "compressed"))
        {
            TraceOptions = OptionalString(json, "traceOptions") ?? string.Empty,
            RecordsWritten = OptionalLong(json, "recordsWritten"),
            RecordsDropped = OptionalLong(json, "recordsDropped"),
            BytesWritten = OptionalLong(json, "bytesWritten"),
            IsTruncated = Flag(json, "isTruncated"),
            IsClosed = Flag(json, "isClosed"),
            ObserverFailures = OptionalLong(json, "observerFailures"),
            FirstFailureMessage = OptionalString(json, "firstFailureMessage"),
            ImprovementCount = OptionalLong(json, "improvementCount"),
            DeltaSampleCount = OptionalLong(json, "deltaSampleCount"),
            CacheHits = OptionalLong(json, "cacheHits"),
            BestQualityDelta = OptionalDouble(json, "bestQualityDelta"),
            WorstQualityDelta = OptionalDouble(json, "worstQualityDelta"),
            TotalCostUnits = OptionalDouble(json, "totalCostUnits") ?? 0,
            TotalElapsed = TimeSpan.FromTicks(OptionalLong(json, "totalElapsedTicks")),
            StatusCounts = statusCounts,
            TotalMetricDeltas = ValuesFromJson(json, "totalMetricDeltas"),
            BestMetricDeltas = ValuesFromJson(json, "bestMetricDeltas"),
            WorstMetricDeltas = ValuesFromJson(json, "worstMetricDeltas")
        };
    }

    private static EvolutionTraceReadResult BuildResult(byte[] raw, string fullPath)
    {
        string content = DecodeBytes(raw, out bool compressed, out bool decodeComplete);
        EvolutionTraceFormat format = DetectFormat(content);
        var records = new List<EvolutionTraceRecord>();
        bool parsedCompletely = format == EvolutionTraceFormat.JsonLines
            ? ParseJsonLines(content, records)
            : ParseJsonDocument(content, records);
        return new EvolutionTraceReadResult(records, decodeComplete && parsedCompletely, format, compressed,
            ReadSummary(fullPath));
    }

    private static IEnumerable<EvolutionTraceRecord> StreamRecords(string fullPath)
    {
        using var file = new FileStream(fullPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        int firstByte = file.ReadByte();
        int secondByte = file.ReadByte();
        file.Seek(0, SeekOrigin.Begin);
        bool compressed = firstByte == 0x1F && secondByte == 0x8B;
        Stream decoded = compressed ? new GZipStream(file, CompressionMode.Decompress, leaveOpen: true) : file;
        try
        {
            using var reader = new StreamReader(decoded, Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
            string? firstLine = ReadLineSafely(reader);
            if (firstLine is null) yield break;

            if (!IsRecordLine(firstLine))
            {
                var documentRecords = new List<EvolutionTraceRecord>();
                ParseJsonDocument(firstLine + "\n" + ReadToEndSafely(reader), documentRecords);
                foreach (EvolutionTraceRecord record in documentRecords) yield return record;
                yield break;
            }

            string? line = firstLine;
            while (line is not null)
            {
                if (line.Trim().Length > 0)
                {
                    EvolutionTraceRecord? record = TryParseLine(line);
                    if (record is null) yield break;
                    yield return record;
                }
                line = ReadLineSafely(reader);
            }
        }
        finally
        {
            if (compressed) decoded.Dispose();
        }
    }

    private static EvolutionTraceRecord? TryParseLine(string line)
    {
        try
        {
            JObject? json = JsonConvert.DeserializeObject<JObject>(line, ReadSettings);
            return json is null ? null : FromJson(json);
        }
        catch (Exception exception) when (exception is JsonException or InvalidDataException)
        {
            return null;
        }
    }

    private static string? ReadLineSafely(TextReader reader)
    {
        try { return reader.ReadLine(); }
        catch (InvalidDataException) { return null; }
        catch (IOException) { return null; }
    }

    private static string ReadToEndSafely(TextReader reader)
    {
        try { return reader.ReadToEnd(); }
        catch (InvalidDataException) { return string.Empty; }
        catch (IOException) { return string.Empty; }
    }

    private static string DecodeBytes(byte[] raw, out bool compressed, out bool complete)
    {
        compressed = raw.Length >= 2 && raw[0] == 0x1F && raw[1] == 0x8B;
        complete = true;
        if (!compressed) return StripByteOrderMark(TraceEncoding.GetString(raw));

        using var source = new MemoryStream(raw, writable: false);
        using var gzip = new GZipStream(source, CompressionMode.Decompress);
        using var target = new MemoryStream();
        var chunk = new byte[81920];
        while (true)
        {
            int read;
            try
            {
                read = gzip.Read(chunk, 0, chunk.Length);
            }
            catch (Exception exception) when (exception is InvalidDataException or EndOfStreamException)
            {
                complete = false;
                break;
            }
            if (read <= 0) break;
            target.Write(chunk, 0, read);
        }
        return StripByteOrderMark(TraceEncoding.GetString(target.ToArray()));
    }

    private static string StripByteOrderMark(string value) =>
        value.Length > 0 && value[0] == ByteOrderMark ? value.Substring(1) : value;

    private static EvolutionTraceFormat DetectFormat(string content)
    {
        foreach (string line in EnumerateLines(content))
        {
            if (line.Trim().Length == 0) continue;
            return IsRecordLine(line) ? EvolutionTraceFormat.JsonLines : EvolutionTraceFormat.Json;
        }
        return EvolutionTraceFormat.JsonLines;
    }

    private static bool IsRecordLine(string line)
    {
        string trimmed = line.Trim();
        if (trimmed.Length < 2 || trimmed[0] != '{' || trimmed[trimmed.Length - 1] != '}') return false;
        try
        {
            JObject? json = JsonConvert.DeserializeObject<JObject>(trimmed, ReadSettings);
            return json is not null && json["evaluationId"] is not null;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    private static IEnumerable<string> EnumerateLines(string content)
    {
        using var reader = new StringReader(content);
        while (true)
        {
            string? line = reader.ReadLine();
            if (line is null) yield break;
            yield return line;
        }
    }

    private static bool ParseJsonLines(string content, List<EvolutionTraceRecord> records)
    {
        foreach (string line in EnumerateLines(content))
        {
            if (line.Trim().Length == 0) continue;
            EvolutionTraceRecord? record = TryParseLine(line);
            if (record is null) return false;
            records.Add(record);
        }
        return true;
    }

    private static bool ParseJsonDocument(string content, List<EvolutionTraceRecord> records)
    {
        using var textReader = new StringReader(content);
        using var reader = new JsonTextReader(textReader) { DateParseHandling = DateParseHandling.None };
        try
        {
            while (reader.Read())
            {
                if (reader.TokenType != JsonToken.PropertyName ||
                    !string.Equals(reader.Value as string, RecordsPropertyName, StringComparison.Ordinal))
                {
                    continue;
                }
                if (!reader.Read() || reader.TokenType != JsonToken.StartArray) return false;
                while (reader.Read() && reader.TokenType != JsonToken.EndArray)
                {
                    records.Add(FromJson(JObject.Load(reader)));
                }
                return reader.TokenType == JsonToken.EndArray;
            }
        }
        catch (Exception exception) when (exception is JsonException or InvalidDataException)
        {
            return false;
        }
        return false;
    }

    private static JArray ToJsonArray(IReadOnlyList<string> values)
    {
        var array = new JArray();
        foreach (string value in values) array.Add(value);
        return array;
    }

    private static JObject ValuesToJson(IReadOnlyDictionary<string, double> values)
    {
        var json = new JObject();
        foreach (KeyValuePair<string, double> pair in values) json[pair.Key] = pair.Value;
        return json;
    }

    private static Dictionary<string, double> ValuesFromJson(JObject json, string propertyName)
    {
        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        if (json[propertyName] is not JObject source) return values;
        foreach (KeyValuePair<string, JToken?> pair in source)
            if (pair.Value is not null) values[pair.Key] = pair.Value.Value<double>();
        return values;
    }

    private static List<string> IdentitiesFromJson(JObject json, string propertyName)
    {
        var values = new List<string>();
        if (json[propertyName] is not JArray source) return values;
        foreach (JToken token in source)
        {
            string? value = token.Value<string>();
            if (value is not null) values.Add(value);
        }
        return values;
    }

    private static List<EvolutionDiagnostic> DiagnosticsFromJson(JObject json)
    {
        var diagnostics = new List<EvolutionDiagnostic>();
        if (json["diagnostics"] is not JArray source) return diagnostics;
        foreach (JToken token in source)
        {
            if (token is not JObject element) continue;
            var data = new Dictionary<string, string>(StringComparer.Ordinal);
            if (element["data"] is JObject dataObject)
            {
                foreach (KeyValuePair<string, JToken?> pair in dataObject)
                {
                    string? value = pair.Value?.Value<string>();
                    if (value is not null) data[pair.Key] = value;
                }
            }
            diagnostics.Add(new EvolutionDiagnostic(RequireString(element, "code"),
                element["message"]?.Value<string>() ?? string.Empty, Flag(element, "isRedacted"), data));
        }
        return diagnostics;
    }

    private static JObject DiagnosticToJson(EvolutionDiagnostic diagnostic)
    {
        var json = new JObject
        {
            ["code"] = diagnostic.Code,
            ["message"] = diagnostic.Message
        };
        if (diagnostic.IsRedacted) json["isRedacted"] = true;
        if (diagnostic.Data.Count > 0)
        {
            var data = new JObject();
            foreach (KeyValuePair<string, string> pair in diagnostic.Data) data[pair.Key] = pair.Value;
            json["data"] = data;
        }
        return json;
    }

    private static DateTimeOffset ParseTimestamp(string value) => DateTimeOffset
        .Parse(value, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind | DateTimeStyles.AssumeUniversal)
        .ToUniversalTime();

    private static bool Flag(JObject json, string propertyName)
    {
        JToken? token = json[propertyName];
        return token is not null && token.Type != JTokenType.Null && token.Value<bool>();
    }

    private static string RequireString(JObject json, string propertyName)
    {
        string? value = json[propertyName]?.Value<string>();
        if (value is null || string.IsNullOrWhiteSpace(value))
            throw new InvalidDataException($"The evolution trace field '{propertyName}' is missing.");
        return value;
    }

    private static long RequireLong(JObject json, string propertyName)
    {
        JToken? token = json[propertyName];
        if (token is null) throw new InvalidDataException($"The evolution trace field '{propertyName}' is missing.");
        return token.Value<long>();
    }

    private static TEnum RequireEnum<TEnum>(JObject json, string propertyName) where TEnum : struct
    {
        string value = RequireString(json, propertyName);
        if (!Enum.IsDefined(typeof(TEnum), value))
            throw new InvalidDataException($"The evolution trace field '{propertyName}' holds an unknown value.");
        return (TEnum)Enum.Parse(typeof(TEnum), value);
    }

    private static TEnum? OptionalEnum<TEnum>(JObject json, string propertyName) where TEnum : struct
    {
        string? value = json[propertyName]?.Value<string>();
        if (value is null || string.IsNullOrWhiteSpace(value) || !Enum.IsDefined(typeof(TEnum), value)) return null;
        return (TEnum)Enum.Parse(typeof(TEnum), value);
    }

    private static string? OptionalString(JObject json, string propertyName)
    {
        string? value = json[propertyName]?.Value<string>();
        return value is null || string.IsNullOrWhiteSpace(value) ? null : value;
    }

    private static double? OptionalDouble(JObject json, string propertyName)
    {
        JToken? token = json[propertyName];
        return token is null || token.Type == JTokenType.Null ? null : token.Value<double>();
    }

    private static long OptionalLong(JObject json, string propertyName)
    {
        JToken? token = json[propertyName];
        return token is null || token.Type == JTokenType.Null ? 0L : token.Value<long>();
    }

    private static int? OptionalInt(JObject json, string propertyName)
    {
        JToken? token = json[propertyName];
        return token is null || token.Type == JTokenType.Null ? null : token.Value<int>();
    }

    private static ulong OptionalSeedStream(JObject json)
    {
        JToken? token = json["seedStream"];
        if (token is null || token.Type == JTokenType.Null) return 0UL;
        string? text = token.Value<string>();
        return text is not null && ulong.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out ulong value)
            ? value
            : 0UL;
    }
}
