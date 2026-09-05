using System.Text;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>Reads a recorded provenance stream back and rebuilds the ancestry it describes.</summary>
/// <remarks>
/// <para>
/// This is the post-hoc half of provenance. <see cref="Read(string, long)"/> loads a stream — one JSON Lines file
/// or a whole directory of segments — tolerating a truncated tail rather than refusing the run's entire history
/// over its last partial line. <see cref="BuildLineages"/> then turns the flat record stream into the chains of
/// accepted edits that produced each final program, oldest first, with the prompt and answer that caused every
/// step attached to it.
/// </para>
/// <para>
/// The reconstruction needs no engine, no archive, and no checkpoint: everything it uses is in the stream. That is
/// the point. Months after a run, from a directory of files, it answers "which conversations produced this
/// program, in what order, at what cost", which is what a post-hoc audit needs and what turns a finished search
/// into a set of training trajectories.
/// </para>
/// <para>
/// Records are untrusted input by construction — the text they carry came from a language model and from program
/// output — so nothing read here is executed, expanded, or used to build a path, and every field was bounded and
/// redacted before it was written and is bounded again on the way back in. A stream that names a parent as its own
/// ancestor cannot make the walk loop: visited identities are tracked and a repeat ends the chain.
/// </para>
/// <para><b>For Beginners:</b> Point <see cref="Read(string, long)"/> at the folder your provenance sink wrote to
/// and you get the notes back. Pass those to <see cref="BuildLineages"/> and you get, for each program the search
/// finished with, the whole story of how it got there — which program it started from, and each AI conversation
/// that improved it along the way, in order.</para>
/// </remarks>
public static class ProposalProvenanceReader
{
    /// <summary>The default ceiling on how many bytes one read will consume.</summary>
    public const long DefaultMaxTotalBytes = 256L * 1024L * 1024L;

    /// <summary>Reads a provenance stream from a JSON Lines file or a directory of segment files.</summary>
    /// <param name="path">A <c>.jsonl</c> file, or a directory whose <c>.jsonl</c> files are read in name order.</param>
    /// <param name="maxTotalBytes">
    /// The most bytes this read will consume; reading stops once the budget is exhausted and the result reports
    /// how much was read. Defaults to <see cref="DefaultMaxTotalBytes"/>.
    /// </param>
    /// <returns>The records that parsed, plus a count of the lines that did not.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxTotalBytes"/> is not positive.</exception>
    /// <exception cref="FileNotFoundException">No file or directory exists at <paramref name="path"/>.</exception>
    public static ProposalProvenanceReadResult Read(string path, long maxTotalBytes = DefaultMaxTotalBytes)
    {
        Guard.NotNullOrWhiteSpace(path);
        if (maxTotalBytes <= 0L)
        {
            throw new ArgumentOutOfRangeException(nameof(maxTotalBytes), maxTotalBytes, "Value must be positive.");
        }

        string full = Path.GetFullPath(path);
        List<string> files;
        if (Directory.Exists(full))
        {
            files = new List<string>(Directory.GetFiles(full, "*" + JsonLinesProposalProvenanceSink.SegmentExtension));
            files.Sort(StringComparer.Ordinal);
        }
        else if (File.Exists(full))
        {
            files = new List<string> { full };
        }
        else
        {
            throw new FileNotFoundException("No provenance file or directory was found at the supplied path.", full);
        }

        var records = new List<ProposalProvenanceRecord>();
        var read = new List<string>(files.Count);
        int malformed = 0;
        bool incompleteTail = false;
        long bytesRead = 0L;

        foreach (string file in files)
        {
            if (bytesRead >= maxTotalBytes) break;

            read.Add(file);
            using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var reader = new StreamReader(stream, new UTF8Encoding(false), detectEncodingFromByteOrderMarks: true);

            string? line;
            bool lastLineFailed = false;
            while ((line = reader.ReadLine()) is not null)
            {
                bytesRead += Encoding.UTF8.GetByteCount(line) + 1L;
                if (line.Trim().Length == 0)
                {
                    lastLineFailed = false;
                    continue;
                }

                ProposalProvenanceRecord? record = TryParse(line);
                if (record is null)
                {
                    malformed++;
                    lastLineFailed = true;
                }
                else
                {
                    records.Add(record);
                    lastLineFailed = false;
                }

                if (bytesRead >= maxTotalBytes) break;
            }

            // A stream cut mid-write ends without a newline. Distinguishing that from ordinary corruption tells a
            // caller whether the gap is one interrupted line or a damaged file.
            if (lastLineFailed && !EndsWithNewline(file)) incompleteTail = true;
        }

        return new ProposalProvenanceReadResult(records, read, malformed, incompleteTail, bytesRead);
    }

    /// <summary>Parses one JSON Lines record, returning <c>null</c> when the line is not a usable record.</summary>
    /// <param name="line">One line of a provenance stream.</param>
    /// <returns>The record, or <c>null</c> when the line is empty, malformed, or missing required fields.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="line"/> is <c>null</c>.</exception>
    public static ProposalProvenanceRecord? TryParse(string line)
    {
        Guard.NotNull(line);
        if (line.Trim().Length == 0) return null;

        try
        {
            ProposalProvenanceDocument? document =
                JsonConvert.DeserializeObject<ProposalProvenanceDocument>(line);
            return document?.ToRecord();
        }
        catch (JsonException)
        {
            return null;
        }
        catch (InvalidDataException)
        {
            return null;
        }
        catch (ArgumentException)
        {
            // A document whose bounds are out of range (a negative token count survived an edit, say) is data we
            // cannot trust; skipping the line is preferable to failing the whole read.
            return null;
        }
    }

    /// <summary>Rebuilds the ancestry chains described by a provenance stream.</summary>
    /// <param name="records">The records to reconstruct from, in any order.</param>
    /// <returns>
    /// One lineage per program that no later accepted step built upon, deepest first. Programs whose ancestry is
    /// entirely outside the stream produce no lineage, because there is no recorded step to describe them.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="records"/> is <c>null</c>.</exception>
    public static IReadOnlyList<ProposalProvenanceLineage> BuildLineages(
        IReadOnlyList<ProposalProvenanceRecord> records)
    {
        Guard.NotNull(records);

        // One accepted step per child. When a stream records the same child twice — a resumed run replaying an
        // iteration, two islands converging on identical source — the earlier request is the one that created it.
        var producedBy = new Dictionary<string, ProposalProvenanceRecord>(StringComparer.Ordinal);
        foreach (ProposalProvenanceRecord record in records)
        {
            if (record is null || !record.IsAccepted) continue;
            if (string.Equals(record.ChildGenomeId, record.ParentGenomeId, StringComparison.Ordinal)) continue;

            if (!producedBy.TryGetValue(record.ChildGenomeId, out ProposalProvenanceRecord? existing) ||
                IsEarlier(record, existing))
            {
                producedBy[record.ChildGenomeId] = record;
            }
        }

        if (producedBy.Count == 0) return new List<ProposalProvenanceLineage>();

        var extended = new HashSet<string>(StringComparer.Ordinal);
        foreach (ProposalProvenanceRecord record in producedBy.Values)
        {
            extended.Add(record.ParentGenomeId);
        }

        var lineages = new List<ProposalProvenanceLineage>();
        foreach (string childId in producedBy.Keys)
        {
            if (extended.Contains(childId)) continue;
            lineages.Add(BuildLineage(childId, producedBy));
        }

        lineages.Sort(CompareLineages);
        return lineages;
    }

    /// <summary>Rebuilds the ancestry chain of one program from a provenance stream.</summary>
    /// <param name="genomeId">The canonical identity of the program to trace back from.</param>
    /// <param name="records">The records to reconstruct from, in any order.</param>
    /// <returns>The lineage, whose <c>Depth</c> is zero when the stream records no step that produced the program.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="genomeId"/> or <paramref name="records"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="genomeId"/> is empty or white space.</exception>
    public static ProposalProvenanceLineage BuildLineage(
        string genomeId,
        IReadOnlyList<ProposalProvenanceRecord> records)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNull(records);

        var producedBy = new Dictionary<string, ProposalProvenanceRecord>(StringComparer.Ordinal);
        foreach (ProposalProvenanceRecord record in records)
        {
            if (record is null || !record.IsAccepted) continue;
            if (string.Equals(record.ChildGenomeId, record.ParentGenomeId, StringComparison.Ordinal)) continue;

            if (!producedBy.TryGetValue(record.ChildGenomeId, out ProposalProvenanceRecord? existing) ||
                IsEarlier(record, existing))
            {
                producedBy[record.ChildGenomeId] = record;
            }
        }

        return BuildLineage(genomeId.Trim(), producedBy);
    }

    private static ProposalProvenanceLineage BuildLineage(
        string genomeId,
        Dictionary<string, ProposalProvenanceRecord> producedBy)
    {
        var reversed = new List<ProposalProvenanceRecord>();
        var visited = new HashSet<string>(StringComparer.Ordinal) { genomeId };

        string current = genomeId;
        while (producedBy.TryGetValue(current, out ProposalProvenanceRecord? step))
        {
            reversed.Add(step);
            // A stream that claims a program is its own ancestor must not spin the walk forever.
            if (!visited.Add(step.ParentGenomeId)) break;
            current = step.ParentGenomeId;
        }

        var steps = new List<ProposalProvenanceLineageStep>(reversed.Count);
        for (int index = reversed.Count - 1; index >= 0; index--)
        {
            steps.Add(new ProposalProvenanceLineageStep(reversed.Count - 1 - index, reversed[index]));
        }

        return new ProposalProvenanceLineage(genomeId, steps);
    }

    private static bool IsEarlier(ProposalProvenanceRecord candidate, ProposalProvenanceRecord existing)
    {
        if (candidate.RequestedAtUtc != existing.RequestedAtUtc)
        {
            // A record with no timestamp sorts last, so a timestamped step always wins over an untimed one.
            if (candidate.RequestedAtUtc == default) return false;
            if (existing.RequestedAtUtc == default) return true;
            return candidate.RequestedAtUtc < existing.RequestedAtUtc;
        }

        if (candidate.EvaluationId != existing.EvaluationId) return candidate.EvaluationId < existing.EvaluationId;
        return candidate.AttemptNumber < existing.AttemptNumber;
    }

    private static int CompareLineages(ProposalProvenanceLineage left, ProposalProvenanceLineage right)
    {
        int byDepth = right.Depth.CompareTo(left.Depth);
        return byDepth != 0 ? byDepth : StringComparer.Ordinal.Compare(left.FinalGenomeId, right.FinalGenomeId);
    }

    private static bool EndsWithNewline(string file)
    {
        using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        if (stream.Length == 0L) return true;
        stream.Seek(-1L, SeekOrigin.End);
        int last = stream.ReadByte();
        return last == '\n' || last == '\r';
    }
}
