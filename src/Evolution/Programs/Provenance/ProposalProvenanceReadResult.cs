using System.Collections.ObjectModel;
using System.Globalization;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Provenance;

/// <summary>What a read of a provenance stream produced, including what it could not use.</summary>
/// <remarks>
/// <para>
/// A stream written by a process that was killed mid-batch, or copied while a run was still going, can end in a
/// partial line. Refusing the whole file over its last line would throw away every complete record before it, so
/// the reader keeps what it can parse and reports the rest here: <see cref="MalformedLineCount"/> counts the lines
/// it skipped and <see cref="HasIncompleteTail"/> says whether the final line looked cut off rather than corrupt.
/// A caller that needs certainty can treat either as fatal; a caller doing analysis can carry on with a known
/// gap.
/// </para>
/// <para><b>For Beginners:</b> The result of loading the notes back from disk: the notes themselves, which files
/// they came from, and an honest count of any lines that could not be read — usually just the last one, when the
/// program was stopped while writing.</para>
/// </remarks>
public sealed class ProposalProvenanceReadResult
{
    private readonly ProposalProvenanceRecord[] _records;
    private readonly string[] _files;

    /// <summary>Initializes a read result.</summary>
    /// <param name="records">The records that parsed, in file and line order.</param>
    /// <param name="files">The files that were read, in read order.</param>
    /// <param name="malformedLineCount">How many non-empty lines could not be parsed.</param>
    /// <param name="hasIncompleteTail">Whether the last line of the last file was unterminated and unparseable.</param>
    /// <param name="bytesRead">How many bytes were read in total.</param>
    /// <exception cref="ArgumentNullException"><paramref name="records"/> or <paramref name="files"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A collection contains a <c>null</c> entry.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="malformedLineCount"/> or <paramref name="bytesRead"/> is negative.</exception>
    public ProposalProvenanceReadResult(
        IReadOnlyList<ProposalProvenanceRecord> records,
        IReadOnlyList<string> files,
        int malformedLineCount,
        bool hasIncompleteTail,
        long bytesRead)
    {
        Guard.NotNull(records);
        Guard.NotNull(files);
        Guard.NonNegative(malformedLineCount);
        if (bytesRead < 0L)
        {
            throw new ArgumentOutOfRangeException(nameof(bytesRead), bytesRead, "Value cannot be negative.");
        }

        var recordCopy = new ProposalProvenanceRecord[records.Count];
        for (int index = 0; index < records.Count; index++)
        {
            ProposalProvenanceRecord record = records[index];
            if (record is null)
            {
                throw new ArgumentException("A read result cannot contain a null record.", nameof(records));
            }

            recordCopy[index] = record;
        }

        var fileCopy = new string[files.Count];
        for (int index = 0; index < files.Count; index++)
        {
            string file = files[index];
            if (file is null)
            {
                throw new ArgumentException("A read result cannot contain a null file path.", nameof(files));
            }

            fileCopy[index] = file;
        }

        _records = recordCopy;
        _files = fileCopy;
        MalformedLineCount = malformedLineCount;
        HasIncompleteTail = hasIncompleteTail;
        BytesRead = bytesRead;
    }

    /// <summary>Gets the records that parsed, in file and line order.</summary>
    public IReadOnlyList<ProposalProvenanceRecord> Records =>
        new ReadOnlyCollection<ProposalProvenanceRecord>(_records);

    /// <summary>Gets the files that were read, in read order.</summary>
    public IReadOnlyList<string> Files => new ReadOnlyCollection<string>(_files);

    /// <summary>Gets how many non-empty lines could not be parsed and were skipped.</summary>
    public int MalformedLineCount { get; }

    /// <summary>Gets whether the final line was unterminated and unparseable, the signature of an interrupted write.</summary>
    public bool HasIncompleteTail { get; }

    /// <summary>Gets how many bytes were read across every file.</summary>
    public long BytesRead { get; }

    /// <summary>Gets whether every non-empty line parsed.</summary>
    public bool IsComplete => MalformedLineCount == 0 && !HasIncompleteTail;

    /// <summary>Returns the record count and the number of lines that were skipped.</summary>
    /// <returns>A short description carrying no prompt text or program source.</returns>
    public override string ToString() => string.Format(
        CultureInfo.InvariantCulture,
        "ProposalProvenanceReadResult({0} records, {1} files, {2} malformed, incompleteTail={3})",
        _records.Length,
        _files.Length,
        MalformedLineCount,
        HasIncompleteTail);
}
