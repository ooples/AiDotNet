using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Outputs;

/// <summary>What one best-program write produced: where the files went and which candidate they describe.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve controller writes its best-program files and logs a line; nothing is returned, so a
/// caller that wants to know where the run put its answer has to reconstruct the path from configuration. A record
/// closes that gap: it carries the absolute paths that were actually written, the genome and quality they describe,
/// and the trigger and ordinal that identify which snapshot this was.
/// </para>
/// <para>
/// Records are immutable and never contain program text, so they are safe to log, keep in a list, or return from a
/// run without bounding them first.
/// </para>
/// <para><b>For Beginners:</b> Every time the best program is saved, you get one of these back saying exactly which
/// two files were written and which candidate they hold. Keep the last one and you always know where the run's
/// answer is on disk.</para>
/// </remarks>
public sealed class ProgramRunOutputRecord
{
    /// <summary>Initializes a record.</summary>
    /// <param name="trigger">What caused the write.</param>
    /// <param name="ordinal">The non-negative snapshot ordinal; zero for a final or manual write.</param>
    /// <param name="genomeId">The canonical identity of the program that was written.</param>
    /// <param name="directoryPath">The absolute directory the files were written into.</param>
    /// <param name="programPath">The absolute path of the program file.</param>
    /// <param name="infoPath">The absolute path of the info document.</param>
    /// <param name="quality">The program's score, or <c>null</c> when it was never scored.</param>
    /// <param name="isSourceTruncated">Whether the program on disk was cut to fit the configured byte limit.</param>
    /// <param name="savedAtUtc">When the write completed.</param>
    /// <exception cref="ArgumentNullException">A path or the genome identity is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A path or the genome identity is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ordinal"/> is negative, or <paramref name="trigger"/> is undefined.</exception>
    public ProgramRunOutputRecord(
        ProgramRunOutputTrigger trigger,
        long ordinal,
        string genomeId,
        string directoryPath,
        string programPath,
        string infoPath,
        double? quality,
        bool isSourceTruncated,
        DateTimeOffset savedAtUtc)
    {
        if (!Enum.IsDefined(typeof(ProgramRunOutputTrigger), trigger)) throw new ArgumentOutOfRangeException(nameof(trigger));
        if (ordinal < 0) throw new ArgumentOutOfRangeException(nameof(ordinal), ordinal, "Value cannot be negative.");
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNullOrWhiteSpace(directoryPath);
        Guard.NotNullOrWhiteSpace(programPath);
        Guard.NotNullOrWhiteSpace(infoPath);
        Trigger = trigger;
        Ordinal = ordinal;
        GenomeId = genomeId.Trim();
        DirectoryPath = directoryPath;
        ProgramPath = programPath;
        InfoPath = infoPath;
        Quality = quality;
        IsSourceTruncated = isSourceTruncated;
        SavedAtUtc = savedAtUtc.ToUniversalTime();
    }

    /// <summary>Gets what caused the write.</summary>
    public ProgramRunOutputTrigger Trigger { get; }

    /// <summary>Gets the snapshot ordinal, which names the checkpoint directory.</summary>
    public long Ordinal { get; }

    /// <summary>Gets the canonical identity of the program that was written.</summary>
    public string GenomeId { get; }

    /// <summary>Gets the absolute directory the files were written into.</summary>
    public string DirectoryPath { get; }

    /// <summary>Gets the absolute path of the program file.</summary>
    public string ProgramPath { get; }

    /// <summary>Gets the absolute path of the info document.</summary>
    public string InfoPath { get; }

    /// <summary>Gets the program's score, or <c>null</c> when it was never scored.</summary>
    public double? Quality { get; }

    /// <summary>Gets whether the program on disk was cut to fit the configured byte limit.</summary>
    public bool IsSourceTruncated { get; }

    /// <summary>Gets when the write completed, in UTC.</summary>
    public DateTimeOffset SavedAtUtc { get; }

    /// <summary>Returns the trigger, the identity prefix, and the program path.</summary>
    /// <returns>A short diagnostic label that never echoes program text.</returns>
    public override string ToString() =>
        "ProgramRunOutputRecord(" + Trigger.ToString() + ", " +
        (GenomeId.Length > 12 ? GenomeId.Substring(0, 12) : GenomeId) + ", " + ProgramPath + ")";
}
