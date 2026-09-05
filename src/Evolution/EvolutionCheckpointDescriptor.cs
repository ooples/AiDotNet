using AiDotNet.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution;

/// <summary>One entry in a checkpoint directory listing, including snapshots that failed to load.</summary>
/// <remarks>
/// <para>
/// <c>DirectoryEvolutionCheckpointStore.ListCheckpoints</c> returns these newest-first so a caller can inspect what a
/// run left behind without deserializing any payload: the <see cref="Sequence"/> the snapshot was written at, the
/// <see cref="FileName"/> holding it, its <see cref="SizeBytes"/>, whether it is still readable, and, when it is, the
/// <see cref="RunId"/>, <see cref="CompatibilityHash"/>, and best <see cref="Quality"/> recorded on its envelope. A
/// descriptor with <see cref="IsValid"/> set to <see langword="false"/> describes a truncated, corrupt, or foreign
/// file: its sequence and file name are still trustworthy because they come from the name, but every envelope field
/// is <see langword="null"/>.
/// </para>
/// <para><b>For Beginners:</b> Think of a checkpoint directory as a folder of numbered save files. This class is one
/// row of the folder listing: which save it is, how big, whether it can still be opened, and how good the best
/// solution in it was. Use it to answer questions such as "which save should I resume from?" or "did the run keep
/// improving?" without loading the saves themselves, and to notice a save that has gone bad, since a damaged file
/// still appears in the listing rather than silently disappearing.</para>
/// </remarks>
public sealed class EvolutionCheckpointDescriptor
{
    /// <summary>Initializes a listing entry.</summary>
    /// <param name="sequence">The non-negative committed-state sequence taken from the file name.</param>
    /// <param name="fileName">The non-blank file name within the checkpoint directory.</param>
    /// <param name="sizeBytes">The non-negative file size in bytes.</param>
    /// <param name="isValid">Whether the snapshot loaded and verified successfully.</param>
    /// <param name="runId">The run identifier from the envelope, or <see langword="null"/> when it did not load.</param>
    /// <param name="compatibilityHash">The compatibility hash from the envelope, or <see langword="null"/>.</param>
    /// <param name="quality">The best elite quality from the envelope, or <see langword="null"/>.</param>
    /// <param name="qualityDirection">The direction <paramref name="quality"/> is better in, or <see langword="null"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="fileName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="fileName"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sequence"/> or <paramref name="sizeBytes"/> is negative.</exception>
    public EvolutionCheckpointDescriptor(long sequence, string fileName, long sizeBytes, bool isValid,
        string? runId = null, string? compatibilityHash = null, double? quality = null,
        EvolutionOptimizationDirection? qualityDirection = null)
    {
        Guard.NotNullOrWhiteSpace(fileName);
        if (sequence < 0) throw new ArgumentOutOfRangeException(nameof(sequence));
        if (sizeBytes < 0) throw new ArgumentOutOfRangeException(nameof(sizeBytes));
        Sequence = sequence;
        FileName = fileName.Trim();
        SizeBytes = sizeBytes;
        IsValid = isValid;
        RunId = runId;
        CompatibilityHash = compatibilityHash;
        Quality = quality;
        QualityDirection = qualityDirection;
    }

    /// <summary>Gets the committed-state sequence the snapshot was written at.</summary>
    public long Sequence { get; }

    /// <summary>Gets the file name within the checkpoint directory.</summary>
    public string FileName { get; }

    /// <summary>Gets the size of the snapshot file in bytes.</summary>
    public long SizeBytes { get; }

    /// <summary>Gets whether the snapshot loaded and passed both checksums.</summary>
    public bool IsValid { get; }

    /// <summary>Gets the run identifier recorded on the envelope, or <c>null</c> when the snapshot did not load.</summary>
    public string? RunId { get; }

    /// <summary>Gets the compatibility hash recorded on the envelope, or <c>null</c> when the snapshot did not load.</summary>
    public string? CompatibilityHash { get; }

    /// <summary>Gets the best elite quality recorded on the envelope, or <c>null</c> when none was recorded.</summary>
    public double? Quality { get; }

    /// <summary>Gets the direction <see cref="Quality"/> is better in, or <c>null</c> when the snapshot did not load.</summary>
    public EvolutionOptimizationDirection? QualityDirection { get; }
}
