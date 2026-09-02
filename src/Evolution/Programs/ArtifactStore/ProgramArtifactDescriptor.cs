using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.ArtifactStore;

/// <summary>What a store knows about one artifact without reading its bytes.</summary>
/// <remarks>
/// <para>
/// Listing is the operation that makes an artifact store usable after a run: a run that produced ten thousand
/// evaluations may hold gigabytes of captured output, and a caller triaging it needs to know what exists and how
/// big it is before deciding what to load. A descriptor answers that question at index cost, whereas reading the
/// artifact itself may touch a large file.
/// </para>
/// <para>
/// <see cref="Tier"/> reports which side of the store's size threshold the artifact landed on, so a caller can
/// distinguish an inline read from a file read. <see cref="IsTruncated"/> reports that the store had to cut the
/// content to stay inside its configured limit, which is the difference between a program that printed nothing more
/// and one whose output was discarded.
/// </para>
/// <para><b>For Beginners:</b> This is the label on the box rather than the contents: the name, how large it is,
/// whether it is text, and whether anything was cut off. Use it to decide which outputs are worth loading.</para>
/// </remarks>
public sealed class ProgramArtifactDescriptor
{
    /// <summary>Initializes a descriptor.</summary>
    /// <param name="genomeId">The genome the artifact belongs to.</param>
    /// <param name="name">The artifact's label.</param>
    /// <param name="byteLength">The stored size in bytes.</param>
    /// <param name="isText">Whether the artifact was supplied as text.</param>
    /// <param name="isTruncated">Whether the stored content was cut to fit a limit.</param>
    /// <param name="tier">Where the store keeps the bytes.</param>
    /// <param name="storedAtUtc">When the artifact was written.</param>
    /// <exception cref="ArgumentNullException"><paramref name="genomeId"/> or <paramref name="name"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="genomeId"/> or <paramref name="name"/> is empty or white space.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="byteLength"/> is negative, or <paramref name="tier"/> is undefined.</exception>
    public ProgramArtifactDescriptor(
        string genomeId,
        string name,
        int byteLength,
        bool isText,
        bool isTruncated,
        ProgramArtifactTier tier,
        DateTimeOffset storedAtUtc)
    {
        Guard.NotNullOrWhiteSpace(genomeId);
        Guard.NotNullOrWhiteSpace(name);
        if (byteLength < 0) throw new ArgumentOutOfRangeException(nameof(byteLength), byteLength, "Value cannot be negative.");
        if (!Enum.IsDefined(typeof(ProgramArtifactTier), tier)) throw new ArgumentOutOfRangeException(nameof(tier));
        GenomeId = genomeId.Trim();
        Name = name.Trim();
        ByteLength = byteLength;
        IsText = isText;
        IsTruncated = isTruncated;
        Tier = tier;
        StoredAtUtc = storedAtUtc.ToUniversalTime();
    }

    /// <summary>Gets the genome the artifact belongs to.</summary>
    public string GenomeId { get; }

    /// <summary>Gets the artifact's label.</summary>
    public string Name { get; }

    /// <summary>Gets the stored size in bytes.</summary>
    public int ByteLength { get; }

    /// <summary>Gets whether the artifact was supplied as text.</summary>
    public bool IsText { get; }

    /// <summary>Gets whether the stored content was cut to fit the store's limit.</summary>
    public bool IsTruncated { get; }

    /// <summary>Gets which storage tier holds the bytes.</summary>
    public ProgramArtifactTier Tier { get; }

    /// <summary>Gets when the artifact was written, in UTC.</summary>
    public DateTimeOffset StoredAtUtc { get; }

    /// <summary>Returns the name, size, and tier, never the content.</summary>
    /// <returns>A short diagnostic label that is always safe to log.</returns>
    public override string ToString() =>
        "ProgramArtifactDescriptor(" + Name + ", " + ByteLength.ToString(CultureInfo.InvariantCulture) +
        " bytes, " + Tier.ToString() + ")";
}
