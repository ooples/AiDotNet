using System.Text;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.ArtifactStore;

/// <summary>One named piece of evidence an evaluation produced, held as bytes so binary output survives.</summary>
/// <remarks>
/// <para>
/// An artifact is what a candidate program actually emitted: standard output, a stack trace, a coverage report, a
/// generated image, a profiler trace. The reference OpenEvolve evaluator models artifacts as
/// <c>Dict[str, Union[str, bytes]]</c>, and the <c>bytes</c> half matters - a store that keeps only text cannot
/// hold a profile or a rendered plot, and base64-encoding it into a text field inflates it by a third and loses the
/// distinction on the way back out. This type keeps bytes and records whether they were supplied as UTF-8 text.
/// </para>
/// <para>
/// The content is untrusted: it was produced by code a model wrote, running against data the library did not
/// choose. Nothing here interprets it, executes it, or renders it. <see cref="IsTruncated"/> is set by a store that
/// had to cut the content to stay inside its configured limit, so a caller is never shown a partial artifact that
/// claims to be complete.
/// </para>
/// <para><b>For Beginners:</b> When a candidate program runs, it usually leaves something behind - printed output,
/// an error message, a file it wrote. This holds one of those, with a short name such as <c>stderr</c>. Use
/// <see cref="FromText"/> for anything textual and <see cref="FromBytes"/> for binary content such as an image.
/// Feeding an artifact back into the next prompt is by far the fastest way for a model to fix a broken
/// program.</para>
/// </remarks>
public sealed class ProgramArtifact
{
    /// <summary>The longest artifact name accepted, in characters.</summary>
    public const int MaxNameLength = 128;

    private static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false);

    private readonly byte[] _content;

    private ProgramArtifact(string name, byte[] content, bool isText, bool isTruncated)
    {
        Name = name;
        _content = content;
        IsText = isText;
        IsTruncated = isTruncated;
    }

    /// <summary>Creates a text artifact, encoded as UTF-8 without a byte-order mark.</summary>
    /// <param name="name">A short label such as <c>stdout</c> or <c>stderr</c>.</param>
    /// <param name="text">The captured text, exactly as produced.</param>
    /// <param name="isTruncated">Whether the text was already cut from a longer original.</param>
    /// <returns>A text artifact.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="text"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty, white space, or too long.</exception>
    public static ProgramArtifact FromText(string name, string text, bool isTruncated = false)
    {
        Guard.NotNull(text);
        return new ProgramArtifact(ValidateName(name), Utf8.GetBytes(text), isText: true, isTruncated);
    }

    /// <summary>Creates a binary artifact.</summary>
    /// <param name="name">A short label such as <c>profile.bin</c>.</param>
    /// <param name="content">The captured bytes, copied on construction.</param>
    /// <param name="isTruncated">Whether the content was already cut from a longer original.</param>
    /// <returns>A binary artifact.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="name"/> or <paramref name="content"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="name"/> is empty, white space, or too long.</exception>
    public static ProgramArtifact FromBytes(string name, IReadOnlyList<byte> content, bool isTruncated = false)
    {
        Guard.NotNull(content);
        string validated = ValidateName(name);
        var copy = new byte[content.Count];
        for (int index = 0; index < copy.Length; index++) copy[index] = content[index];
        return new ProgramArtifact(validated, copy, isText: false, isTruncated);
    }

    /// <summary>Gets the artifact's label.</summary>
    public string Name { get; }

    /// <summary>Gets whether the content was supplied as text and can be read back with <see cref="GetText"/>.</summary>
    public bool IsText { get; }

    /// <summary>Gets whether the content was cut to fit a configured limit.</summary>
    public bool IsTruncated { get; }

    /// <summary>Gets the size of the content in bytes.</summary>
    public int ByteLength => _content.Length;

    /// <summary>Gets a read-only view of the content bytes.</summary>
    public IReadOnlyList<byte> Content => Array.AsReadOnly(_content);

    /// <summary>Decodes the content as UTF-8 text.</summary>
    /// <returns>The decoded text, with invalid sequences replaced rather than throwing.</returns>
    /// <remarks>Calling this on a binary artifact is legal and yields a lossy rendering, never an exception.</remarks>
    public string GetText() => Encoding.UTF8.GetString(_content);

    /// <summary>Writes the content to a stream.</summary>
    /// <param name="destination">The stream to write to.</param>
    /// <exception cref="ArgumentNullException"><paramref name="destination"/> is <c>null</c>.</exception>
    public void CopyTo(Stream destination)
    {
        Guard.NotNull(destination);
        destination.Write(_content, 0, _content.Length);
    }

    /// <summary>Returns a copy of this artifact whose content is cut to a byte limit.</summary>
    /// <param name="maxBytes">The positive byte limit.</param>
    /// <returns>This instance when it already fits; otherwise a truncated copy.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxBytes"/> is not positive.</exception>
    public ProgramArtifact Truncate(int maxBytes)
    {
        if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes), maxBytes, "Value must be positive.");
        if (_content.Length <= maxBytes) return this;
        var cut = new byte[maxBytes];
        Array.Copy(_content, cut, maxBytes);
        return new ProgramArtifact(Name, cut, IsText, isTruncated: true);
    }

    /// <summary>Returns the name and size, never the content.</summary>
    /// <returns>A short diagnostic label that is always safe to log.</returns>
    public override string ToString() =>
        "ProgramArtifact(" + Name + ", " + _content.Length.ToString(CultureInfo.InvariantCulture) + " bytes" +
        (IsTruncated ? ", truncated)" : ")");

    internal byte[] GetContentBuffer() => _content;

    internal static ProgramArtifact FromBuffer(string name, byte[] content, bool isText, bool isTruncated) =>
        new(ValidateName(name), content, isText, isTruncated);

    private static string ValidateName(string name)
    {
        Guard.NotNullOrWhiteSpace(name);
        string trimmed = name.Trim();
        if (trimmed.Length > MaxNameLength)
            throw new ArgumentException($"An artifact name cannot exceed {MaxNameLength} characters.", nameof(name));
        return trimmed;
    }
}
