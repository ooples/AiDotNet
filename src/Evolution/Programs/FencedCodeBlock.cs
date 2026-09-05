using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>One fenced code block found in a model response.</summary>
/// <remarks>
/// <para>
/// A fence opens with three or more backticks or tildes and closes with a run of at least the same length of the
/// same character, which is what allows a four-backtick fence to contain three-backtick fences without being cut
/// short. <see cref="Label"/> is the info string that follows the opening run, lowercased and trimmed, and
/// <see cref="Language"/> is that label resolved through <see cref="ProgramLanguageDetector"/> when it is
/// recognized.
/// </para>
/// <para><b>For Beginners:</b> When a chat model shows you code it usually wraps it in triple backticks, often
/// with a language name right after the opening backticks. This object is one such wrapped block: the language
/// name that was written, the code inside, and where in the answer it appeared. Keeping every block rather than
/// only the first lets a caller choose sensibly when the model shows an example before the real answer.</para>
/// </remarks>
public sealed class FencedCodeBlock
{
    /// <summary>Initializes a fenced code block.</summary>
    /// <param name="label">The info string after the opening fence, or an empty string when there was none.</param>
    /// <param name="content">The code between the fences.</param>
    /// <param name="startLine">The one-based response line of the opening fence.</param>
    /// <param name="fenceLength">The number of fence characters in the opening run.</param>
    /// <exception cref="ArgumentNullException"><paramref name="label"/> or <paramref name="content"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="startLine"/> is negative or <paramref name="fenceLength"/> is below three.</exception>
    public FencedCodeBlock(string label, string content, int startLine, int fenceLength)
    {
        Guard.NotNull(label);
        Guard.NotNull(content);
        Guard.NonNegative(startLine);
        if (fenceLength < 3) throw new ArgumentOutOfRangeException(nameof(fenceLength), fenceLength, "A fence needs at least three characters.");

        Label = label;
        Content = content;
        StartLine = startLine;
        FenceLength = fenceLength;
        Language = ProgramLanguageDetector.TryGetLanguageForFenceLabel(label, out ProgramLanguage resolved)
            ? resolved
            : (ProgramLanguage?)null;
    }

    /// <summary>Gets the trimmed info string that followed the opening fence, or an empty string.</summary>
    public string Label { get; }

    /// <summary>Gets the code between the fences.</summary>
    public string Content { get; }

    /// <summary>Gets the language the label resolves to, or <c>null</c> when the label is missing or unknown.</summary>
    public ProgramLanguage? Language { get; }

    /// <summary>Gets the one-based response line of the opening fence.</summary>
    public int StartLine { get; }

    /// <summary>Gets the number of fence characters in the opening run.</summary>
    public int FenceLength { get; }

    /// <summary>Gets whether the fence carried no info string.</summary>
    public bool IsUnlabeled => Label.Length == 0;

    /// <summary>Returns the label and content size without echoing the code.</summary>
    /// <returns>A short diagnostic label for this block.</returns>
    public override string ToString() => string.Concat(
        IsUnlabeled ? "(unlabeled)" : Label, " @ line ",
        StartLine.ToString(System.Globalization.CultureInfo.InvariantCulture), ", ",
        Content.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), " chars");
}
