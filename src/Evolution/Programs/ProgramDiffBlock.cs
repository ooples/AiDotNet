using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>One parsed SEARCH/REPLACE edit: the text to find and the text that replaces it.</summary>
/// <remarks>
/// <para>
/// A block is the unit an LLM emits when it edits code without rewriting a whole file. Both halves are stored with
/// line endings normalized to line feeds and with trailing white space removed, matching the reference
/// OpenEvolve parser so an edit that works there works here. <see cref="Ordinal"/> and <see cref="ResponseLine"/>
/// record where the block came from in the response, which is what lets a failure message point the model at the
/// exact instruction that could not be applied.
/// </para>
/// <para><b>For Beginners:</b> This is a single "find and replace" instruction. <see cref="SearchText"/> is the
/// snippet that must already exist in the program, and <see cref="ReplaceText"/> is what it becomes; an empty
/// replacement deletes the snippet. Because the instruction knows its own position in the model's answer, an error
/// can say "your second edit failed" rather than just "something failed".</para>
/// </remarks>
public sealed class ProgramDiffBlock
{
    /// <summary>Initializes an edit block.</summary>
    /// <param name="searchText">The text that must be found in the target program.</param>
    /// <param name="replaceText">The text that replaces the match; an empty string deletes the match.</param>
    /// <param name="ordinal">The zero-based position of this block within the response it was parsed from.</param>
    /// <param name="responseLine">The one-based response line the block's search marker appeared on.</param>
    /// <exception cref="ArgumentNullException"><paramref name="searchText"/> or <paramref name="replaceText"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="ordinal"/> or <paramref name="responseLine"/> is negative.</exception>
    public ProgramDiffBlock(string searchText, string replaceText, int ordinal = 0, int responseLine = 0)
    {
        Guard.NotNull(searchText);
        Guard.NotNull(replaceText);
        Guard.NonNegative(ordinal);
        Guard.NonNegative(responseLine);
        SearchText = searchText;
        ReplaceText = replaceText;
        Ordinal = ordinal;
        ResponseLine = responseLine;
    }

    /// <summary>Gets the text that must be found in the target program.</summary>
    public string SearchText { get; }

    /// <summary>Gets the replacement text; an empty string deletes the matched lines.</summary>
    public string ReplaceText { get; }

    /// <summary>Gets the zero-based position of this block within its response.</summary>
    public int Ordinal { get; }

    /// <summary>Gets the one-based response line the search marker appeared on, or zero when unknown.</summary>
    public int ResponseLine { get; }

    /// <summary>Gets whether the block is a pure deletion.</summary>
    public bool IsDeletion => ReplaceText.Length == 0;

    /// <summary>Returns the block ordinal and its two text sizes without echoing any code.</summary>
    /// <returns>A short diagnostic label for this block.</returns>
    public override string ToString() => string.Concat(
        "block ", Ordinal.ToString(System.Globalization.CultureInfo.InvariantCulture),
        " (search ", SearchText.Length.ToString(System.Globalization.CultureInfo.InvariantCulture),
        " chars, replace ", ReplaceText.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), " chars)");
}
