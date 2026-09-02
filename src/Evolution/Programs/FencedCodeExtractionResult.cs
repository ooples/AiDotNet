using System.Collections.ObjectModel;
using AiDotNet.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>The code recovered from a model response, together with every fence that was considered.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation returns a bare string and falls all the way back to the entire
/// response when no fence matches, so a caller cannot tell whether it received clean code or a paragraph of prose
/// that happens to be non-empty. This result reports the exact rung of the fallback ladder that produced
/// <see cref="Code"/> in <see cref="SelectionSource"/>, keeps every <see cref="Blocks"/> entry it saw, and records
/// a bounded diagnostic whenever it had to settle for something weaker than a language-labelled fence.
/// </para>
/// <para><b>For Beginners:</b> This is what you get back when you ask for the code out of a chat answer. Check
/// <see cref="HasCode"/> to see whether anything was found and <see cref="SelectionSource"/> to see how confident
/// the answer is: code taken from a properly labelled block is trustworthy, whereas code taken from the raw
/// response usually means the model ignored the requested format and is worth retrying.</para>
/// </remarks>
public sealed class FencedCodeExtractionResult
{
    private readonly ReadOnlyCollection<FencedCodeBlock> _blocks;
    private readonly ReadOnlyCollection<string> _diagnostics;

    internal FencedCodeExtractionResult(
        string code,
        FencedCodeSelectionSource selectionSource,
        IReadOnlyList<FencedCodeBlock> blocks,
        IReadOnlyList<string> diagnostics)
    {
        Code = code;
        SelectionSource = selectionSource;

        var blockCopy = new FencedCodeBlock[blocks.Count];
        for (int index = 0; index < blocks.Count; index++) blockCopy[index] = blocks[index];
        _blocks = Array.AsReadOnly(blockCopy);

        var diagnosticCopy = new string[diagnostics.Count];
        for (int index = 0; index < diagnostics.Count; index++) diagnosticCopy[index] = diagnostics[index];
        _diagnostics = Array.AsReadOnly(diagnosticCopy);
    }

    /// <summary>Gets the selected code, or an empty string when nothing could be recovered.</summary>
    public string Code { get; }

    /// <summary>Gets the fallback rung that produced <see cref="Code"/>.</summary>
    public FencedCodeSelectionSource SelectionSource { get; }

    /// <summary>Gets every fenced block seen in the response, in the order they appeared.</summary>
    public IReadOnlyList<FencedCodeBlock> Blocks => _blocks;

    /// <summary>Gets bounded notes explaining any fallback or malformed fence.</summary>
    public IReadOnlyList<string> Diagnostics => _diagnostics;

    /// <summary>Gets whether any code was recovered.</summary>
    public bool HasCode => Code.Length > 0;

    /// <summary>Gets whether the code came from a fence whose label matched the requested language.</summary>
    public bool IsConfident => SelectionSource == FencedCodeSelectionSource.LanguageLabeledFence;
}
