namespace AiDotNet.Enums;

/// <summary>Says which request a prompt makes of the model: targeted edits, a full rewrite, or either by size.</summary>
/// <remarks>
/// <para>
/// Targeted edits keep a prompt and its answer small and preserve the parts of a program that already work, but
/// they fail when the model cannot reproduce existing lines exactly. A full rewrite always parses but costs more
/// tokens and discards structure the search had already found. <see cref="AutoBySize"/> picks between them from
/// the parent program's length, which is the choice that matters in practice: a twenty-line seed program is
/// cheaper and more reliable to rewrite whole, while a thousand-line program must be edited.
/// </para>
/// <para>
/// The reference OpenEvolve implementation exposes only the first two as a boolean and never switches between
/// them during a run.
/// </para>
/// <para><b>For Beginners:</b> There are two ways to ask an AI to improve a file. You can ask for a patch ("find
/// these lines and replace them with those lines"), or you can ask for the whole file back. Patches are cheaper
/// and safer for big files; whole files are simpler and more reliable for small ones. The third option lets the
/// library decide for you based on how long the program currently is, which is usually what you want.</para>
/// </remarks>
public enum ProgramPromptEvolutionMode
{
    /// <summary>Always ask for SEARCH/REPLACE edit blocks.</summary>
    Diff = 0,

    /// <summary>Always ask for the complete replacement program.</summary>
    FullRewrite = 1,

    /// <summary>Ask for a rewrite while the parent is short and for edits once it grows past the threshold.</summary>
    AutoBySize = 2
}
