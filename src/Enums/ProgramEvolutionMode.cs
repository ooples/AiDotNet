namespace AiDotNet.Enums;

/// <summary>Selects how a language model is asked to change a candidate program.</summary>
/// <remarks>
/// <para>
/// Diff mode asks for SEARCH/REPLACE edit blocks and applies them to the parent, which keeps the untouched parts
/// of a file byte identical and makes small, reviewable changes; full-rewrite mode asks for the whole program back
/// inside a fenced code block, which lets the model restructure freely at the cost of losing everything it forgot
/// to reproduce. The reference OpenEvolve implementation exposes this as the boolean <c>diff_based_evolution</c>,
/// defaulting to diff mode; the same default applies here.
/// </para>
/// <para><b>For Beginners:</b> There are two ways to have a model improve a program. You can ask for a list of
/// small "find this, replace it with that" edits, or you can ask for the entire improved file. Edits are safer,
/// because anything the model does not mention stays exactly as it was, and they use fewer tokens on long files.
/// Full rewrites give the model more freedom to reorganize, which helps when the change is structural. Start with
/// <see cref="Diff"/> and switch only if the model keeps failing to express a change as edits.</para>
/// </remarks>
public enum ProgramEvolutionMode
{
    /// <summary>Ask for SEARCH/REPLACE edit blocks and apply them to the parent program.</summary>
    Diff = 0,

    /// <summary>Ask for the complete replacement program inside a fenced code block.</summary>
    FullRewrite = 1
}
