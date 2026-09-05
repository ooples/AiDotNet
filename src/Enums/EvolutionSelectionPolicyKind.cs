namespace AiDotNet.Enums;

/// <summary>Names the built-in parent-selection policy an engine builds when the caller supplies none.</summary>
/// <remarks>
/// <para>
/// <c>EvolutionEngineOptions.SelectionPolicy</c> is read by <c>EvolutionEngine&lt;TGenome&gt;</c> only when the
/// <c>selection</c> constructor argument is <see langword="null"/>, so an explicitly supplied policy always wins and
/// this enumeration can never override code the caller wrote. <see cref="Uniform"/> is the default and is the rule
/// the engine has always used, so an existing configuration keeps its exact behaviour.
/// </para>
/// <para>
/// The constructed policy's own identifier and version hash are folded into the engine's compatibility hash, and a
/// ratio policy's version hash additionally covers every value of <c>EvolutionEngineOptions.Selection</c>. A
/// checkpoint written by a ratio-configured run is therefore refused by a uniform-configured one, and the reverse,
/// without this enumeration having to enter the configuration hash itself.
/// </para>
/// <para><b>For Beginners:</b> Before an evolutionary run can create a new candidate it has to choose which
/// existing candidate to build on. That choice is the "selection policy", and this setting picks one of the four
/// policies that ship with the library instead of making you write code. <see cref="Uniform"/> treats every stored
/// solution equally, which explores widely; <see cref="Ratio"/> mostly builds on the strongest solutions, which
/// converges faster; <see cref="Curiosity"/> favours solutions whose children have recently been paying off; and
/// <see cref="Double"/> explores like <see cref="Uniform"/> but shows the variation operator the best solutions as
/// inspiration. Start with <see cref="Uniform"/> and switch to <see cref="Ratio"/> when the search is exploring
/// widely but not improving.</para>
/// </remarks>
public enum EvolutionSelectionPolicyKind
{
    /// <summary>Every occupied archive cell is equally likely to become the parent; the engine default.</summary>
    Uniform = 0,

    /// <summary>Mixes exploration, elite exploitation, and island-best draws using <c>EvolutionEngineOptions.Selection</c>.</summary>
    Ratio = 1,

    /// <summary>Weights each elite by a bounded curiosity score that rises when its offspring improve an archive.</summary>
    Curiosity = 2,

    /// <summary>Samples the parent uniformly but supplies the highest-quality elites as inspirations.</summary>
    Double = 3
}
