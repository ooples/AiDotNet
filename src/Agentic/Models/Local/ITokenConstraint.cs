namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Restricts which tokens the local engine may generate next, enabling <em>constrained decoding</em>: the
/// model can only emit tokens the constraint permits, so the output is guaranteed to satisfy a structure
/// (a fixed vocabulary, a grammar, a JSON shape) rather than merely being asked to in the prompt.
/// </summary>
/// <remarks>
/// <para>
/// At each step the engine calls <see cref="AllowedNextTokens"/> with the tokens generated so far. Returning
/// <c>null</c> means "no restriction this step"; returning a set restricts sampling to exactly those token
/// ids; returning an empty set tells the engine to stop (nothing valid can follow). This is the foundation
/// for local structured output and tool-calling — capabilities cloud models approximate with prompting but
/// cannot guarantee, and which a local engine can enforce at the logits because it controls decoding.
/// </para>
/// <para><b>For Beginners:</b> Normally a model can pick any next word-piece. A constraint is a gate that
/// only lets through the choices that keep the answer valid — for example, only digits when you want a
/// number, or only tokens that continue well-formed JSON. Because the gate is applied while generating, the
/// result is always valid by construction, not just "usually" valid.
/// </para>
/// </remarks>
public interface ITokenConstraint
{
    /// <summary>
    /// Returns the token ids permitted as the next token given what has been generated so far.
    /// </summary>
    /// <param name="generatedTokenIds">The tokens generated since the prompt (excludes the prompt itself).</param>
    /// <returns>
    /// <c>null</c> for no restriction; a non-empty set to restrict sampling to those ids; an empty set to
    /// signal that generation should stop (no valid continuation exists).
    /// </returns>
    IReadOnlyCollection<int>? AllowedNextTokens(IReadOnlyList<int> generatedTokenIds);
}
