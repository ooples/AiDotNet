using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNet.ProgramSynthesis.Engines;

/// <summary>
/// The host-level fallback <see cref="ICodeSyntaxProvider"/>, used when neither the program-synthesis options nor the
/// synthesizer were given one. Hosts in this repository (AiDotNet.Serving, the test assembly) set it at startup;
/// applications pass a provider through <see cref="Options.ProgramSynthesisOptions.SyntaxProvider"/> instead.
/// </summary>
internal static class CodeSyntaxProviders
{
    /// <summary>Gets or sets the fallback provider, or <see langword="null"/> for none.</summary>
    internal static ICodeSyntaxProvider? Provider { get; set; }
}
