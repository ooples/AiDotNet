using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNet.ProgramSynthesis.Engines;

/// <summary>
/// The process-wide <see cref="ICodeSyntaxProvider"/> the program synthesizer and the code-tokenization pipeline use.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> register a parser once at startup, for example the TreeSitter one from the opt-in
/// <c>AiDotNet.CodeAnalysis.TreeSitter</c> package:
/// <code>
/// AiDotNet.ProgramSynthesis.Engines.CodeSyntaxProviders.Provider =
///     new AiDotNet.ProgramSynthesis.Tokenization.TreeSitterCodeSyntaxProvider();
/// </code>
/// With none registered, syntax validation uses generic structural checks and pipelines produce no AST.
/// </remarks>
public static class CodeSyntaxProviders
{
    /// <summary>Gets or sets the registered provider, or <see langword="null"/> for none.</summary>
    public static ICodeSyntaxProvider? Provider { get; set; }
}