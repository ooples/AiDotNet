using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Tokenization;

namespace AiDotNet.ProgramSynthesis.Interfaces;

/// <summary>
/// Parses source code for the program synthesizer and the code-tokenization pipeline: syntax validation and
/// abstract-syntax-tree extraction.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> a real parser lets AiDotNet check that generated code is valid and see its structure.
/// The TreeSitter implementation ships in the opt-in <c>AiDotNet.CodeAnalysis.TreeSitter</c> package, so its native
/// grammars stay out of the core package. Pass one through <see cref="Options.ProgramSynthesisOptions.SyntaxProvider"/>
/// (with <c>AiModelBuilder.ConfigureProgramSynthesis</c>) or the <c>NeuralProgramSynthesizer</c> constructor; without
/// one, validation falls back to generic structural checks and pipelines produce no AST.
/// </remarks>
public interface ICodeSyntaxProvider
{
    /// <summary>
    /// Validates <paramref name="sourceCode"/> as <paramref name="language"/>.
    /// </summary>
    /// <returns>Whether the code parses without errors, or <see langword="null"/> when this provider has no parser
    /// for the language (the caller then uses its generic validation).</returns>
    bool? TryValidate(ProgramLanguage language, string sourceCode);

    /// <summary>
    /// Extracts the abstract syntax tree of <paramref name="code"/>.
    /// </summary>
    /// <returns><see langword="true"/> when an AST was produced; <see langword="false"/> when the language has no
    /// parser or <paramref name="options"/> does not ask for an AST.</returns>
    bool TryExtractAst(
        string code,
        ProgramLanguage language,
        CodeTokenizationPipelineOptions options,
        out List<CodeAstNode> nodes,
        out List<CodeAstEdge> edges);
}