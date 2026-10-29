using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using TreeSitter;

namespace AiDotNet.ProgramSynthesis.Tokenization;

/// <summary>
/// <see cref="ICodeSyntaxProvider"/> backed by TreeSitter grammars for C#, Python, Java, JavaScript, TypeScript, C,
/// C++, Go and Rust.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> pass it to the builder so the program synthesizer validates generated code with a real
/// parser and the code-tokenization pipeline can produce syntax trees:
/// <code>
/// builder.ConfigureProgramSynthesis(new ProgramSynthesisOptions
/// {
///     SyntaxProvider = new AiDotNet.ProgramSynthesis.Tokenization.TreeSitterCodeSyntaxProvider()
/// });
/// </code>
/// </remarks>
public sealed class TreeSitterCodeSyntaxProvider : ICodeSyntaxProvider
{
    /// <inheritdoc />
    public bool? TryValidate(ProgramLanguage language, string sourceCode)
    {
        var languageSpec = TreeSitterAstExtractor.GetTreeSitterLanguageSpec(language);
        if (languageSpec is null)
            return null;

        using var tsLanguage = new Language(languageSpec.Value.LibraryName, languageSpec.Value.FunctionName);
        using var parser = new Parser(tsLanguage);
        using var tree = parser.Parse(sourceCode ?? string.Empty);
        if (tree is null)
            return false;

        var root = tree.RootNode;
        return !root.HasError && !root.IsError;
    }

    /// <inheritdoc />
    public bool TryExtractAst(
        string code,
        ProgramLanguage language,
        CodeTokenizationPipelineOptions options,
        out List<CodeAstNode> nodes,
        out List<CodeAstEdge> edges)
        => TreeSitterAstExtractor.TryExtractAst(code, language, options, out nodes, out edges);
}