using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

/// <summary>
/// #2393: TreeSitter left the core package for AiDotNet.CodeAnalysis.TreeSitter. Core reaches a parser only through
/// <see cref="CodeSyntaxProviders.Provider"/>; with none registered the pipeline produces no AST.
/// </summary>
[Collection("NonParallelIntegration")]
public class CodeSyntaxProviderTests
{
    private static CodeTokenizationResult Tokenize()
    {
        var baseTokenizer = CharacterTokenizer.CreateAscii(SpecialTokens.Bert(), lowercase: false);
        var tokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.CSharp, splitIdentifiers: true);
        return new CodeTokenizationPipeline().TokenizeWithStructure(
            code: "public class C { void M() { } }",
            language: ProgramLanguage.CSharp,
            tokenizer: tokenizer,
            pipelineOptions: new CodeTokenizationPipelineOptions { IncludeAst = true, MaxAstNodes = 5000 },
            options: new EncodingOptions { AddSpecialTokens = false });
    }

    [Fact]
    public void Pipeline_produces_an_ast_only_through_a_registered_provider()
    {
        var prior = CodeSyntaxProviders.Provider;
        try
        {
            CodeSyntaxProviders.Provider = null;
            Assert.Empty(Tokenize().AstNodes);

            CodeSyntaxProviders.Provider = new TreeSitterCodeSyntaxProvider();
            var withProvider = Tokenize();
            Assert.NotEmpty(withProvider.AstNodes);
            Assert.NotEmpty(withProvider.AstEdges);
        }
        finally
        {
            CodeSyntaxProviders.Provider = prior;
        }
    }

    [Fact]
    public void TreeSitter_provider_validates_and_declines_languages_it_cannot_parse()
    {
        var provider = new TreeSitterCodeSyntaxProvider();
        Assert.Equal(true, provider.TryValidate(ProgramLanguage.Python, "def f(x):\n    return x + 1\n"));
        Assert.Equal(false, provider.TryValidate(ProgramLanguage.Python, "def f(:\n    return\n"));
        Assert.Null(provider.TryValidate(ProgramLanguage.SQL, "SELECT 1"));
    }
}