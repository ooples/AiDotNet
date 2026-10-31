using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

/// <summary>
/// #2393: TreeSitter left the core package for AiDotNet.CodeAnalysis.TreeSitter. Core reaches a parser through the
/// provider passed to the pipeline (the builder passes ProgramSynthesisOptions.SyntaxProvider), falling back to the
/// host-level <see cref="CodeSyntaxProviders.Provider"/>; with neither the pipeline produces no AST.
/// </summary>
[Collection("NonParallelIntegration")]
public class CodeSyntaxProviderTests
{
    private static CodeTokenizationResult Tokenize(ICodeSyntaxProvider? provider = null)
    {
        var baseTokenizer = CharacterTokenizer.CreateAscii(SpecialTokens.Bert(), lowercase: false);
        var tokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.CSharp, splitIdentifiers: true);
        return new CodeTokenizationPipeline(provider).TokenizeWithStructure(
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
    public void Pipeline_uses_the_provider_it_was_given_over_the_host_fallback()
    {
        var prior = CodeSyntaxProviders.Provider;
        try
        {
            CodeSyntaxProviders.Provider = null;
            var result = Tokenize(new TreeSitterCodeSyntaxProvider());
            Assert.NotEmpty(result.AstNodes);
            Assert.Null(result.AstUnavailableReason);
        }
        finally
        {
            CodeSyntaxProviders.Provider = prior;
        }
    }

    [Theory]
    [InlineData(typeof(DllNotFoundException))]
    [InlineData(typeof(BadImageFormatException))]
    [InlineData(typeof(EntryPointNotFoundException))]
    public void A_grammar_that_cannot_load_keeps_the_tokenization_and_says_why(Type failure)
    {
        var exception = Activator.CreateInstance(failure) as Exception
            ?? throw new InvalidOperationException($"{failure.Name} has no parameterless constructor.");
        var result = Tokenize(new FailingProvider(exception));
        Assert.NotEmpty(result.Tokenization.Tokens);
        Assert.Empty(result.AstNodes);
        Assert.NotNull(result.AstUnavailableReason);
        Assert.Contains(failure.Name, result.AstUnavailableReason);
    }

    [Fact]
    public void A_provider_failure_that_is_not_a_native_load_still_surfaces()
    {
        Assert.Throws<InvalidOperationException>(() => Tokenize(new FailingProvider(new InvalidOperationException("bug"))));
    }

    [Fact]
    public async Task Builder_passes_the_configured_provider_to_the_result()
    {
        var prior = CodeSyntaxProviders.Provider;
        try
        {
            CodeSyntaxProviders.Provider = null;
            var result = await new AiDotNet.AiModelBuilder<float, AiDotNet.Tensors.LinearAlgebra.Tensor<float>, AiDotNet.Tensors.LinearAlgebra.Tensor<float>>()
                .ConfigureProgramSynthesis(new AiDotNet.ProgramSynthesis.Options.ProgramSynthesisOptions
                {
                    TargetLanguage = ProgramLanguage.CSharp,
                    SyntaxProvider = new TreeSitterCodeSyntaxProvider(),
                    MaxSequenceLength = 64,
                    VocabularySize = 256,
                    NumEncoderLayers = 1,
                    NumDecoderLayers = 1,
                })
                .BuildAsync();
            var tokens = result.TokenizeCode("public class C { void M() { } }", ProgramLanguage.CSharp,
                pipelineOptions: new CodeTokenizationPipelineOptions { IncludeAst = true, MaxAstNodes = 5000 });
            Assert.NotEmpty(tokens.AstNodes);
        }
        finally
        {
            CodeSyntaxProviders.Provider = prior;
        }
    }

    private sealed class FailingProvider : ICodeSyntaxProvider
    {
        private readonly Exception _failure;
        public FailingProvider(Exception failure) { _failure = failure; }
        public bool? TryValidate(ProgramLanguage language, string sourceCode) => throw _failure;
        public bool TryExtractAst(string code, ProgramLanguage language, CodeTokenizationPipelineOptions options,
            out List<CodeAstNode> nodes, out List<CodeAstEdge> edges) => throw _failure;
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