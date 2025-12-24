using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.CodeTokenization;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public class CodeTokenizationPipelineTests
{
    [Fact]
    public void Tokenize_Sql_ProducesOffsetsAndSpans()
    {
        var baseTokenizer = CharacterTokenizer.CreateAscii(SpecialTokens.Bert(), lowercase: false);
        var tokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.SQL, splitIdentifiers: true);
        var pipeline = new CodeTokenizationPipeline();

        var code = "SELECT 1;";
        var result = pipeline.Tokenize(
            code: code,
            language: ProgramLanguage.SQL,
            tokenizer: tokenizer,
            options: new EncodingOptions { AddSpecialTokens = false });

        Assert.Equal(ProgramLanguage.SQL, result.Language);
        Assert.NotEmpty(result.Tokenization.Tokens);
        Assert.Equal(result.Tokenization.Tokens.Count, result.Tokenization.Offsets.Count);
        Assert.Equal(result.Tokenization.Tokens.Count, result.TokenSpans.Count);
        Assert.Equal("SELECT", result.Tokenization.Tokens[0]);
        Assert.Equal(0, result.Tokenization.Offsets[0].Start);
        Assert.Equal(6, result.Tokenization.Offsets[0].End);
        Assert.Equal(1, result.TokenSpans[0].Start.Line);
        Assert.Equal(1, result.TokenSpans[0].Start.Column);
    }

    [Fact]
    public void TokenizeWithStructure_CSharp_IncludesAstNodes()
    {
        var baseTokenizer = CharacterTokenizer.CreateAscii(SpecialTokens.Bert(), lowercase: false);
        var tokenizer = new CodeTokenizer(baseTokenizer, ProgrammingLanguage.CSharp, splitIdentifiers: true);
        var pipeline = new CodeTokenizationPipeline();

        var result = pipeline.TokenizeWithStructure(
            code: "public class C { void M() { } }",
            language: ProgramLanguage.CSharp,
            tokenizer: tokenizer,
            pipelineOptions: new CodeTokenizationPipelineOptions { IncludeAst = true, MaxAstNodes = 5000 },
            options: new EncodingOptions { AddSpecialTokens = false });

        Assert.Equal(ProgramLanguage.CSharp, result.Language);
        Assert.NotEmpty(result.AstNodes);
        Assert.NotEmpty(result.AstEdges);
        Assert.All(result.AstNodes, n => Assert.Equal(ProgramLanguage.CSharp, n.Language));
    }
}
