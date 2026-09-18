using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramSourceIdentityTests
{
    [Theory]
    [InlineData(ProgramLanguage.CSharp, "var text = \"\"\"\nvalue \n\"\"\";", "var text = \"\"\"\nvalue\n\"\"\";")]
    [InlineData(ProgramLanguage.CSharp, "var text = @\"value \nend\";", "var text = @\"value\nend\";")]
    [InlineData(ProgramLanguage.Python, "text = '''value \nend'''", "text = '''value\nend'''")]
    public void DisplayNormalizationCannotMergeMeaningfulStringWhitespace(ProgramLanguage language, string first, string second)
    {
        var left = new ProgramGenome(first, language); var right = new ProgramGenome(second, language);
        Assert.Equal(left.NormalizedSource, right.NormalizedSource);
        Assert.NotEqual(left.Id, right.Id);
        Assert.NotEqual(left, right);
        Assert.Equal(left.Id, ProgramGenome.ComputeId(first, language));
        if (language == ProgramLanguage.CSharp)
        {
            // Roslyn independently confirms these are different literal values, not cosmetic edits.
            string Literal(string source) => CSharpSyntaxTree.ParseText(source).GetRoot().DescendantTokens()
                .Single(token => token.RawKind == (int)SyntaxKind.StringLiteralToken || token.RawKind == (int)SyntaxKind.MultiLineRawStringLiteralToken).ValueText;
            Assert.NotEqual(Literal(first), Literal(second));
        }
        var restored = new ProgramGenomeCodec().Deserialize(new ProgramGenomeCodec().Serialize(left));
        Assert.Equal(first, restored.Source); Assert.Equal(left.Id, restored.Id);
    }

    [Fact]
    public void ExactLineEndingsArePartOfTheExecutionIdentity()
    {
        var first = new ProgramGenome("var text = @\"a\r\nb\";", ProgramLanguage.CSharp);
        var second = new ProgramGenome("var text = @\"a\nb\";", ProgramLanguage.CSharp);
        Assert.Equal(first.NormalizedSource, second.NormalizedSource);
        Assert.NotEqual(first.Id, second.Id);
    }

    [Fact]
    public async Task SourceLimitIncludesWhitespaceActuallySentToTheEvaluator()
    {
        int calls = 0;
        var task = new ProgramEvolutionTask(new DelegateProgramFitnessEvaluator(_ => { calls++; return 1; }),
            options: new ProgramEvolutionOptions { MaxProgramChars = 10 });
        var genome = new ProgramGenome("x" + new string(' ', 20));
        var canonical = await task.CanonicalizeAsync(genome);
        var candidate = new EvolutionCandidate<ProgramGenome>(0, canonical, new EvolutionLineage(null, null, "seed", null, 0, 0, 0));
        var result = await task.EvaluateAsync(candidate, new EvolutionEvaluationContext(0, 1, 1, 1));
        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status); Assert.Equal(0, calls);
    }

    [Fact]
    public void CheckpointCompatibilityIsVersionedWhenIdentitySemanticsChange() =>
        Assert.NotEqual("program-genome-v1", new ProgramGenomeCodec().VersionHash);

    [Theory]
    [InlineData(0xD800, false)]
    [InlineData(0xDC00, false)]
    [InlineData(0xD800, true)]
    public void IllFormedUnicodeCannotCollapseThroughReplacementEncoding(int codeUnit, bool addSuffix)
    {
        string source = "x" + (char)codeUnit + (addSuffix ? "y" : string.Empty);
        Assert.Throws<ArgumentException>(() => new ProgramGenome(source));
        Assert.Throws<ArgumentException>(() => ProgramGenome.ComputeId(source));
    }
}
