using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramDescriptorTests
{
    [Fact]
    public void LengthDescriptorMeasuresTheNormalizedSource()
    {
        var descriptor = new ProgramLengthDescriptor();
        Assert.Equal("length", descriptor.Name);

        var unix = new ProgramGenome("ab\ncd\n");
        var windows = new ProgramGenome("ab  \r\ncd\t\r\n\r\n");

        Assert.Equal(5.0, descriptor.Compute(unix));
        Assert.Equal(descriptor.Compute(unix), descriptor.Compute(windows));
    }

    [Fact]
    public void LengthDescriptorNameIsConfigurable() =>
        Assert.Equal("size", new ProgramLengthDescriptor("size").Name);

    [Theory]
    [InlineData("x = 1", 3)]
    [InlineData("x=1", 3)]
    [InlineData("x   =   1", 3)]
    [InlineData("someVeryLongName = 1", 3)]
    [InlineData("print(\"hi\")", 4)]
    [InlineData("self.value += 1", 4)]
    [InlineData("a = 3.14", 3)]
    public void TokenComplexityCountsLexicalTokens(string source, int expected) =>
        Assert.Equal(expected, ProgramTokenComplexityDescriptor.CountTokens(source));

    [Fact]
    public void TokenComplexityIsInvariantToFormattingAndNaming()
    {
        var descriptor = new ProgramTokenComplexityDescriptor();
        Assert.Equal("tokenComplexity", descriptor.Name);

        double compact = descriptor.Compute(new ProgramGenome("def f(i):\n    return i+1\n"));
        double spaced = descriptor.Compute(new ProgramGenome("def f(rowIndex):\r\n        return rowIndex + 1\r\n"));
        Assert.Equal(compact, spaced);
    }

    [Fact]
    public void TokenComplexityHandlesUnterminatedStrings() =>
        Assert.Equal(2, ProgramTokenComplexityDescriptor.CountTokens("x \"unterminated"));

    [Fact]
    public void DiversityDistanceMatchesTheDocumentedFormula()
    {
        Assert.Equal(0.0, ProgramDiversityDescriptor.ComputeDistance("ab", "ab"));
        Assert.Equal(0.6, ProgramDiversityDescriptor.ComputeDistance("ab", "abc"), 10);
        Assert.Equal(1.2, ProgramDiversityDescriptor.ComputeDistance("ab", "abcd"), 10);
        Assert.Equal(10.6, ProgramDiversityDescriptor.ComputeDistance("ab", "a\nb"), 10);
    }

    [Fact]
    public void DiversityIsIndependentOfReferenceOrderAndDuplicates()
    {
        var forward = new ProgramDiversityDescriptor(new[] { "ab", "abcd" });
        var reversed = new ProgramDiversityDescriptor(new[] { "abcd", "ab", "abcd" });
        var candidate = new ProgramGenome("ab");

        Assert.Equal("diversity", forward.Name);
        Assert.Equal(2, forward.ReferenceSources.Count);
        Assert.Equal(2, reversed.ReferenceSources.Count);
        Assert.Equal(1.2, forward.Compute(candidate), 10);
        Assert.Equal(forward.Compute(candidate), reversed.Compute(candidate), 10);
    }

    [Fact]
    public void DiversityIgnoresLineEndingDifferencesInReferences()
    {
        var descriptor = new ProgramDiversityDescriptor(new[] { "ab\r\ncd\r\n" });
        Assert.Single(descriptor.ReferenceSources);
        Assert.Equal("ab\ncd", descriptor.ReferenceSources[0]);
        Assert.Equal(0.0, descriptor.Compute(new ProgramGenome("ab\ncd\n")));
    }

    [Fact]
    public void DiversityWithoutReferencesIsZero()
    {
        var descriptor = new ProgramDiversityDescriptor(Array.Empty<string>());
        Assert.Equal(0.0, descriptor.Compute(new ProgramGenome("ab")));
    }

    [Fact]
    public void DiversityAcceptsReferenceGenomes()
    {
        var descriptor = new ProgramDiversityDescriptor(
            new[] { new ProgramGenome("abcd", ProgramLanguage.Python) });
        Assert.Equal(1.2, descriptor.Compute(new ProgramGenome("ab")), 10);
    }

    [Fact]
    public void DescriptorSetComputesEveryNamedAxis()
    {
        var set = new ProgramDescriptorSet(
            new ProgramLengthDescriptor(),
            new ProgramTokenComplexityDescriptor(),
            new ProgramDiversityDescriptor(new[] { "zzzz" }));

        IReadOnlyDictionary<string, double> values = set.Compute(new ProgramGenome("x = 1"));

        Assert.Equal(3, set.Count);
        Assert.Equal(new[] { "length", "tokenComplexity", "diversity" }, set.Names);
        Assert.Equal(5.0, values["length"]);
        Assert.Equal(3.0, values["tokenComplexity"]);
        Assert.True(values["diversity"] > 0);
    }

    [Fact]
    public void DescriptorSetRejectsDuplicateNames()
    {
        Assert.Throws<ArgumentException>(() => new ProgramDescriptorSet(
            new ProgramLengthDescriptor(), new ProgramLengthDescriptor()));
        Assert.Throws<ArgumentException>(() => new ProgramDescriptorSet(new BlankNameDescriptor()));
    }

    [Fact]
    public void DescriptorSetRejectsNonFiniteValues()
    {
        var set = new ProgramDescriptorSet(new NonFiniteDescriptor());
        Assert.Throws<InvalidOperationException>(() => set.Compute(new ProgramGenome("x")));
    }

    [Fact]
    public void DescriptorSetVersionHashTracksItsMembers()
    {
        string first = new ProgramDescriptorSet(new ProgramLengthDescriptor()).VersionHash;
        string same = new ProgramDescriptorSet(new ProgramLengthDescriptor()).VersionHash;
        string renamed = new ProgramDescriptorSet(new ProgramLengthDescriptor("size")).VersionHash;
        string extended = new ProgramDescriptorSet(
            new ProgramLengthDescriptor(), new ProgramTokenComplexityDescriptor()).VersionHash;

        Assert.Equal(first, same);
        Assert.NotEqual(first, renamed);
        Assert.NotEqual(first, extended);
        Assert.NotEqual(first, ProgramDescriptorSet.Empty().VersionHash);
    }

    [Fact]
    public void EmptySetComputesNothing()
    {
        ProgramDescriptorSet set = ProgramDescriptorSet.Empty();
        Assert.Equal(0, set.Count);
        Assert.Empty(set.Compute(new ProgramGenome("x")));
    }

    [Fact]
    public void CreateDefaultOmitsDiversityWithoutReferences()
    {
        ProgramDescriptorSet without = ProgramDescriptorSet.CreateDefault(Array.Empty<string>());
        Assert.Equal(new[] { "length", "tokenComplexity" }, without.Names);

        ProgramDescriptorSet with = ProgramDescriptorSet.CreateDefault(new[] { "seed" });
        Assert.Equal(new[] { "length", "tokenComplexity", "diversity" }, with.Names);
    }

    private sealed class BlankNameDescriptor : IProgramDescriptor
    {
        public string Name => "   ";
        public double Compute(ProgramGenome genome) => 0;
    }

    private sealed class NonFiniteDescriptor : IProgramDescriptor
    {
        public string Name => "broken";
        public double Compute(ProgramGenome genome) => double.NaN;
    }
}
