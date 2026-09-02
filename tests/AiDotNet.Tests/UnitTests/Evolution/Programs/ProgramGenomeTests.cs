using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramGenomeTests
{
    [Fact]
    public void NormalizationUnifiesLineEndingsAndTrailingWhitespace()
    {
        var unix = new ProgramGenome("def solve():\n    return 1\n");
        var windows = new ProgramGenome("def solve():  \r\n    return 1\t\r\n\r\n");
        var oldMac = new ProgramGenome("def solve():\r    return 1\r");

        Assert.Equal("def solve():\n    return 1", unix.NormalizedSource);
        Assert.Equal(unix.NormalizedSource, windows.NormalizedSource);
        Assert.Equal(unix.NormalizedSource, oldMac.NormalizedSource);
        Assert.Equal(unix.Id, windows.Id);
        Assert.Equal(unix.Id, oldMac.Id);
    }

    [Fact]
    public void NormalizationPreservesLeadingIndentation()
    {
        var genome = new ProgramGenome("if x:\n        return 1\n");
        Assert.Equal("if x:\n        return 1", genome.NormalizedSource);
    }

    [Fact]
    public void ByteOrderMarkIsStrippedBeforeHashing()
    {
        var withMark = new ProgramGenome("\uFEFFprint(1)");
        var without = new ProgramGenome("print(1)");
        Assert.Equal(without.Id, withMark.Id);
    }

    [Fact]
    public void IdIsStableLowercaseHexSha256OfNormalizedSource()
    {
        var genome = new ProgramGenome("print(1)\n");
        Assert.Equal(64, genome.Id.Length);
        foreach (char character in genome.Id)
        {
            Assert.True((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'),
                "Identity must be lowercase hexadecimal.");
        }

        Assert.Equal(genome.Id, ProgramGenome.ComputeId("print(1)   \r\n"));
        Assert.Equal(genome.Id, new ProgramGenome("print(1)").Id);
    }

    [Fact]
    public void IdIsUnchangedAcrossRepeatedConstruction()
    {
        const string Source = "class A:\n    def run(self):\n        return 42\n";
        string first = new ProgramGenome(Source).Id;
        for (int index = 0; index < 5; index++) Assert.Equal(first, new ProgramGenome(Source).Id);
    }

    [Fact]
    public void UnicodeSourcesAreSupportedAndDistinguished()
    {
        var accented = new ProgramGenome("caf\u00E9 = 1\n");
        var plain = new ProgramGenome("cafe = 1\n");
        var emoji = new ProgramGenome("x = \"\uD83D\uDE80\"\n");

        Assert.NotEqual(plain.Id, accented.Id);
        Assert.NotEqual(plain.Id, emoji.Id);
        Assert.Equal(accented.Id, new ProgramGenome("caf\u00E9 = 1").Id);
    }

    [Fact]
    public void ValueEqualityCoversSourceLanguageAndDescription()
    {
        var first = new ProgramGenome("print(1)\n", ProgramLanguage.Python, "seed");
        var second = new ProgramGenome("print(1)", ProgramLanguage.Python, "seed");
        var otherLanguage = new ProgramGenome("print(1)", ProgramLanguage.Generic, "seed");
        var otherDescription = new ProgramGenome("print(1)", ProgramLanguage.Python, "child");

        Assert.True(first.Equals(second));
        Assert.True(first == second);
        Assert.Equal(first.GetHashCode(), second.GetHashCode());
        Assert.False(first == otherLanguage);
        Assert.True(first != otherDescription);
        Assert.False(first.Equals(null));
    }

    [Fact]
    public void WithSourceAndWithDescriptionProduceNewInstances()
    {
        var original = new ProgramGenome("print(1)", ProgramLanguage.Python, "seed");
        ProgramGenome rewritten = original.WithSource("print(2)");
        ProgramGenome described = original.WithDescription("child");

        Assert.Equal(ProgramLanguage.Python, rewritten.Language);
        Assert.Equal("seed", rewritten.Description);
        Assert.NotEqual(original.Id, rewritten.Id);
        Assert.Equal(original.Id, described.Id);
        Assert.Equal("child", described.Description);
        Assert.Equal("print(1)", original.Source);
    }

    [Fact]
    public void LineCountCountsNormalizedLines()
    {
        Assert.Equal(1, new ProgramGenome("a").LineCount);
        Assert.Equal(3, new ProgramGenome("a\r\nb\r\nc\r\n").LineCount);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("\r\n\r\n")]
    [InlineData("\t \n \t")]
    public void BlankSourcesAreRejected(string source) =>
        Assert.Throws<ArgumentException>(() => new ProgramGenome(source));

    [Fact]
    public void NullSourceIsRejected()
    {
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => new ProgramGenome(null));
#pragma warning restore CS8600, CS8625
    }

    [Fact]
    public void HugeSourcesAreRejectedBeforeHashing()
    {
        string oversized = new string('x', ProgramGenome.MaxSourceLength + 1);
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramGenome(oversized));
    }

    [Fact]
    public void SourcesAtTheLimitAreAccepted()
    {
        string atLimit = new string('x', ProgramGenome.MaxSourceLength);
        var genome = new ProgramGenome(atLimit);
        Assert.Equal(ProgramGenome.MaxSourceLength, genome.NormalizedSource.Length);
    }

    [Fact]
    public void OversizedDescriptionsAreRejected()
    {
        string description = new string('d', ProgramGenome.MaxDescriptionLength + 1);
        Assert.Throws<ArgumentException>(() => new ProgramGenome("print(1)", ProgramLanguage.Python, description));
    }

    [Fact]
    public void UndefinedLanguageIsRejected() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramGenome("print(1)", (ProgramLanguage)999));

    [Fact]
    public void ToStringDoesNotEchoSource()
    {
        var genome = new ProgramGenome("secret_token = 'abc'\n");
        string text = genome.ToString();
        Assert.DoesNotContain("secret_token", text, StringComparison.Ordinal);
        Assert.Contains(genome.Id.Substring(0, 12), text, StringComparison.Ordinal);
    }
}
