using System.IO;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramGenomeCodecTests
{
    private static readonly ProgramGenomeCodec Codec = new();

    [Theory]
    [InlineData("print(1)", ProgramLanguage.Python, null)]
    [InlineData("Console.WriteLine(1);", ProgramLanguage.CSharp, "seed")]
    [InlineData("SELECT 1;", ProgramLanguage.SQL, "")]
    public void RoundTripPreservesEveryField(string source, ProgramLanguage language, string? description)
    {
        var genome = new ProgramGenome(source, language, description);
        ProgramGenome restored = Codec.Deserialize(Codec.Serialize(genome));

        Assert.Equal(genome.Source, restored.Source);
        Assert.Equal(genome.Language, restored.Language);
        Assert.Equal(genome.Description, restored.Description);
        Assert.Equal(genome.Id, restored.Id);
        Assert.Equal(genome, restored);
    }

    [Fact]
    public void RoundTripSurvivesCarriageReturnsTabsAndUnicode()
    {
        var genome = new ProgramGenome(
            "def f():\r\n\treturn \"caf\u00E9 \uD83D\uDE80\"\r\n", ProgramLanguage.Python);
        ProgramGenome restored = Codec.Deserialize(Codec.Serialize(genome));
        Assert.Equal(genome.Source, restored.Source);
        Assert.Equal(genome.Id, restored.Id);
    }

    [Fact]
    public void SerializationIsDeterministic()
    {
        var genome = new ProgramGenome("print(1)", ProgramLanguage.Python, "seed");
        string first = Codec.Serialize(genome);
        for (int index = 0; index < 5; index++) Assert.Equal(first, Codec.Serialize(genome));
        Assert.Equal(first, new ProgramGenomeCodec().Serialize(new ProgramGenome("print(1)", ProgramLanguage.Python, "seed")));
    }

    [Fact]
    public void OmittedDescriptionStaysNullAfterRoundTrip()
    {
        var genome = new ProgramGenome("print(1)", ProgramLanguage.Python);
        string payload = Codec.Serialize(genome);
        Assert.DoesNotContain("description", payload, StringComparison.Ordinal);
        Assert.Null(Codec.Deserialize(payload).Description);
    }

    [Fact]
    public void IdentityIsStable()
    {
        Assert.Equal("program-genome", Codec.Id);
        Assert.Equal("program-genome-v1", Codec.VersionHash);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("{}")]
    [InlineData("{\"v\":1}")]
    [InlineData("{\"v\":1,\"language\":\"Python\"}")]
    [InlineData("{\"v\":1,\"language\":\"Klingon\",\"source\":\"x\"}")]
    [InlineData("{\"v\":2,\"language\":\"Python\",\"source\":\"x\"}")]
    [InlineData("{\"v\":\"one\",\"language\":\"Python\",\"source\":\"x\"}")]
    [InlineData("{\"v\":1,\"language\":\"Python\",\"source\":\"   \"}")]
    public void MalformedPayloadsRaiseInvalidData(string payload) =>
        Assert.Throws<InvalidDataException>(() => Codec.Deserialize(payload));

    [Fact]
    public void NullArgumentsAreRejected()
    {
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => Codec.Serialize(null));
        Assert.Throws<ArgumentNullException>(() => Codec.Deserialize(null));
#pragma warning restore CS8600, CS8625
    }
}
