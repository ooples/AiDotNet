using System;
using AiDotNet.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

public class MetaLearningAlgorithmTypeIntegrationTests
{
    [Fact]
    public void Enum_ContainsCoreAlgorithms()
    {
        var names = Enum.GetNames(typeof(MetaLearningAlgorithmType));

        Assert.Contains("MAML", names);
        Assert.Contains("Reptile", names);
        Assert.Contains("MetaSGD", names);
        Assert.Contains("ProtoNets", names);
        Assert.Contains("MatchingNetworks", names);
        Assert.Contains("ANIL", names);
        Assert.Contains("BOIL", names);
        Assert.Contains("iMAML", names);
        Assert.Contains("CNAP", names);
        Assert.Contains("SEAL", names);
        Assert.Contains("TADAM", names);
        Assert.Contains("GNNMeta", names);
        Assert.Contains("NTM", names);
        Assert.Contains("MANN", names);
        Assert.Contains("RelationNetwork", names);
        Assert.Contains("LEO", names);
        Assert.Contains("MetaOptNet", names);
    }

    [Theory]
    [InlineData("MAML", MetaLearningAlgorithmType.MAML)]
    [InlineData("Reptile", MetaLearningAlgorithmType.Reptile)]
    [InlineData("MetaSGD", MetaLearningAlgorithmType.MetaSGD)]
    [InlineData("ProtoNets", MetaLearningAlgorithmType.ProtoNets)]
    [InlineData("MatchingNetworks", MetaLearningAlgorithmType.MatchingNetworks)]
    [InlineData("ANIL", MetaLearningAlgorithmType.ANIL)]
    [InlineData("BOIL", MetaLearningAlgorithmType.BOIL)]
    [InlineData("iMAML", MetaLearningAlgorithmType.iMAML)]
    [InlineData("CNAP", MetaLearningAlgorithmType.CNAP)]
    [InlineData("SEAL", MetaLearningAlgorithmType.SEAL)]
    [InlineData("TADAM", MetaLearningAlgorithmType.TADAM)]
    [InlineData("GNNMeta", MetaLearningAlgorithmType.GNNMeta)]
    [InlineData("NTM", MetaLearningAlgorithmType.NTM)]
    [InlineData("MANN", MetaLearningAlgorithmType.MANN)]
    [InlineData("RelationNetwork", MetaLearningAlgorithmType.RelationNetwork)]
    [InlineData("LEO", MetaLearningAlgorithmType.LEO)]
    [InlineData("MetaOptNet", MetaLearningAlgorithmType.MetaOptNet)]
    public void Enum_Parse_Works(string value, MetaLearningAlgorithmType expected)
    {
        Assert.True(Enum.TryParse(value, out MetaLearningAlgorithmType parsed));
        Assert.Equal(expected, parsed);
    }
}
