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
    }

    [Theory]
    [InlineData("MAML", MetaLearningAlgorithmType.MAML)]
    [InlineData("Reptile", MetaLearningAlgorithmType.Reptile)]
    [InlineData("MetaSGD", MetaLearningAlgorithmType.MetaSGD)]
    [InlineData("ProtoNets", MetaLearningAlgorithmType.ProtoNets)]
    [InlineData("MatchingNetworks", MetaLearningAlgorithmType.MatchingNetworks)]
    public void Enum_Parse_Works(string value, MetaLearningAlgorithmType expected)
    {
        Assert.True(Enum.TryParse(value, out MetaLearningAlgorithmType parsed));
        Assert.Equal(expected, parsed);
    }
}
