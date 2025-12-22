using AiDotNet.FederatedLearning.Personalization;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class PersonalizedFederatedLearningTests
{
    [Fact]
    public void Constructor_ThrowsForInvalidFraction()
    {
        Assert.Throws<ArgumentException>(() => new PersonalizedFederatedLearning<double>(personalizationFraction: -0.1));
        Assert.Throws<ArgumentException>(() => new PersonalizedFederatedLearning<double>(personalizationFraction: 1.1));
    }

    [Fact]
    public void IdentifyPersonalizedLayers_LastN_SelectsExpectedTailLayers()
    {
        var pfl = new PersonalizedFederatedLearning<double>(personalizationFraction: 0.4);
        var structure = new Dictionary<string, double[]>
        {
            ["layer1"] = new[] { 1.0 },
            ["layer2"] = new[] { 2.0 },
            ["layer3"] = new[] { 3.0 },
            ["layer4"] = new[] { 4.0 },
            ["layer5"] = new[] { 5.0 }
        };

        pfl.IdentifyPersonalizedLayers(structure, strategy: "last_n");

        var personalized = pfl.GetPersonalizedLayers();
        Assert.Equal(0.4, pfl.GetPersonalizationFraction(), precision: 10);

        Assert.Contains("layer4", personalized);
        Assert.Contains("layer5", personalized);
        Assert.DoesNotContain("layer1", personalized);
        Assert.DoesNotContain("layer2", personalized);
        Assert.DoesNotContain("layer3", personalized);

        Assert.True(pfl.IsLayerPersonalized("layer4"));
        Assert.False(pfl.IsLayerPersonalized("layer2"));
    }

    [Fact]
    public void SeparateModel_ThenCombineModels_RoundTripsStructure()
    {
        var pfl = new PersonalizedFederatedLearning<double>(personalizationFraction: 0.5);
        var fullModel = new Dictionary<string, double[]>
        {
            ["a"] = new[] { 1.0, 2.0 },
            ["b"] = new[] { 3.0 },
            ["c"] = new[] { 4.0 }
        };

        pfl.IdentifyPersonalizedLayers(fullModel, strategy: "last_n");
        pfl.SeparateModel(fullModel, out var globalPart, out var personalizedPart);

        Assert.Single(globalPart);
        Assert.Equal(2, personalizedPart.Count);
        Assert.Contains("a", globalPart.Keys);
        Assert.Contains("b", personalizedPart.Keys);
        Assert.Contains("c", personalizedPart.Keys);

        var combined = pfl.CombineModels(globalPart, personalizedPart);
        Assert.Equal(3, combined.Count);

        Assert.Equal(fullModel["a"][0], combined["a"][0], precision: 10);
        Assert.Equal(fullModel["b"][0], combined["b"][0], precision: 10);
        Assert.Equal(fullModel["c"][0], combined["c"][0], precision: 10);

        var stats = pfl.GetModelSplitStatistics(fullModel);
        Assert.Equal(4.0, stats["total_parameters"], precision: 10);
        Assert.Equal(2.0, stats["global_parameters"], precision: 10);
        Assert.Equal(2.0, stats["personalized_parameters"], precision: 10);
        Assert.Equal(0.5, stats["communication_reduction"], precision: 10);
    }

    [Fact]
    public void IdentifyPersonalizedLayers_ByPattern_SelectsMatchingLayers()
    {
        var pfl = new PersonalizedFederatedLearning<double>(personalizationFraction: 0.2);
        var structure = new Dictionary<string, double[]>
        {
            ["conv1"] = new[] { 1.0 },
            ["head_fc"] = new[] { 2.0 },
            ["head_bn"] = new[] { 3.0 }
        };

        pfl.IdentifyPersonalizedLayers(structure, strategy: "by_pattern", customPatterns: new HashSet<string> { "head" });

        Assert.True(pfl.IsLayerPersonalized("head_fc"));
        Assert.True(pfl.IsLayerPersonalized("head_bn"));
        Assert.False(pfl.IsLayerPersonalized("conv1"));
    }

    [Fact]
    public void IdentifyPersonalizedLayers_ThrowsForInvalidInputs()
    {
        var pfl = new PersonalizedFederatedLearning<double>(personalizationFraction: 0.5);

        Assert.Throws<ArgumentException>(() => pfl.IdentifyPersonalizedLayers(null!, strategy: "last_n"));
        Assert.Throws<ArgumentException>(() => pfl.IdentifyPersonalizedLayers(new Dictionary<string, double[]>(), strategy: "last_n"));
        Assert.Throws<ArgumentException>(() => pfl.IdentifyPersonalizedLayers(new Dictionary<string, double[]> { ["a"] = new[] { 1.0 } }, strategy: "bogus"));
        Assert.Throws<ArgumentException>(() => pfl.IdentifyPersonalizedLayers(new Dictionary<string, double[]> { ["a"] = new[] { 1.0 } }, strategy: "by_pattern", customPatterns: null));
    }

    [Fact]
    public void SeparateModel_AndSplitStatistics_ValidateInputs()
    {
        var pfl = new PersonalizedFederatedLearning<double>(personalizationFraction: 0.5);

        Assert.Throws<ArgumentException>(() => pfl.SeparateModel(null!, out _, out _));
        Assert.Throws<ArgumentException>(() => pfl.GetModelSplitStatistics(null!));
        Assert.Throws<ArgumentException>(() => pfl.GetModelSplitStatistics(new Dictionary<string, double[]>()));
    }

    [Fact]
    public void CombineModels_ThrowsForNullArguments()
    {
        var pfl = new PersonalizedFederatedLearning<double>(personalizationFraction: 0.5);

        Assert.Throws<ArgumentNullException>(() => pfl.CombineModels(null!, new Dictionary<string, double[]>()));
        Assert.Throws<ArgumentNullException>(() => pfl.CombineModels(new Dictionary<string, double[]>(), null!));
    }
}
