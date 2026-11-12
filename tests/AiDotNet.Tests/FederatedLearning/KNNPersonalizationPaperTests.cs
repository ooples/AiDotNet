using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Personalization;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Reference tests for kNN-Per (Marfoq et al., ICML 2022, Eq. 4: neighbor weights exp(-d / tau)) and the
/// generic-T FedAGHN gradient similarity / adaptive weights.
/// </summary>
public class KNNPersonalizationPaperTests
{
    private const int Precision = 5;

    private static KNNPersonalization<double> BuildLine(int k, double temperature = 1.0, double offset = 0.0)
    {
        var knn = new KNNPersonalization<double>(k: k, temperature: temperature);
        knn.BuildCache(
            new[] { new[] { offset + 0.0 }, new[] { offset + 1.0 }, new[] { offset + 3.0 } },
            new[] { 0, 1, 1 });
        return knn;
    }

    [Fact(Timeout = 60000)]
    public void KNNPredict_UsesExpNegativeEuclideanDistanceWeights()
    {
        // Query 0 -> neighbors at distance 0 (class 0) and 1 (class 1): p = [1, e^-1] / (1 + e^-1).
        // The old max(0, cosine) weighting returned [0.5, 0.5] here (zero-norm query).
        var p = BuildLine(k: 2).KNNPredict(new[] { 0.0 }, numClasses: 2);

        Assert.Equal(0.7310585786, p[0], Precision);
        Assert.Equal(0.2689414214, p[1], Precision);
    }

    [Fact(Timeout = 60000)]
    public void KNNPredict_TemperatureScalesDistances()
    {
        // tau = 0.5: weights [1, e^-2].
        var p = BuildLine(k: 2, temperature: 0.5).KNNPredict(new[] { 0.0 }, numClasses: 2);

        Assert.Equal(0.8807970780, p[0], Precision);
        Assert.Equal(0.1192029220, p[1], Precision);
    }

    [Fact(Timeout = 60000)]
    public void KNNPredict_Float_DistantCacheDoesNotUnderflowToUniform()
    {
        // Distances 1000 and 1001: exp(-1000) underflows, but Eq. 4 normalizes, so the answer equals the
        // offset-0 case.
        var knn = new KNNPersonalization<float>(k: 2);
        knn.BuildCache(new[] { new[] { 1000f }, new[] { 1001f }, new[] { 1003f } }, new[] { 0, 1, 1 });

        var p = knn.KNNPredict(new[] { 0f }, numClasses: 2);

        Assert.Equal(0.7310586, (double)p[0], 4);
        Assert.Equal(0.2689414, (double)p[1], 4);
    }

    [Fact(Timeout = 60000)]
    public void KNNPredict_CosineDistance_UsesOneMinusCosine()
    {
        var knn = new KNNPersonalization<double>(k: 2, distanceMetric: DistanceMetricType.Cosine);
        knn.BuildCache(
            new[] { new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 }, new[] { 1.0, 1.0 } },
            new[] { 0, 1, 1 });

        // Distances 0, 1, 1 - 1/sqrt(2): neighbors 0 (class 0) and 2 (class 1).
        var p = knn.KNNPredict(new[] { 1.0, 0.0 }, numClasses: 2);

        Assert.Equal(0.5727042928, p[0], Precision);
        Assert.Equal(0.4272957072, p[1], Precision);
    }

    [Fact(Timeout = 60000)]
    public void KNNPredict_NoValidNeighborLabel_ReturnsUniform()
    {
        var knn = new KNNPersonalization<double>(k: 2);
        knn.BuildCache(new[] { new[] { 0.0 }, new[] { 1.0 } }, new[] { 5, -1 });

        var p = knn.KNNPredict(new[] { 0.0 }, numClasses: 4);

        Assert.All(p, value => Assert.Equal(0.25, value, Precision));
    }

    [Fact(Timeout = 60000)]
    public void CombinedPredict_MixesKnnAndGlobalWithLambda()
    {
        var knn = new KNNPersonalization<double>(k: 2, lambda: 0.5);
        knn.BuildCache(new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 3.0 } }, new[] { 0, 1, 1 });

        var p = knn.CombinedPredict(new[] { 0.0 }, new[] { 0.0, 1.0 }, numClasses: 2);

        Assert.Equal(0.3655292893, p[0], Precision);
        Assert.Equal(0.6344707107, p[1], Precision);
    }

    [Fact(Timeout = 60000)]
    public void BuildCache_RejectsMixedFeatureLengths()
    {
        var knn = new KNNPersonalization<double>();

        Assert.Throws<ArgumentException>(() =>
            knn.BuildCache(new[] { new[] { 0.0 }, new[] { 1.0, 2.0 } }, new[] { 0, 1 }));
    }

    [Fact(Timeout = 60000)]
    public void Constructor_RejectsUnsupportedDistance()
    {
        Assert.Throws<ArgumentException>(() =>
            new KNNPersonalization<double>(distanceMetric: DistanceMetricType.Hamming));
    }

    [Fact(Timeout = 60000)]
    public void FedAGHN_ComputeGradientSimilarity_IsGenericCosine()
    {
        var aghn = new FedAGHNPersonalization<float>(sharedDimension: 2);

        Assert.Equal(0.7071068, (double)aghn.ComputeGradientSimilarity(new[] { 1f, 0f }, new[] { 1f, 1f }), 5);
        Assert.Equal(0f, aghn.ComputeGradientSimilarity(new[] { 0f, 0f }, new[] { 1f, 1f }));
    }

    [Fact(Timeout = 60000)]
    public void FedAGHN_ComputeAdaptiveWeights_ReluNormalizesThenBlendsMomentum()
    {
        var aghn = new FedAGHNPersonalization<double>(sharedDimension: 2, adaptiveWeightMomentum: 0.9);
        var clients = new Dictionary<int, double[]>
        {
            [1] = new[] { 1.0, 0.0 },
            [2] = new[] { 0.0, 1.0 },
            [3] = new[] { 1.0, 1.0 },
        };

        // Similarities 1, 0, 1/sqrt(2) -> weights 1/(1 + 1/sqrt(2)), 0, (1/sqrt(2))/(1 + 1/sqrt(2)).
        var weights = aghn.ComputeAdaptiveWeights(new[] { 1.0, 0.0 }, clients);
        Assert.Equal(0.5857864376, weights[1], Precision);
        Assert.Equal(0.0, weights[2], Precision);
        Assert.Equal(0.4142135624, weights[3], Precision);

        // Client 1 had weight 0 last round; clients 2 and 3 have none, so they keep their current weight.
        var blended = aghn.ComputeAdaptiveWeights(
            new[] { 1.0, 0.0 }, clients, new Dictionary<int, double> { [1] = 0.0 });
        Assert.Equal(0.1238993431, blended[1], Precision);
        Assert.Equal(0.0, blended[2], Precision);
        Assert.Equal(0.8761006569, blended[3], Precision);
    }
}
