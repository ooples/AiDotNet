using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Reference tests for the noisy top-K gating of <see cref="MixtureOfExpertsLayer{T}"/>
/// (Shazeer et al. 2017, §3.2): the gating noise is a fresh standard-normal draw per training step.
/// </summary>
public sealed class MixtureOfExpertsNoisyGatingTests
{
    [Fact]
    public void BatchedTopK_RetainsExactlyTheSelectedExpertsAndNormalizesEachRow()
    {
        var layer = Build(MixtureOfExpertsGatingNoise.LearnedNoiseWeights);
        var weights = new Tensor<double>(new[] { 0.7, 0.2, 0.1, 0.1, 0.3, 0.6 }, new[] { 2, 3 });
        var method = layer.GetType().GetMethod("ApplyTopK", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var (sparse, indices) = ((Tensor<double>, int[,]))method.Invoke(layer, new object[] { weights, 2 })!;
        var expected = new[] { 7.0 / 9, 2.0 / 9, 0, 0, 1.0 / 3, 2.0 / 3 };
        for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(sparse.GetFlat(i) - expected[i]), 0, 1e-9);
        Assert.Equal(0, indices[0, 0]);
        Assert.Equal(2, indices[1, 0]);
    }

    [Fact]
    public void LearnedNoiseWeights_LargeFiniteLogitsDoNotOverflowSoftplus()
    {
        var layer = Build(MixtureOfExpertsGatingNoise.LearnedNoiseWeights);
        var projection = NoiseProjection(layer);
        projection.SetParameters(new Vector<double>(Enumerable.Repeat(1000.0, checked((int)projection.ParameterCount)).ToArray()));
        layer.SetTrainingMode(true);
        var input = new Tensor<double>(new[] { 0.3, 1.2, 0.8, 2.0, 1.1, 0.4, 0.6, 0.9 }, new[] { 2, 4 });
        Assert.All(layer.Forward(input).ToArray(), value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
    }

    [Fact]
    public void TrainingForward_DrawsFreshGatingNoiseEachStep()
    {
        // A Random re-seeded with 42 inside every forward produced the same "noise" each step, so two
        // training forwards over the same input returned identical outputs.
        var experts = new List<ILayer<double>>
        {
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
        };
        var router = new DenseLayer<double>(3);
        var layer = new MixtureOfExpertsLayer<double>(experts, router, [4], [3]);
        layer.SetTrainingMode(true);
        var input = new Tensor<double>([0.3, -1.2, 0.8, 2.0, 1.1, 0.4, -0.6, 0.9], [2, 4]);

        var first = layer.Forward(input);
        var second = layer.Forward(input);

        double maxDifference = 0;
        for (int i = 0; i < first.Length; i++)
        {
            maxDifference = Math.Max(maxDifference, Math.Abs(first.GetFlat(i) - second.GetFlat(i)));
        }

        Assert.True(maxDifference > 1e-9, "Two training forwards drew identical gating noise.");
    }

    [Fact]
    public void EvaluationForward_IsDeterministic()
    {
        // Paper §3.4 / inference: clean logits, no noise.
        var experts = new List<ILayer<double>> { new DenseLayer<double>(3), new DenseLayer<double>(3) };
        var router = new DenseLayer<double>(2);
        var layer = new MixtureOfExpertsLayer<double>(experts, router, [4], [3], topK: 1);
        layer.SetTrainingMode(false);
        var input = new Tensor<double>([0.3, -1.2, 0.8, 2.0], [1, 4]);

        var first = layer.Forward(input);
        var second = layer.Forward(input);

        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first.GetFlat(i), second.GetFlat(i), precision: 12);
        }
    }

    [Fact]
    public void DefaultGating_HasZeroInitializedNoiseWeightsPerExpert()
    {
        // Eq. 4: W_noise maps the input to one noise logit per expert and starts at zero.
        var learned = Build(MixtureOfExpertsGatingNoise.LearnedNoiseWeights);
        var legacy = Build(MixtureOfExpertsGatingNoise.SoftplusOfRouterLogits);

        Assert.Equal(legacy.ParameterCount + (4 * 3), learned.ParameterCount);
        var noiseWeights = NoiseProjection(learned).GetParameters();
        Assert.Equal(4 * 3, noiseWeights.Length);
        for (int i = 0; i < noiseWeights.Length; i++)
        {
            Assert.Equal(0.0, noiseWeights[i]);
        }

        Assert.Null(typeof(MixtureOfExpertsLayer<double>)
            .GetField("_noiseProjection", BindingFlags.NonPublic | BindingFlags.Instance)?
            .GetValue(legacy));
    }

    [Fact]
    public void ConstructorDefault_HasZeroInitializedNoiseWeightsPerExpert()
    {
        // No gatingNoise argument: the constructor's own default must be the paper's learned W_noise
        // (Eq. 4), zero-initialized, one noise logit per expert.
        var experts = new List<ILayer<double>>
        {
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
        };
        var layer = new MixtureOfExpertsLayer<double>(experts, new DenseLayer<double>(3), [4], [3], topK: 2);

        var noiseWeights = NoiseProjection(layer).GetParameters();
        Assert.Equal(4 * 3, noiseWeights.Length);
        for (int i = 0; i < noiseWeights.Length; i++)
        {
            Assert.Equal(0.0, noiseWeights[i]);
        }

        Assert.Equal(
            MixtureOfExpertsGatingNoise.LearnedNoiseWeights.ToString(),
            layer.GetMetadata()["GatingNoise"]);
    }

    [Theory]
    [InlineData("NotAMode")]
    [InlineData("17")]
    [InlineData("")]
    public void Deserialization_RejectsInvalidGatingNoiseMetadata(string rawValue)
    {
        var parameters = new Dictionary<string, object>
        {
            ["NumExperts"] = 3,
            ["TopK"] = 2,
            ["GatingNoise"] = rawValue,
        };

        Assert.Throws<InvalidOperationException>(() =>
            AiDotNet.Helpers.DeserializationHelper.CreateLayerFromType<double>(
                "MixtureOfExpertsLayer`1", [4], [3], parameters));
    }

    [Fact]
    public void Deserialization_MissingGatingNoiseKey_RestoresLegacyRouterLogitNoise()
    {
        var parameters = new Dictionary<string, object> { ["NumExperts"] = 3, ["TopK"] = 2 };

        var layer = AiDotNet.Helpers.DeserializationHelper.CreateLayerFromType<double>(
            "MixtureOfExpertsLayer`1", [4], [3], parameters);

        Assert.Null(typeof(MixtureOfExpertsLayer<double>)
            .GetField("_noiseProjection", BindingFlags.NonPublic | BindingFlags.Instance)?
            .GetValue(layer));
    }

    [Fact]
    public void LearnedNoiseWeights_SetTheNoiseScale()
    {
        // Softplus(x W_noise) with W_noise = -60 and positive inputs is below 1e-70, so the noisy
        // training logits equal the clean ones. The router logits play no part in the scale.
        var layer = Build(MixtureOfExpertsGatingNoise.LearnedNoiseWeights);
        var projection = NoiseProjection(layer);
        var weights = new double[projection.GetParameters().Length];
        for (int i = 0; i < weights.Length; i++) weights[i] = -60.0;
        projection.SetParameters(new Vector<double>(weights));
        var input = new Tensor<double>([0.3, 1.2, 0.8, 2.0, 1.1, 0.4, 0.6, 0.9], [2, 4]);

        layer.SetTrainingMode(false);
        var clean = layer.Forward(input);
        layer.SetTrainingMode(true);
        var noisy = layer.Forward(input);

        for (int i = 0; i < clean.Length; i++)
        {
            Assert.Equal(clean.GetFlat(i), noisy.GetFlat(i), precision: 12);
        }
    }

    private static MixtureOfExpertsLayer<double> Build(MixtureOfExpertsGatingNoise gatingNoise)
    {
        var experts = new List<ILayer<double>>
        {
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
            new DenseLayer<double>(3),
        };
        return new MixtureOfExpertsLayer<double>(experts, new DenseLayer<double>(3), [4], [3], topK: 2,
            gatingNoise: gatingNoise);
    }

    private static DenseLayer<double> NoiseProjection(MixtureOfExpertsLayer<double> layer)
    {
        var field = typeof(MixtureOfExpertsLayer<double>)
            .GetField("_noiseProjection", BindingFlags.NonPublic | BindingFlags.Instance);
        return field?.GetValue(layer) as DenseLayer<double>
            ?? throw new Xunit.Sdk.XunitException("The layer has no W_noise projection.");
    }
}
