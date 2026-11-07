using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class ZonosOptionContextTests
{
    [Theory]
    [InlineData(8000, 2, 32)]
    [InlineData(16000, 3, 64)]
    public void PairedCodecGeometryRunsNativeEncodingDecodingAndTraining(int rate, int codebooks, int vocabulary)
    {
        var options = ZonosFixture.Options();
        options.SampleRate = options.Codec.SampleRate = rate;
        options.Codebooks = options.Codec.NumQuantizers = codebooks;
        options.Codec.CodebookSize = vocabulary;
        options.EndToken = vocabulary;
        options.MaskToken = vocabulary + 1;
        options.Codec.TargetBandwidthKbps = rate / 8.0 * Math.Log(vocabulary, 2) * codebooks / 1000;
        using var model = Create(options);
        var audio = new Tensor<double>(new[] { 64 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.3 * Math.Sin(i * 0.13);
        var codes = model.EncodeToTokens(audio);
        Assert.Equal(codebooks, codes.Shape[1]);
        Assert.All(codes.ToArray(), code => Assert.InRange(code, 0, vocabulary - 1));
        var decoded = model.DecodeFromTokens(codes);
        Assert.Equal(codes.Shape[0] * model.HopSize, decoded.Length);
        Assert.All(decoded.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.Equal(vocabulary + 2, model.Network.Configuration.CodeVocab);
        Assert.Equal(vocabulary + 1, model.Network.Configuration.HeadVocab);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 })), CodecTokens = codes,
        };
        var before = model.GetParameters().ToArray();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.False(before.SequenceEqual(model.GetParameters()));
    }

    [Theory]
    [InlineData("SampleRate")]
    [InlineData("Codebooks")]
    [InlineData("EndToken")]
    [InlineData("MaskToken")]
    public void UnpairedCodecAndSentinelChangesAreRejected(string option)
    {
        var options = ZonosFixture.Options();
        var property = typeof(ZonosOptions).GetProperty(option)!;
        property.SetValue(options, (int)property.GetValue(options)! + 1);
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    [Theory]
    [InlineData("CtcLoss")]
    [InlineData("DnsmosOverall")]
    [InlineData("VqScore")]
    [InlineData("SpeakerNoised")]
    [InlineData("RepetitionPenalty")]
    [InlineData("RepetitionWindow")]
    [InlineData("UnifiedConfidence")]
    [InlineData("UnifiedQuadratic")]
    public void ContextualOptionsChangeNativeConditioningOrSampledFrames(string option)
    {
        var options = ZonosFixture.Options();
        ZonosFixture.PrepareOptions(options, option);
        var candidate = new ZonosOptions(options);
        var property = typeof(ZonosOptions).GetProperty(option)!;
        var value = property.GetValue(candidate);
        property.SetValue(candidate, value switch
        {
            bool flag => !flag,
            int count => count + 1,
            double scalar => scalar == 0 ? 0.1 : 2 * scalar,
            double[] array => array.Select((element, index) => index == 0 ? element + 0.1 : element).ToArray(),
            _ => throw new InvalidOperationException("Unsupported native context option."),
        });
        using var control = Create(options);
        using var changed = Create(candidate);
        var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var observed = ZonosFixture.Observe(control, input, option);
        Assert.NotNull(observed);
        Assert.NotEqual(observed, ZonosFixture.Observe(changed, input, option));
    }

    [Theory]
    [InlineData("ConditionerDropout")]
    [InlineData("MaxGradNorm")]
    public void LabeledTrainingExercisesConditionerDropoutAndGradientClipping(string option)
    {
        var options = ZonosFixture.Options();
        ZonosFixture.PrepareOptions(options, option);
        var candidate = new ZonosOptions(options);
        if (option == "ConditionerDropout") candidate.ConditionerDropout *= 2;
        else candidate.MaxGradNorm *= 2;
        using var control = Create(options);
        using var changed = Create(candidate);
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        for (int step = 0; step < 2; step++)
        {
            Assert.True(ZonosFixture.TryTrain(control, input, option));
            Assert.True(ZonosFixture.TryTrain(changed, input, option));
        }
        Assert.False(control.GetParameters().SequenceEqual(changed.GetParameters()));
    }

    private static Zonos<double> Create(ZonosOptions options) => new(new NeuralNetworkArchitecture<double>(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    [Fact]
    public void SelectingHybridBuildsMambaLayersWithoutAnExplicitAttentionList()
    {
        var options = ZonosFixture.Options();
        options.Backbone = ZonosBackbone.Hybrid;
        using var model = Create(options);
        Assert.NotNull(model.Network.Layers[0].SelfAttention);
        Assert.NotNull(model.Network.Layers[1].Ssm);
        var codes = new Tensor<double>(new[] { 6, options.Codebooks });
        for (int i = 0; i < codes.Length; i++) codes[i] = i % options.Codec.CodebookSize;
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 1.0, 2.0, 3.0 })), CodecTokens = codes,
        };
        var before = model.GetParameters().ToArray();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
        Assert.False(before.SequenceEqual(model.GetParameters()));
    }

    [Fact]
    public void ExplicitAttentionLayersOverrideTheHybridDefaultPattern()
    {
        var options = ZonosFixture.Options();
        options.Backbone = ZonosBackbone.Hybrid;
        options.AttentionLayers = new[] { 1 };
        using var model = Create(options);
        Assert.NotNull(model.Network.Layers[0].Ssm);
        Assert.NotNull(model.Network.Layers[1].SelfAttention);
    }

    [Fact]
    public void NullSharedSeedRetainsTheArchitectureInitializationFallback()
    {
        var options = ZonosFixture.Options();
        using var implicitSeed = Create(options);
        using var explicitSeed = Create(new ZonosOptions(options) { Seed = 7 });
        Assert.Equal(implicitSeed.GetParameters().ToArray(), explicitSeed.GetParameters().ToArray());
        Assert.Equal(implicitSeed.Codec.GetParameters().ToArray(), explicitSeed.Codec.GetParameters().ToArray());
    }

    [Fact]
    public void InitializationSeedControlsBackboneAndCodecIndependentlyOfSampling()
    {
        var options = ZonosFixture.Options(); options.Seed = 11;
        using var first = Create(options);
        using var repeat = Create(new ZonosOptions(options) { SamplingSeed = 31 });
        using var changed = Create(new ZonosOptions(options) { Seed = 12 });
        Assert.Equal(first.GetParameters().ToArray(), repeat.GetParameters().ToArray());
        Assert.False(first.GetParameters().SequenceEqual(changed.GetParameters()));
        Assert.Equal(first.Codec.GetParameters().ToArray(), repeat.Codec.GetParameters().ToArray());
        Assert.False(first.Codec.GetParameters().SequenceEqual(changed.Codec.GetParameters()));
    }
}
