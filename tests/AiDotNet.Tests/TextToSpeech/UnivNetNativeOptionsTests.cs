using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class UnivNetNativeOptionsTests
{
    public sealed class DerivedOptions : UnivNetOptions { }

    private static UnivNet<double> Create(UnivNetOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 4 }, options);

    private static Tensor<double> Audio(int length = 4096)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = ((i * 37) % 211 - 105) / 256.0 + (i % 7 - 3) / 64.0;
        return audio;
    }

    private static TtsTrainingSample<double> Sample() => new()
    { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio() };

    [Fact]
    public void PublishedDefaults_KeepAllThreeLvcStacksAndThePaperTrainingRecipe()
    {
        var options = new UnivNetOptions();
        Assert.Equal(new[] { 8, 8, 4 }, options.UpsampleRates);
        Assert.Equal(new[] { 1, 3, 9, 27 }, options.Dilations);
        Assert.Equal((256, 16384, 64, 32, 100), (options.HopSize, options.SegmentSize, options.NoiseDim, options.ChannelSize, options.MelChannels));
        Assert.Equal(200000, options.PretrainingSteps);
        Assert.Equal((1e-4, .5, .9, 2.5), (options.LearningRate, options.Beta1, options.Beta2, options.StftLossWeight));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(24)]
    [InlineData(32)]
    [InlineData(45)]
    public void SharedHop_ChangesTheActualLvcResolutions_AndTrainsWithoutRemovingStages(int hop)
    {
        var options = UnivNetPaperTests.Options();
        TtsModelOptions shared = options;
        shared.HopSize = hop;
        options.SegmentSize = 8 * hop;
        Assert.Equal(hop, options.UpsampleRates.Aggregate(1, (a, b) => a * b));
        Assert.Equal(2, options.UpsampleRates.Length);
        Assert.All(options.UpsampleRates, s => Assert.True(s >= 2));
        var copy = new UnivNetOptions(options);
        var restored = JsonConvert.DeserializeObject<UnivNetOptions>(JsonConvert.SerializeObject(options))!;
        foreach (var result in new[] { copy, restored })
        {
            Assert.Equal(hop, result.HopSize);
            Assert.Equal(options.UpsampleRates, result.UpsampleRates);
            Assert.NotSame(options.UpsampleRates, result.UpsampleRates);
        }
        using var model = Create(options);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 5 });
        var output = model.Predict(mel);
        Assert.Equal(5 * hop, output.Length);
        Assert.All(output.ToArray(), x => Assert.True(double.IsFinite(x)));
        Assert.True(double.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(257)]
    public void ImpossibleHop_IsRejectedInsteadOfDroppingAPublishedStage(int hop)
    {
        var options = new UnivNetOptions();
        Assert.Throws<ArgumentOutOfRangeException>(() => options.HopSize = hop);
        Assert.Equal(new[] { 8, 8, 4 }, options.UpsampleRates);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CanonicalNativeStrides_WinOverLegacySharedHop_InEitherOrderAndDerivedTypes(bool nativeFirst)
    {
        const string native = "\"UpsampleRates\":[4,4]";
        const string alias = "\"HopSize\":256";
        string json = "{" + (nativeFirst ? native + "," + alias : alias + "," + native) + "}";
        foreach (var type in new[] { typeof(UnivNetOptions), typeof(DerivedOptions) })
            Assert.Equal(16, ((UnivNetOptions)JsonConvert.DeserializeObject(json, type)!).HopSize);
        Assert.Equal(128, JsonConvert.DeserializeObject<UnivNetOptions>("{\"HopSize\":128}")!.HopSize);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(31)]
    [InlineData(65)]
    public void SegmentSize_MustBeAPositiveWholeNumberOfNativeHops(int size)
    {
        var options = UnivNetPaperTests.Options(); options.SegmentSize = size;
        Assert.Equal("SegmentSize", Assert.Throws<ArgumentException>(() => Create(options)).ParamName);
    }

    private sealed class SelectedFrameRandom(int frame, int maximum) : Random
    {
        public override int Next(int minValue, int maxValue)
        {
            Assert.Equal(0, minValue);
            Assert.Equal(maximum + 1, maxValue);
            Assert.InRange(frame, minValue, maxValue - 1);
            return frame;
        }
    }

    [Fact]
    public void RecordingPair_MatchesUnmodifiedReleasedDatasetAndStft_OnLongAndShortInputs()
    {
        var reference = JObject.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "univnet_recording_reference.json")));
        Assert.Equal("9bb2b54838bb6d7ce767131cc7b8b61198bc7558", (string)reference["commit"]!);
        Assert.Equal("37f17420eff07ce8449c6e837715c47ce3013cdb", (string)reference["git_blobs"]!["datasets/dataloader.py"]!);
        Assert.Equal("ca588d0c4a568b433b300a7b362b4353c709b0cd", (string)reference["git_blobs"]!["utils/stft.py"]!);
        foreach (var c in reference["cases"]!)
        {
            var options = UnivNetPaperTests.Options(); options.SegmentSize = (int)c["segment_samples"]!;
            using var model = Create(options);
            // The released Python loader chose this frame using its real seeded RNG. Match that selection,
            // independently of the different Python/.NET RNG algorithms; all signal processing remains native.
            var pair = model.PrepareRecording(Audio((int)c["raw_length"]!),
                new SelectedFrameRandom((int)c["start_frame"]!, (int)c["maximum_start"]!));
            Assert.Equal(c["audio"]!.Select(x => (double)x).ToArray(), pair.Audio.ToArray());
            Assert.Equal(new[] { 1, options.MelChannels, options.SegmentSize / options.HopSize }, pair.Mel.Shape.ToArray());
            var expected = c["mel"]!.Select(x => (double)x).ToArray();
            var actual = pair.Mel.ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 1e-10);
            if ((string)c["name"]! == "long_context_crop")
                Assert.True(actual.Zip(model.ComputeMel(pair.Audio).ToArray(), (a, b) => Math.Abs(a - b)).Max() > 1e-5,
                    "The training features retain the whole recording's boundary context.");
        }
    }

    [Theory]
    [InlineData(0.0, .5, .9)]
    [InlineData(.2, .6, .8)]
    public void BothCachedAdams_HonorConfiguredRateBetasAndDecay_WithoutClipping(double decay, double beta1, double beta2)
    {
        var options = UnivNetPaperTests.Options(pretraining: 0);
        options.SegmentSize = 256; options.LearningRate = 3e-4; options.WeightDecay = decay;
        options.Beta1 = beta1; options.Beta2 = beta2;
        using var model = Create(options);
        Assert.True(double.IsFinite(model.Train(Sample())));
        var optimizers = (IDictionary)typeof(GanVocoderBase<double>)
            .GetField("_optimizers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        Assert.Equal(2, optimizers.Count);
        foreach (string group in new[] { "generator", "discriminator" })
        {
            var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(optimizers[group]);
            var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.Equal(options.LearningRate, optimizer.GetCurrentLearningRate());
            Assert.Equal(beta1, settings.Beta1); Assert.Equal(beta2, settings.Beta2); Assert.Equal(1e-8, settings.Epsilon);
            Assert.False(settings.UseAdaptiveBetas); Assert.False(settings.EnableGradientClipping);
            Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
            if (decay > 0)
                Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        }
        Assert.Equal(0, model.MaxGradNormValue);
    }

    [Theory]
    [InlineData("SegmentSize")]
    [InlineData("WeightDecay")]
    public void ConfiguredCropAndDecay_ChangeRealRecordingGeneratorUpdates_WhilePretrainingFreezesCritics(string option)
    {
        var a = UnivNetPaperTests.Options(); a.SegmentSize = 256;
        var b = new UnivNetOptions(a);
        if (option == "SegmentSize") b.SegmentSize = 512;
        else b.WeightDecay = .2;
        using var control = Create(a); using var candidate = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        var critics = GanVocoderWeights.Of(control.DiscriminatorLayers);
        for (int step = 0; step < 2; step++) { control.Train(Sample()); candidate.Train(Sample()); }
        Assert.False(GanVocoderWeights.Of(control.GeneratorLayers).SequenceEqual(GanVocoderWeights.Of(candidate.GeneratorLayers)));
        Assert.Equal(critics, GanVocoderWeights.Of(control.DiscriminatorLayers));
        Assert.Equal(critics, GanVocoderWeights.Of(candidate.DiscriminatorLayers));
    }
}
