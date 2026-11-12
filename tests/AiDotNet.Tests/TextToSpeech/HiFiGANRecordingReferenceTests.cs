using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Goldens from the unmodified, pinned HiFi-GAN dataset and feature functions.</summary>
public class HiFiGANRecordingReferenceTests
{
    private sealed class SelectedCropRandom : Random
    {
        private readonly int _start;
        public SelectedCropRandom(int start) => _start = start;
        public int Calls { get; private set; }
        public int MaximumStart { get; private set; }
        public override int Next(int minValue, int maxValue)
        {
            Calls++; MaximumStart = maxValue - 1;
            Assert.Equal(0, minValue); Assert.InRange(_start, minValue, maxValue - 1);
            return _start;
        }
    }

    private sealed class Probe : HiFiGAN<double>
    {
        public Probe(HiFiGANOptions o) : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 16) { RandomSeed = 11 }, o) { }
        public (Tensor<double> Mel, Tensor<double> Audio) Pair(Tensor<double> audio, Random random) => RecordingPair(audio, random);
    }

    private static HiFiGANOptions Options(int segment = 128) => new()
    {
        MelChannels = 8, UpsampleRates = [4, 4], UpsampleKernelSizes = [8, 8], UpsampleInitialChannels = 16,
        ResblockKernelSizes = [3], ResblockDilationSizes = [[1, 3]], DiscriminatorPeriods = [2, 3], HopSize = 16,
        FftSize = 64, WindowSize = 64, SampleRate = 4000, MelMaxFrequency = 2000, SegmentSize = segment,
        DiscriminatorWidthDivisor = 16,
    };

    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "hifigan_recording_reference.json")));
    private static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static Tensor<double> Audio(double[] values) => new(new[] { values.Length }, new Vector<double>(values));
    private static Tensor<double> Raw(JsonElement item) => Audio(item.GetProperty("pcm").EnumerateArray().Select(x => x.GetInt32() / 32768.0).ToArray());
    private static double[] Learned(Probe model, string group) =>
        ((IEnumerable<LayerBase<double>>)typeof(GanVocoderBase<double>)
            .GetField(group, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!)
        .SelectMany(layer => layer.GetTrainableParameters()).SelectMany(tensor => tensor.ToArray()).ToArray();
    private static IReadOnlyDictionary<string, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Optimizers(Probe model) =>
        (Dictionary<string, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)typeof(GanVocoderBase<double>)
            .GetField("_optimizers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
    private static void Close(Tensor<double> actual, double[] expected, double tolerance, string description)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(TtsPaperNumerics.IsFinite(error) && error <= tolerance, $"{description}: maximum error {error:R} exceeds {tolerance:R}.");
    }

    [Fact]
    public void PublishedV1Defaults_KeepTheirFullGeneratorAndRecordingGeometry()
    {
        var o = new HiFiGANOptions();
        Assert.Equal((512, 8192, 22050, 1024, 1024, 256, 80),
            (o.UpsampleInitialChannels, o.SegmentSize, o.SampleRate, o.FftSize, o.WindowSize, o.HopSize, o.MelChannels));
        Assert.Equal(new[] { 8, 8, 2, 2 }, o.UpsampleRates);
        Assert.Equal(new[] { 16, 16, 4, 4 }, o.UpsampleKernelSizes);
        Assert.Equal(new[] { 3, 7, 11 }, o.ResblockKernelSizes);
        Assert.Equal((2e-4, .01, .999), (o.LearningRate, o.WeightDecay, o.LearningRateDecay));
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void RecordingPair_MatchesUnmodifiedAuthorNormalization_Crop_Padding_AndFeatures(int index)
    {
        using var document = Reference(); var item = document.RootElement.GetProperty("cases")[index];
        using var model = new Probe(Options()); var random = new SelectedCropRandom(item.GetProperty("start_sample").GetInt32());
        var pair = model.Pair(Raw(item), random);
        Assert.Equal(128, pair.Audio.Length); Assert.Equal(new[] { 1, 8, 8 }, pair.Mel.Shape.ToArray());
        Assert.Equal(item.GetProperty("raw_length").GetInt32() >= 128 ? 1 : 0, random.Calls);
        if (random.Calls > 0) Assert.Equal(item.GetProperty("maximum_start").GetInt32(), random.MaximumStart);
        Close(pair.Audio, Values(item.GetProperty("audio")), 1e-7, "released float32 waveform");
        Close(pair.Mel, Values(item.GetProperty("mel")), 2e-6, "released float32 mel");
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void ExistingFrontend_MatchesReleasedFloat32Window_Filterbank_AndMagnitudeEpsilon(int index)
    {
        using var document = Reference(); var item = document.RootElement.GetProperty("cases")[index];
        using var model = new Probe(Options());
        Close(model.ComputeMel(Audio(Values(item.GetProperty("audio")))), Values(item.GetProperty("mel")), 2e-6, "released mel frontend");
    }

    [Theory]
    [InlineData(0)] [InlineData(2)]
    public void ShortAndLongRecordings_TrainTheActualGeneratorAndDiscriminators(int index)
    {
        using var document = Reference(); using var model = new Probe(Options());
        var generator = Learned(model, "_generatorLayers"); var discriminator = Learned(model, "_discriminatorLayers");
        Assert.NotEmpty(generator); Assert.NotEmpty(discriminator);
        double loss = model.Train(new TtsTrainingSample<double>
        { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[index]) });
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.False(generator.SequenceEqual(Learned(model, "_generatorLayers")));
        Assert.False(discriminator.SequenceEqual(Learned(model, "_discriminatorLayers")));
    }

    [Theory]
    [InlineData("_inputMel", "input_mel_basis")]
    [InlineData("_lossMel", "loss_mel_basis")]
    public void ReleasedFilterbanks_RetainTheirExactFloat32Coefficients(string field, string key)
    {
        using var document = Reference(); using var model = new Probe(Options());
        var frontend = typeof(HiFiGAN<double>).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var basis = (Tensor<double>)frontend.GetType().GetField("_mel", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(frontend)!;
        Assert.Equal(Values(document.RootElement.GetProperty(key)), basis.ToArray());
    }

    [Fact]
    public void SegmentSize_ChangesRealPairedCropsAndTrainingUpdates()
    {
        using var document = Reference(); var audio = Raw(document.RootElement.GetProperty("cases")[2]);
        using var control = new Probe(Options(128)); using var candidate = new Probe(Options(256));
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        var a = control.Pair(audio, new SelectedCropRandom(0)); var b = candidate.Pair(audio, new SelectedCropRandom(0));
        Assert.Equal((128, 8), (a.Audio.Length, a.Mel.Shape[2])); Assert.Equal((256, 16), (b.Audio.Length, b.Mel.Shape[2]));
        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio };
        control.Train(sample); candidate.Train(sample);
        Assert.False(Learned(control, "_generatorLayers").SequenceEqual(Learned(candidate, "_generatorLayers")));
        Assert.False(Learned(control, "_discriminatorLayers").SequenceEqual(Learned(candidate, "_discriminatorLayers")));
    }

    [Theory]
    [InlineData(.01)] [InlineData(.2)]
    public void BothNativeAdamWOptimizers_UseFixedPaperBetasAndConfiguredRateDecay_WithoutClipping(double decay)
    {
        using var document = Reference(); var o = Options(); o.LearningRate = 3e-4; o.WeightDecay = decay;
        using var model = new Probe(o);
        model.Train(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) });
        Assert.Equal(0, model.MaxGradNormValue); Assert.Equal(2, Optimizers(model).Count);
        foreach (var optimizer in Optimizers(model).Values)
        {
            Assert.IsType<AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(optimizer);
            var settings = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.Equal(o.LearningRate, optimizer.GetCurrentLearningRate());
            Assert.Equal((.8, .99, 1e-8), (settings.Beta1, settings.Beta2, settings.Epsilon));
            Assert.False(settings.UseAdaptiveBetas); Assert.False(settings.EnableGradientClipping);
            Assert.Equal(decay, settings.WeightDecay);
        }
    }

    [Fact]
    public void ConfiguredDecay_ChangesBothActualGeneratorAndDiscriminatorUpdates()
    {
        using var document = Reference(); var a = Options(); var b = new HiFiGANOptions(a) { WeightDecay = .2 };
        using var control = new Probe(a); using var candidate = new Probe(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) };
        for (int i = 0; i < 2; i++) { control.Train(sample); candidate.Train(sample); }
        Assert.False(Learned(control, "_generatorLayers").SequenceEqual(Learned(candidate, "_generatorLayers")));
        Assert.False(Learned(control, "_discriminatorLayers").SequenceEqual(Learned(candidate, "_discriminatorLayers")));
    }

    [Fact]
    public void BothEpochSchedulers_MatchTorchExponentialLR_AfterActualUpdates()
    {
        using var document = Reference(); var expected = Values(document.RootElement.GetProperty("epoch_learning_rates"));
        var o = Options(); o.UpdatesPerEpoch = 2; o.LearningRateDecay = .9;
        using var model = new Probe(o);
        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) };
        foreach (double rate in expected)
        {
            model.Train(sample); Assert.Equal(2, Optimizers(model).Count);
            foreach (var optimizer in Optimizers(model).Values) Assert.Equal(rate, optimizer.GetCurrentLearningRate(), 14);
        }
    }
}
