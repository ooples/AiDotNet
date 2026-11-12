using System;
using System.Linq;
using System.IO;
using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class WaveNetNativeOptionsTests
{
    public sealed class DerivedOptions : WaveNetOptions { }
    private sealed class FirstCrop : Random { public override int Next(int minValue, int maxValue) => minValue; }
    private sealed class Probe : WaveNet<double>
    {
        public Probe(WaveNetOptions o) : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: o.MelChannels, outputSize: o.HopSize) { RandomSeed = 8 }, o) { }
        public (Tensor<double> Condition, Tensor<double> Audio) Pair(Tensor<double> audio)
            => RecordingPair(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio }, audio, new FirstCrop());
    }

    private static WaveNetOptions Options() => new()
    {
        MelChannels = 4, FftSize = 64, WindowSize = 64, SampleRate = 4000,
        MelMinFrequency = 0, MelMaxFrequency = 2000, UpsampleScales = [4, 4],
        ResidualChannels = 4, GateChannels = 4, SkipChannels = 4,
        NumDilatedLayers = 4, DilationCycle = 2, MuLawLevels = 16, SegmentSamples = 64,
    };

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 2048 });
        for (int i = 0; i < audio.Length; i++) audio[i] = .3 * Math.Sin(.17 * i) + .1 * Math.Cos(.31 * i);
        return audio;
    }

    private static TtsTrainingSample<double> Sample() => new() { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio() };
    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> Optimizer(Probe model)
        => Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(typeof(SegmentVocoderBase<double>)
            .GetField("_paperOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));

    [Fact]
    public void Defaults_KeepThePaperArchitectureAndExplicitReferenceRecipe()
    {
        var o = new WaveNetOptions();
        Assert.Equal((128, 128, 128, 128, 256), (o.HiddenDim, o.ResidualChannels, o.GateChannels, o.SkipChannels, o.MuLawLevels));
        Assert.Equal((30, 10, 2, 10240), (o.NumDilatedLayers, o.DilationCycle, o.KernelSize, o.SegmentSamples));
        Assert.Equal((22050, 80, 1024, 1024, 256), (o.SampleRate, o.MelChannels, o.FftSize, o.WindowSize, o.HopSize));
        Assert.Equal(new[] { 4, 4, 4, 4 }, o.UpsampleScales);
        Assert.Equal((0.0, 1e-3, 0.0, 200000), (o.DropoutRate, o.LearningRate, o.WeightDecay, o.LearningRateHalvingSteps));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(24)]
    [InlineData(32)]
    public void SharedHop_ReconfiguresRealConditioningStages_AndTrains(int hop)
    {
        var o = Options(); TtsModelOptions shared = o; shared.HopSize = hop;
        Assert.Equal(2, o.UpsampleScales.Length); Assert.All(o.UpsampleScales, s => Assert.True(s > 0));
        Assert.Equal(hop, o.UpsampleScales.Aggregate(1, (a, b) => a * b));
        foreach (var candidate in new[] { new WaveNetOptions(o), JsonConvert.DeserializeObject<WaveNetOptions>(JsonConvert.SerializeObject(o))! })
        {
            Assert.Equal(hop, candidate.HopSize); Assert.Equal(o.UpsampleScales, candidate.UpsampleScales);
            Assert.NotSame(o.UpsampleScales, candidate.UpsampleScales);
        }
        using var model = new Probe(o);
        Assert.Equal(2 * hop, model.Predict(new Tensor<double>(new[] { 1, 4, 2 })).Length);
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(16)]
    public void SharedWidth_ChangesTheActualResidualConvolution(int width)
    {
        var o = Options(); o.HiddenDim = width;
        using var model = new Probe(o);
        var network = (WaveNetNetwork<double>)typeof(WaveNet<double>).GetField("_network", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var input = (NormedConv1DLayer<double>)network.Layers[o.UpsampleScales.Length];
        Assert.Equal(new[] { width, o.MuLawLevels, 1, o.KernelSize }, input.GetTrainableParameters().First().Shape.ToArray());
        Assert.Equal(width, new WaveNetOptions(o).HiddenDim);
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeSavedDimensions_WinOverLegacyAliases_InEitherOrder(bool nativeFirst)
    {
        const string native = "\"ResidualChannels\":4,\"UpsampleScales\":[4,4]";
        const string aliases = "\"HiddenDim\":256,\"HopSize\":256";
        string json = "{" + (nativeFirst ? native + "," + aliases : aliases + "," + native) + "}";
        foreach (var type in new[] { typeof(WaveNetOptions), typeof(DerivedOptions) })
        {
            var o = (WaveNetOptions)JsonConvert.DeserializeObject(json, type)!;
            Assert.Equal((4, 4, 16), (o.ResidualChannels, o.HiddenDim, o.HopSize));
        }
        var legacy = JsonConvert.DeserializeObject<WaveNetOptions>("{\"HiddenDim\":8,\"HopSize\":128}")!;
        Assert.Equal((8, 128), (legacy.ResidualChannels, legacy.HopSize));
    }

    [Fact]
    public void RecordingSegment_ChangesRealTargetsFeaturesAndLearning()
    {
        var a = Options(); var b = new WaveNetOptions(a) { SegmentSamples = 128 };
        using var control = new Probe(a); using var candidate = new Probe(b);
        var audio = Audio(); var fullMel = control.ComputeMel(audio);
        var small = control.Pair(audio); var large = candidate.Pair(audio);
        Assert.Equal((64, 4), (small.Audio.Length, small.Condition.Shape[2]));
        Assert.Equal((128, 8), (large.Audio.Length, large.Condition.Shape[2]));
        Assert.Equal(audio.ToArray().Take(64), small.Audio.ToArray());
        Assert.Equal(audio.ToArray().Take(128), large.Audio.ToArray());
        for (int c = 0; c < 4; c++) for (int t = 0; t < 8; t++) Assert.Equal(fullMel[0, c, t], large.Condition[0, c, t]);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        control.Train(Sample()); candidate.Train(Sample());
        Assert.False(control.GetParameters().ToArray().SequenceEqual(candidate.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(-.1)]
    [InlineData(double.NaN)]
    public void NonpaperDropout_IsRejectedByTheActualConstructor(double dropout)
    {
        var o = Options(); o.DropoutRate = dropout;
        Assert.Equal("DropoutRate", Assert.Throws<ArgumentException>(() => new Probe(o)).ParamName);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(.2)]
    public void NativeAdam_UsesConfiguredRateAndDecay_WithoutExtraClipping(double decay)
    {
        var o = Options(); o.LearningRate = 3e-4; o.WeightDecay = decay;
        using var model = new Probe(o); model.Train(Sample());
        var optimizer = Optimizer(model);
        var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(o.LearningRate, optimizer.GetCurrentLearningRate());
        Assert.Equal((.9, .999, 1e-8), (settings.Beta1, settings.Beta2, settings.Epsilon));
        Assert.False(settings.UseAdaptiveBetas, "The reference Adam uses fixed betas.");
        Assert.False(settings.EnableGradientClipping, "The selected reference recipe disables clipping.");
        Assert.Equal(0, model.MaxGradNormValue);
        if (decay > 0) Assert.Equal(decay,
            Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
    }

    [Fact]
    public void ConfiguredDecay_ChangesRealUpdates()
    {
        var a = Options(); var b = new WaveNetOptions(a) { WeightDecay = .2 };
        using var control = new Probe(a); using var candidate = new Probe(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        for (int i = 0; i < 2; i++) { control.Train(Sample()); candidate.Train(Sample()); }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(candidate.GetParameters().ToArray()));
    }

    [Fact]
    public void ActualUpdateSchedule_MatchesIndependentTorchStepLR()
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "wavenet_equations_reference.json")));
        var o = Options(); o.LearningRateHalvingSteps = 2;
        using var model = new Probe(o);
        foreach (var rate in document.RootElement.GetProperty("learning_rates_after_updates").EnumerateArray())
        {
            model.Train(Sample());
            Assert.Equal(rate.GetDouble(), Optimizer(model).GetCurrentLearningRate(), 14);
        }
    }
}
