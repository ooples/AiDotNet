using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
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

public class WaveRNNNativeOptionsTests
{
    public sealed class DerivedOptions : WaveRNNOptions { }
    private sealed class LastCropRandom : Random { public override int Next(int maxValue) => maxValue - 1; }
    private sealed class Probe : WaveRNN<double>
    {
        public Probe(WaveRNNOptions o) : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: o.MelChannels, outputSize: o.HopSize) { RandomSeed = 8 }, o) { }
        public (Tensor<double> Condition, Tensor<double> Audio) Pair(Tensor<double> mel, Tensor<double> audio)
            => TrainingPair(mel, audio, new LastCropRandom());
        public void ApplyPruning(int step) => AfterTrainingStep(step);
    }

    private static WaveRNNOptions Options() => new()
    {
        MelChannels = 4, FftSize = 64, WindowSize = 64, SampleRate = 4000,
        MelMinFrequency = 0, MelMaxFrequency = 2000, UpsampleScales = [4, 4],
        RnnDim = 16, SequenceSamples = 32,
    };

    private static Tensor<double> Audio(int length = 2048)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = .3 * Math.Sin(.17 * i) + .1 * Math.Cos(.31 * i);
        return audio;
    }

    private static TtsTrainingSample<double> Sample() => new() { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio() };
    private static WaveRnnCore<double> Core(WaveRNN<double> model) => (WaveRnnCore<double>)typeof(WaveRNN<double>)
        .GetField("_core", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    [Fact]
    public void PublishedDefaults_Keep896Units_Exact960Samples_AndTheSparseSchedule()
    {
        var o = new WaveRNNOptions();
        Assert.Equal((896, 896, 960, 24000, 300), (o.RnnDim, o.HiddenDim, o.SequenceSamples, o.SampleRate, o.HopSize));
        Assert.Equal(new[] { 5, 5, 4, 3 }, o.UpsampleScales);
        Assert.Equal(80, o.MelChannels);
        Assert.Equal((0.0, 1000, 200000, 500, 1, 1),
            (o.SparsityTarget, o.PruningStartStep, o.PruningSteps, o.PruningInterval, o.PruneBlockRows, o.PruneBlockColumns));
        Assert.Equal((0.0, 4.0, 1e-4, 0.0), (o.DropoutRate, o.MaxGradientNorm, o.LearningRate, o.WeightDecay));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(24)]
    [InlineData(32)]
    public void SharedHop_ReconfiguresAllRealConditioningStages_AndTrains(int hop)
    {
        var o = Options(); TtsModelOptions shared = o; shared.HopSize = hop;
        Assert.Equal(2, o.UpsampleScales.Length); Assert.All(o.UpsampleScales, s => Assert.True(s > 0));
        Assert.Equal(hop, o.UpsampleScales.Aggregate(1, (a, b) => a * b));
        var copy = new WaveRNNOptions(o);
        var restored = JsonConvert.DeserializeObject<WaveRNNOptions>(JsonConvert.SerializeObject(o))!;
        foreach (var candidate in new[] { copy, restored })
        {
            Assert.Equal(hop, candidate.HopSize); Assert.Equal(o.UpsampleScales, candidate.UpsampleScales);
            Assert.NotSame(o.UpsampleScales, candidate.UpsampleScales);
        }
        using var model = new Probe(o);
        Assert.Equal(2 * hop, model.Predict(new Tensor<double>(new[] { 1, 4, 2 })).Length);
        Assert.True(double.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    public void SharedWidth_ControlsTheActualRecurrentStateAndLearning(int width)
    {
        var o = Options(); o.HiddenDim = width;
        using var model = new Probe(o);
        Assert.Equal(width, Core(model).Hidden); Assert.Equal(width, new WaveRNNOptions(o).HiddenDim);
        Assert.True(double.IsFinite(model.Train(Sample())));
        Assert.Throws<ArgumentOutOfRangeException>(() => o.HiddenDim = 3);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeSavedDimensions_WinOverLegacyAliases_InEitherOrder(bool nativeFirst)
    {
        const string native = "\"RnnDim\":16,\"UpsampleScales\":[4,4]";
        const string aliases = "\"HiddenDim\":256,\"HopSize\":300";
        string json = "{" + (nativeFirst ? native + "," + aliases : aliases + "," + native) + "}";
        foreach (var type in new[] { typeof(WaveRNNOptions), typeof(DerivedOptions) })
        {
            var o = (WaveRNNOptions)JsonConvert.DeserializeObject(json, type)!;
            Assert.Equal((16, 16, 16), (o.RnnDim, o.HiddenDim, o.HopSize));
        }
        var legacy = JsonConvert.DeserializeObject<WaveRNNOptions>("{\"HiddenDim\":32,\"HopSize\":128}")!;
        Assert.Equal((32, 128), (legacy.RnnDim, legacy.HopSize));
    }

    [Theory]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(65)]
    [InlineData(960)]
    public void TrainingSequence_KeepsEverySample_IncludingThePartialLastFeatureFrame(int samples)
    {
        var o = Options(); o.SequenceSamples = samples;
        using var model = new Probe(o);
        var audio = Audio(); var mel = model.ComputeMel(audio);
        var pair = model.Pair(mel, audio);
        Assert.Equal(samples, pair.Audio.Length);
        Assert.Equal(1 + (samples - 1) / 16, pair.Condition.Shape[2]);
        int start = (audio.Length - samples) / 16;
        Assert.Equal(audio.ToArray().Skip(start * 16).Take(samples).ToArray(), pair.Audio.ToArray());
        for (int c = 0; c < 4; c++) for (int f = 0; f < pair.Condition.Shape[2]; f++)
            Assert.Equal(mel[0, c, start + f], pair.Condition[0, c, f]);
        Assert.True(double.IsFinite(model.Train(Sample())));
        Assert.True(double.IsFinite(((ITrainingObjectiveProvider<double>)model)
            .EvaluateTrainingObjective(pair.Condition, pair.Audio)));
    }

    [Fact]
    public void Paper960SampleSequence_With300SampleHop_IsNotSilentlyReducedTo900()
    {
        var o = new WaveRNNOptions { RnnDim = 4 };
        using var model = new Probe(o);
        var pair = model.Pair(new Tensor<double>(new[] { 1, 80, 8 }), Audio(2400));
        Assert.Equal(960, pair.Audio.Length); Assert.Equal(4, pair.Condition.Shape[2]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PairedTraining_AcceptsBothFlatAndThreeDimensionalWaveforms(bool shaped)
    {
        var o = Options(); o.SequenceSamples = 33;
        using var model = new Probe(o);
        var samples = Audio(128);
        var audio = shaped ? new Tensor<double>(new[] { 1, 1, 128 }, new Vector<double>(samples.ToArray())) : samples;
        var mel = model.ComputeMel(samples);
        var pair = model.Pair(mel, audio);
        Assert.Equal(33, pair.Audio.Length); Assert.Equal(3, pair.Condition.Shape[2]);
        var before = model.GetParameters().ToArray(); model.Train(mel, audio);
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
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
    [InlineData(0, 0)]
    [InlineData(1, 4)]
    [InlineData(2, 7)]
    [InlineData(3, 7)]
    [InlineData(4, 8)]
    [InlineData(6, 8)]
    public void PaperCubicSchedule_PrunesTheExactLowestMagnitudeWeights_SeparatelyInEachGate(int step, int zeros)
    {
        var o = Options(); o.RnnDim = 4; o.SparsityTarget = .5; o.PruningStartStep = 0;
        o.PruningSteps = 4; o.PruningInterval = 1;
        using var model = new Probe(o);
        var weights = Core(model).Recurrent.GetTrainableParameters()[0];
        for (int gate = 0; gate < 3; gate++) for (int i = 0; i < 16; i++)
            weights[gate * 16 + i] = (gate + 1) * 100 + i + 1;
        model.ApplyPruning(step);
        var actual = Core(model).Recurrent.Kernel();
        for (int gate = 0; gate < 3; gate++) for (int i = 0; i < 16; i++)
            Assert.Equal(i < zeros ? 0 : (gate + 1) * 100 + i + 1, actual[gate * 16 + i]);
        Assert.Equal(zeros / 16.0, model.RecurrentSparsity);
    }

    [Theory]
    [InlineData(4, 4)]
    [InlineData(16, 1)]
    public void PublishedBlockShapes_PruneWholeBlocks_AndStayMaskedAfterRealTraining(int rows, int columns)
    {
        var o = Options(); o.SparsityTarget = .5; o.PruningStartStep = 0; o.PruningSteps = 2;
        o.PruningInterval = 1; o.PruneBlockRows = rows; o.PruneBlockColumns = columns;
        using var model = new Probe(o);
        for (int step = 0; step < 4; step++) Assert.True(double.IsFinite(model.Train(Sample())));
        Assert.Equal(.5, model.RecurrentSparsity);
        var kernel = Core(model).Recurrent.Kernel();
        for (int gate = 0; gate < 3; gate++) for (int r = 0; r < 16; r += rows) for (int c = 0; c < 16; c += columns)
        {
            bool zero = kernel[(gate * 16 + r) * 16 + c] == 0;
            for (int dr = 0; dr < rows; dr++) for (int dc = 0; dc < columns; dc++)
                Assert.Equal(zero, kernel[(gate * 16 + r + dr) * 16 + c + dc] == 0);
        }
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(.2)]
    public void NativeAdam_UsesConfiguredRateAndDecay_WithOneGlobalClip(double decay)
    {
        var o = Options(); o.LearningRate = 3e-4; o.WeightDecay = decay;
        using var model = new Probe(o); model.Train(Sample());
        var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(typeof(SegmentVocoderBase<double>)
            .GetField("_paperOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(o.LearningRate, optimizer.GetCurrentLearningRate());
        Assert.Equal((.9, .999, 1e-8), (settings.Beta1, settings.Beta2, settings.Epsilon));
        Assert.False(settings.UseAdaptiveBetas); Assert.False(settings.EnableGradientClipping);
        Assert.Equal(4, model.MaxGradNormValue);
        if (decay > 0) Assert.Equal(decay,
            Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
    }

    [Fact]
    public void ConfiguredDecay_ChangesRealRecurrentUpdates()
    {
        var a = Options(); var b = new WaveRNNOptions(a) { WeightDecay = .2 };
        using var control = new Probe(a); using var candidate = new Probe(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        for (int i = 0; i < 2; i++) { control.Train(Sample()); candidate.Train(Sample()); }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(candidate.GetParameters().ToArray()));
    }
}
