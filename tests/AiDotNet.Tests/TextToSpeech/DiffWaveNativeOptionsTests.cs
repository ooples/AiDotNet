using System;
using System.Linq;
using System.Reflection;
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

public class DiffWaveNativeOptionsTests
{
    public sealed class DerivedOptions : DiffWaveOptions { }

    private static DiffWave<double> Create(DiffWaveOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 8 }, options);

    private static TtsTrainingSample<double> Sample(int? label = null)
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < audio.Length; i++) audio[i] = .3 * Math.Sin(.17 * i) + .1 * Math.Cos(.31 * i);
        return new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio, ClassLabel = label };
    }

    private static DiffWaveOptions Task(DiffWaveConditioner conditioner)
    {
        var o = DiffWavePaperTests.Options(fast: true); o.Conditioner = conditioner;
        o.NumClasses = conditioner == DiffWaveConditioner.ClassLabel ? 2 : 0; o.UtteranceSamples = 128;
        return o;
    }

    [Fact]
    public void PublishedPresets_KeepAllThreeRealTasksAndTheirActualWidths()
    {
        var vocoder = new DiffWaveOptions();
        Assert.Equal((64, 64, 30, 10, 50, 22050, 256),
            (vocoder.HiddenDim, vocoder.ResChannels, vocoder.NumResLayers, vocoder.DilationCycle,
                vocoder.NoiseSchedule.Length, vocoder.SampleRate, vocoder.HopSize));
        foreach (var o in new[] { DiffWaveOptions.Unconditional(), DiffWaveOptions.ClassConditional(10) })
        {
            Assert.Equal((256, 256, 36, 12, 200, 16000, 16000),
                (o.HiddenDim, o.ResChannels, o.NumResLayers, o.DilationCycle, o.NoiseSchedule.Length, o.SampleRate, o.UtteranceSamples));
            Assert.Equal(.02, o.NoiseSchedule[^1]); Assert.Equal(0, o.DropoutRate);
        }
        Assert.Equal(10, DiffWaveOptions.ClassConditional(10).NumClasses);
        Assert.Equal(128, DiffWaveNetwork<double>.LabelDimension);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(24)]
    [InlineData(32)]
    [InlineData(64)]
    public void SharedHop_ReconfiguresBothActualMelUpsamplingStages_AndTrains(int hop)
    {
        var o = DiffWavePaperTests.Options(fast: true); TtsModelOptions shared = o; shared.HopSize = hop;
        Assert.Equal(2, o.UpsampleStrides.Length);
        Assert.All(o.UpsampleStrides, s => Assert.True(s >= 2 && s % 2 == 0));
        Assert.Equal(hop, o.UpsampleStrides.Aggregate(1, (a, b) => a * b));
        var copy = new DiffWaveOptions(o);
        var restored = JsonConvert.DeserializeObject<DiffWaveOptions>(JsonConvert.SerializeObject(o))!;
        foreach (var result in new[] { copy, restored })
        {
            Assert.Equal(hop, result.HopSize); Assert.Equal(o.UpsampleStrides, result.UpsampleStrides);
            Assert.NotSame(o.UpsampleStrides, result.UpsampleStrides);
        }
        using var model = Create(o);
        Assert.Equal(4 * hop, model.Predict(new Tensor<double>(new[] { 1, o.MelChannels, 4 })).Length);
        Assert.True(double.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SavedNativeWidthAndStrides_WinOverLegacyAliases_InEitherOrder(bool nativeFirst)
    {
        const string native = "\"ResChannels\":8,\"UpsampleStrides\":[4,4]";
        const string aliases = "\"HiddenDim\":256,\"HopSize\":256";
        string json = "{" + (nativeFirst ? native + "," + aliases : aliases + "," + native) + "}";
        foreach (var type in new[] { typeof(DiffWaveOptions), typeof(DerivedOptions) })
        {
            var o = (DiffWaveOptions)JsonConvert.DeserializeObject(json, type)!;
            Assert.Equal((8, 8, 16), (o.ResChannels, o.HiddenDim, o.HopSize));
        }
        var legacy = JsonConvert.DeserializeObject<DiffWaveOptions>("{\"HiddenDim\":12,\"HopSize\":128}")!;
        Assert.Equal((12, 128), (legacy.ResChannels, legacy.HopSize));
    }

    [Theory]
    [InlineData(DiffWaveConditioner.MelSpectrogram)]
    [InlineData(DiffWaveConditioner.Unconditional)]
    [InlineData(DiffWaveConditioner.ClassLabel)]
    public void SharedWidth_IsTheRealResidualWidth_InEveryPublishedTask(DiffWaveConditioner conditioner)
    {
        var a = Task(conditioner); var b = new DiffWaveOptions(a); b.HiddenDim = 12;
        Assert.Equal(12, b.ResChannels);
        Assert.Equal(12, new DiffWaveOptions(b).HiddenDim);
        using var control = Create(a); using var wider = Create(b);
        Assert.True(wider.GetParameters().Length > control.GetParameters().Length);
        Assert.True(double.IsFinite(wider.Train(Sample(conditioner == DiffWaveConditioner.ClassLabel ? 0 : null))));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(257)]
    public void InvalidMelGeometry_IsRejectedWithoutDroppingStages(int hop)
    {
        var o = new DiffWaveOptions(); Assert.Throws<ArgumentOutOfRangeException>(() => o.HopSize = hop);
        Assert.Equal(new[] { 16, 16 }, o.UpsampleStrides);
        Assert.Throws<ArgumentOutOfRangeException>(() => o.HiddenDim = 0);
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(-.1)]
    [InlineData(double.NaN)]
    public void NonpaperDropout_IsRejectedByTheActualNetworkConstructor(double dropout)
    {
        foreach (var task in new[] { DiffWaveConditioner.MelSpectrogram, DiffWaveConditioner.Unconditional, DiffWaveConditioner.ClassLabel })
        {
            var o = Task(task); o.DropoutRate = dropout;
            Assert.Equal("DropoutRate", Assert.Throws<ArgumentException>(() => Create(o)).ParamName);
        }
    }

    [Theory]
    [InlineData(DiffWaveConditioner.Unconditional)]
    [InlineData(DiffWaveConditioner.ClassLabel)]
    public void UtteranceLength_ControlsRealReverseDiffusionOutsideTheVocoder(DiffWaveConditioner task)
    {
        var a = Task(task); var b = new DiffWaveOptions(a) { UtteranceSamples = 192 };
        using var control = Create(a); using var candidate = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        Assert.Equal(128, control.Generate(task == DiffWaveConditioner.ClassLabel ? 0 : null).Length);
        var waveform = candidate.Generate(task == DiffWaveConditioner.ClassLabel ? 0 : null);
        Assert.Equal(192, waveform.Length); Assert.All(waveform.ToArray(), x => Assert.True(double.IsFinite(x)));
        Assert.True(double.IsFinite(candidate.Train(Sample(task == DiffWaveConditioner.ClassLabel ? 0 : null))));
    }

    [Fact]
    public void ClassCount_AddsReal128WideEmbeddingColumns_AndChosenLabelsTrain()
    {
        var a = Task(DiffWaveConditioner.ClassLabel); var b = new DiffWaveOptions(a) { NumClasses = 4 };
        using var control = Create(a); using var expanded = Create(b);
        // The flat state vector also contains persistent buffers. Count actual trainable tensors.
        Assert.Equal(2 * 128,
            expanded.Layers.Cast<LayerBase<double>>().Sum(layer => layer.GetTrainableParameters().Sum(tensor => tensor.Length)) -
            control.Layers.Cast<LayerBase<double>>().Sum(layer => layer.GetTrainableParameters().Sum(tensor => tensor.Length)));
        var network = (DiffWaveNetwork<double>)typeof(DiffWave<double>)
            .GetField("_network", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(control)!;
        var embedding = (NormedConv1DLayer<double>)typeof(DiffWaveNetwork<double>)
            .GetField("_labelEmbedding", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(network)!;
        var weights = Assert.Single(embedding.GetTrainableParameters());
        Assert.Equal(new[] { 128, 2, 1, 1 }, weights.Shape);
        var before = weights.ToArray();
        for (int i = 0; i < 3; i++) control.Train(Sample(0));
        var after = Assert.Single(embedding.GetTrainableParameters()).ToArray();
        Assert.Contains(Enumerable.Range(0, 128), row => before[row * 2] != after[row * 2]);
        for (int row = 0; row < 128; row++) Assert.Equal(before[row * 2 + 1], after[row * 2 + 1]);
        Assert.False(control.Generate(0).ToArray().SequenceEqual(control.Generate(1).ToArray()));
        Assert.ThrowsAny<ArgumentException>(() => control.Generate(2));
    }

    [Theory]
    [InlineData(DiffWaveConditioner.MelSpectrogram, 0.0)]
    [InlineData(DiffWaveConditioner.Unconditional, .2)]
    [InlineData(DiffWaveConditioner.ClassLabel, .2)]
    public void CachedAdam_UsesThePaperRecipeAndConfiguredDecay_ForEveryTask(DiffWaveConditioner task, double decay)
    {
        var o = Task(task); o.WeightDecay = decay; o.LearningRate = 3e-4;
        using var model = Create(o); Assert.True(double.IsFinite(model.Train(Sample(task == DiffWaveConditioner.ClassLabel ? 0 : null))));
        var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(typeof(SegmentVocoderBase<double>)
            .GetField("_paperOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(o.LearningRate, optimizer.GetCurrentLearningRate());
        Assert.Equal((.9, .999, 1e-8), (settings.Beta1, settings.Beta2, settings.Epsilon));
        Assert.False(settings.UseAdaptiveBetas); Assert.False(settings.EnableGradientClipping);
        Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
        if (decay > 0) Assert.Equal(decay,
            Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        Assert.Equal(0, model.MaxGradNormValue);
    }

    [Theory]
    [InlineData(DiffWaveConditioner.MelSpectrogram)]
    [InlineData(DiffWaveConditioner.Unconditional)]
    [InlineData(DiffWaveConditioner.ClassLabel)]
    public void ConfiguredDecay_ChangesActualDenoiserUpdates_InAllThreeTasks(DiffWaveConditioner task)
    {
        var a = Task(task); var b = new DiffWaveOptions(a) { WeightDecay = .2 };
        using var control = Create(a); using var candidate = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        for (int i = 0; i < 2; i++)
        {
            control.Train(Sample(task == DiffWaveConditioner.ClassLabel ? 0 : null));
            candidate.Train(Sample(task == DiffWaveConditioner.ClassLabel ? 0 : null));
        }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(candidate.GetParameters().ToArray()));
    }
}
