using System;
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
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class FreGradNativeOptionsTests
{
    public sealed class DerivedOptions : FreGradOptions { }

    private static FreGrad<double> Create(FreGradOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 8 }, options);

    private static TtsTrainingSample<double> Sample()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < audio.Length; i++) audio[i] = .3 * Math.Sin(.17 * i) + .1 * Math.Cos(.31 * i);
        return new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio };
    }

    [Fact]
    public void PaperDefaults_KeepTheHalfRateUpsamplerAndMelConditionedTwoBandNetwork()
    {
        var options = new FreGradOptions();
        Assert.Equal(256, options.HopSize);
        Assert.Equal(new[] { 16, 8 }, options.UpsampleStrides);
        Assert.Equal((30, 32, 7, 80), (options.NumResLayers, options.ResChannels, options.DilationCycle, options.MelChannels));
        Assert.Equal(80, options.MelMinFrequency); // Primary paper Section 4.1; the released params.py uses zero.
        Assert.Equal(8000, options.MelMaxFrequency);
        Assert.Equal(0, options.DropoutRate);
        Assert.Equal(DiffWaveConditioner.MelSpectrogram, options.Conditioner);
        Assert.Equal(0, options.NumClasses);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(24)]
    [InlineData(32)]
    public void SharedWaveformHop_UpdatesBothRealSubBandResolutions_AndTrains(int hop)
    {
        var options = FreGradPaperTests.Options();
        TtsModelOptions shared = options;
        shared.HopSize = hop;
        Assert.Equal(hop / 2, options.UpsampleStrides.Aggregate(1, (a, b) => a * b));
        Assert.Equal(2, options.UpsampleStrides.Length);
        Assert.All(options.UpsampleStrides, s => Assert.True(s >= 2 && s % 2 == 0));
        var copy = new FreGradOptions(options);
        var restored = JsonConvert.DeserializeObject<FreGradOptions>(JsonConvert.SerializeObject(options))!;
        foreach (var result in new[] { copy, restored })
        {
            Assert.Equal(hop, result.HopSize);
            Assert.Equal(options.UpsampleStrides, result.UpsampleStrides);
            Assert.NotSame(options.UpsampleStrides, result.UpsampleStrides);
        }
        using var model = Create(options);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 4 });
        Assert.Equal(new[] { 1, 2, 4 * hop / 2 }, model.PriorStd(mel).Shape.ToArray());
        var waveform = model.Predict(mel);
        Assert.Equal(4 * hop, waveform.Length);
        Assert.All(waveform.ToArray(), x => Assert.True(double.IsFinite(x)));
        Assert.True(double.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeSubBandStrides_WinOverLegacyWaveformHop_InEitherOrderAndDerivedTypes(bool nativeFirst)
    {
        const string native = "\"UpsampleStrides\":[4,2]";
        const string alias = "\"HopSize\":256";
        string json = "{" + (nativeFirst ? native + "," + alias : alias + "," + native) + "}";
        foreach (var type in new[] { typeof(FreGradOptions), typeof(DerivedOptions) })
            Assert.Equal(16, ((FreGradOptions)JsonConvert.DeserializeObject(json, type)!).HopSize);
        Assert.Equal(128, JsonConvert.DeserializeObject<FreGradOptions>("{\"HopSize\":128}")!.HopSize);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(12)]
    [InlineData(17)]
    public void WaveformHops_ThatCannotKeepBothEvenTransposeStages_AreRejected(int hop)
    {
        var options = FreGradPaperTests.Options();
        Assert.Throws<ArgumentOutOfRangeException>(() => options.HopSize = hop);
    }

    [Theory]
    [InlineData(DiffWaveConditioner.Unconditional)]
    [InlineData(DiffWaveConditioner.ClassLabel)]
    public void UnsupportedNonMelTasks_AreRejectedByTheNativeEnergyPriorNetwork(DiffWaveConditioner mode)
    {
        var options = FreGradPaperTests.Options(); options.Conditioner = mode;
        Assert.Equal("Conditioner", Assert.Throws<ArgumentException>(() => Create(options)).ParamName);
    }

    [Fact]
    public void UnsupportedLabelEmbedding_DropoutAndOddTransposeStride_AreRejected()
    {
        var options = FreGradPaperTests.Options(); options.NumClasses = 10;
        Assert.Equal("NumClasses", Assert.Throws<ArgumentException>(() => Create(options)).ParamName);
        options = FreGradPaperTests.Options(); options.DropoutRate = .1;
        Assert.Equal("DropoutRate", Assert.Throws<ArgumentException>(() => Create(options)).ParamName);
        options = FreGradPaperTests.Options(); options.UpsampleStrides = new[] { 3, 2 };
        Assert.Equal("UpsampleStrides", Assert.Throws<ArgumentException>(() => Create(options)).ParamName);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(.2)]
    public void CachedAdam_UsesConfiguredRateAndDecay_WithoutAnyGradientClipping(double decay)
    {
        var options = FreGradPaperTests.Options(); options.LearningRate = 3e-4; options.WeightDecay = decay;
        using var model = Create(options);
        Assert.True(double.IsFinite(model.Train(Sample())));
        var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(
            typeof(SegmentVocoderBase<double>).GetField("_paperOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(options.LearningRate, optimizer.GetCurrentLearningRate());
        Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
        if (decay > 0)
            Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        Assert.Equal(.9, settings.Beta1); Assert.Equal(.999, settings.Beta2); Assert.Equal(1e-8, settings.Epsilon);
        Assert.False(settings.UseAdaptiveBetas);
        Assert.False(settings.EnableGradientClipping);
        Assert.Equal(0, model.MaxGradNormValue);
    }

    [Fact]
    public void ConfiguredDecay_ChangesRealRecordingUpdates_FromIdenticalInitialWeights()
    {
        var a = FreGradPaperTests.Options(); var b = new FreGradOptions(a) { WeightDecay = .2 };
        using var control = Create(a); using var regularized = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), regularized.GetParameters().ToArray());
        for (int step = 0; step < 2; step++) { control.Train(Sample()); regularized.Train(Sample()); }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(regularized.GetParameters().ToArray()));
    }

    [Fact]
    public void MelFrameCountAndHop_DetermineTheActualOutput_RegardlessOfInheritedUnconditionalLength()
    {
        var a = FreGradPaperTests.Options(); var b = new FreGradOptions(a) { UtteranceSamples = 32000 };
        using var first = Create(a); using var second = Create(b);
        var mel = new Tensor<double>(new[] { 1, a.MelChannels, 4 });
        var wave = first.Predict(mel);
        Assert.Equal(4 * a.HopSize, wave.Length);
        Assert.Equal(wave.ToArray(), second.Predict(mel).ToArray());
    }
}
