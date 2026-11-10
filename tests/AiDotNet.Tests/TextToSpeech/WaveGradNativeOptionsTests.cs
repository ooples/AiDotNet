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

public class WaveGradNativeOptionsTests
{
    public sealed class DerivedOptions : WaveGradOptions { }

    private static WaveGrad<double> Create(WaveGradOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 8 }, options);

    private static Tensor<double> Audio()
    {
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < audio.Length; i++) audio[i] = .3 * Math.Sin(.17 * i) + .1 * Math.Cos(.31 * i);
        return audio;
    }

    private static TtsTrainingSample<double> Sample() => new()
    { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio() };

    [Fact]
    public void PaperDefaults_KeepAllFiveUBlocksAndTheirPublishedDilations_WithoutDropout()
    {
        var options = new WaveGradOptions();
        Assert.Equal(300, options.HopSize);
        Assert.Equal(new[] { 5, 5, 3, 2, 2 }, options.UpsampleFactors);
        Assert.Equal(new[] { 512, 512, 256, 128, 128 }, options.UpsampleChannels);
        Assert.Equal(0, options.DropoutRate);
        for (int i = 0; i < 5; i++)
            Assert.Equal(i < 3 ? new[] { 1, 2, 4, 8 } : new[] { 1, 2, 1, 2 }, options.UpsampleDilations[i]);
        var large = WaveGradOptions.Large();
        Assert.True(large.RepeatBlocks);
        Assert.Equal(60, large.CropFrames);
        Assert.All(large.UpsampleDilations, d => Assert.Equal(new[] { 1, 2, 4, 8 }, d));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(24)]
    [InlineData(32)]
    public void SharedHop_ConfiguresEveryNativeResolution_AndSurvivesCopiesAndJson(int hop)
    {
        var options = WaveGradPaperTests.Options();
        TtsModelOptions shared = options;
        shared.HopSize = hop;
        Assert.Equal(hop, options.UpsampleFactors.Aggregate(1, (a, b) => a * b));
        Assert.Equal(4, options.UpsampleFactors.Length);
        Assert.All(options.UpsampleFactors, f => Assert.True(f > 0));
        var copy = new WaveGradOptions(options);
        var restored = JsonConvert.DeserializeObject<WaveGradOptions>(JsonConvert.SerializeObject(options))!;
        foreach (var result in new[] { copy, restored })
        {
            Assert.Equal(hop, result.HopSize);
            Assert.Equal(options.UpsampleFactors, result.UpsampleFactors);
            Assert.NotSame(options.UpsampleFactors, result.UpsampleFactors);
        }
        using var model = Create(options);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 4 });
        var output = model.Predict(mel);
        Assert.Equal(4 * hop, output.Length);
        Assert.All(output.ToArray(), x => Assert.True(double.IsFinite(x)));
        Assert.True(double.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeUBlockFactors_WinOverLegacyHop_InEitherOrderAndDerivedTypes(bool factorsFirst)
    {
        const string native = "\"UpsampleFactors\":[5,5,3,2,2]";
        const string alias = "\"HopSize\":256";
        string json = "{" + (factorsFirst ? native + "," + alias : alias + "," + native) + "}";
        foreach (var type in new[] { typeof(WaveGradOptions), typeof(DerivedOptions) })
            Assert.Equal(300, ((WaveGradOptions)JsonConvert.DeserializeObject(json, type)!).HopSize);
        Assert.Equal(150, JsonConvert.DeserializeObject<WaveGradOptions>("{\"HopSize\":150}")!.HopSize);
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(-.1)]
    [InlineData(double.NaN)]
    public void UnsupportedDropout_IsRejectedByTheNativePaperNetwork(double rate)
    {
        var options = WaveGradPaperTests.Options(); options.DropoutRate = rate;
        var error = Assert.Throws<ArgumentException>(() => Create(options));
        Assert.Equal("DropoutRate", error.ParamName);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(.2)]
    public void CachedNativeAdam_UsesConfiguredRateAndDecay_WithOneGlobalGradientClip(double decay)
    {
        var options = WaveGradPaperTests.Options(); options.LearningRate = 3e-4; options.WeightDecay = decay;
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
        Assert.Equal(options.GradientClipNorm, model.MaxGradNormValue);
    }

    [Fact]
    public void ConfiguredDecay_ChangesActualNativeRecordingUpdates_FromIdenticalWeights()
    {
        var a = WaveGradPaperTests.Options(); var b = new WaveGradOptions(a) { WeightDecay = .2 };
        using var control = Create(a); using var regularized = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), regularized.GetParameters().ToArray());
        for (int step = 0; step < 2; step++) { control.Train(Sample()); regularized.Train(Sample()); }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(regularized.GetParameters().ToArray()));
    }
}
