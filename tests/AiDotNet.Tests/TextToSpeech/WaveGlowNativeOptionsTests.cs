using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Vocoders;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class WaveGlowNativeOptionsTests
{
    private static WaveGlowOptions Options(bool releasedGeometry = false)
    {
        var options = new WaveGlowOptions
        {
            NumFlows = 4, EarlyOutputEvery = 2, NumWaveNetLayers = 2,
            ResidualChannels = 8, GateChannels = 8, SkipChannels = 4,
        };
        if (!releasedGeometry)
        {
            options.MelChannels = 8; options.FftSize = options.WindowSize = 64; options.HopSize = 16;
            options.SampleRate = 4000; options.MelMaxFrequency = 2000; options.UpsampleKernel = 64;
            options.SegmentSamples = 80;
        }
        return options;
    }

    private static WaveGlow<double> Create(WaveGlowOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 8 }, options);

    private static Tensor<double> Audio(int length)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = .3 * Math.Sin(.17 * i) + .1 * Math.Cos(.31 * i);
        return audio;
    }

    private static TtsTrainingSample<double> Sample(Tensor<double> audio) => new()
    { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio };

    [Fact]
    public void ReleasedCropKeeps16000SamplesAndComputesFeaturesAfterSampleLevelCropping()
    {
        var options = Options(releasedGeometry: true);
        Assert.Equal(16000, options.SegmentSamples);
        var model = Create(options);
        var recording = Audio(16384);
        int start = new Random(7).Next(0, 16384 - 16000 + 1);
        Assert.NotEqual(0, start % options.HopSize);
        var pair = model.PrepareRecording(recording, new Random(7));
        Assert.Equal(16000, pair.Audio.Length);
        Assert.Equal(Enumerable.Range(start, 16000).Select(i => recording[i]).ToArray(), pair.Audio.ToArray());
        Assert.Equal(model.ComputeMel(pair.Audio).ToArray(), pair.Condition.ToArray());
        Assert.Equal(63, pair.Condition.Shape[2]);
        Assert.Equal(16000, model.Encode(pair.Condition, pair.Audio).Z.Length);
        double meanSquare = pair.Audio.Select(v => v * v).Average();
        var objective = (ITrainingObjectiveProvider<double>)model;
        Assert.Equal(meanSquare, objective.EvaluateTrainingObjective(pair.Condition, pair.Audio), 9);
        var before = model.GetParameters().ToArray();
        Assert.True(double.IsFinite(model.Train(Sample(recording))));
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
    }

    [Fact]
    public void ShortRecordingIsZeroPaddedBeforeFeatureExtractionAndRealTraining()
    {
        var options = Options(); var model = Create(options); var recording = Audio(31);
        var pair = model.PrepareRecording(recording, new Random(3));
        Assert.Equal(80, pair.Audio.Length);
        Assert.Equal(recording.ToArray(), pair.Audio.Take(31).ToArray());
        Assert.All(pair.Audio.Skip(31), v => Assert.Equal(0, v));
        Assert.Equal(model.ComputeMel(pair.Audio).ToArray(), pair.Condition.ToArray());
        Assert.Equal(6, pair.Condition.Shape[2]);
        Assert.True(double.IsFinite(model.Train(Sample(recording))));
    }

    [Theory]
    [InlineData(81)]
    [InlineData(95)]
    public void NonalignedCropIsExactAndUnfoldDropsOnlyTheTrailingIncompleteAudioGroup(int samples)
    {
        var options = Options(); options.SegmentSamples = samples;
        var model = Create(options); var recording = Audio(200);
        var pair = model.PrepareRecording(recording, new Random(5));
        Assert.Equal(samples, pair.Audio.Length);
        var (z, _) = model.Encode(pair.Condition, pair.Audio);
        int used = samples / options.GroupSize * options.GroupSize;
        Assert.Equal(used, z.Length);
        var back = model.Decode(pair.Condition, z);
        Assert.Equal(used, back.Length);
        for (int i = 0; i < used; i++) Assert.InRange(Math.Abs(back[i] - pair.Audio[i]), 0, 1e-8);
        double expected = pair.Audio.Take(used).Select(v => v * v).Average();
        Assert.Equal(expected, ((ITrainingObjectiveProvider<double>)model).EvaluateTrainingObjective(pair.Condition, pair.Audio), 9);
        Assert.True(double.IsFinite(model.Train(Sample(recording))));
    }

    [Theory]
    [InlineData(80)]
    [InlineData(81)]
    [InlineData(95)]
    public void SuppliedMelPairRetainsExactRequestedSampleCount(int samples)
    {
        var options = Options(); options.SegmentSamples = samples;
        var model = Create(options); var recording = Audio(256); var mel = model.ComputeMel(recording);
        var pair = model.PrepareAligned(mel, recording, new Random(4));
        Assert.Equal(samples, pair.Audio.Length);
        var before = model.GetParameters().ToArray();
        model.Train(mel, recording);
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
    }

    [Fact]
    public void SuppliedSingleMelFrameNeverSlicesBeyondItsActualConditionCapacity()
    {
        var options = Options(); var model = Create(options); var recording = Audio(256);
        var mel = new Tensor<double>(new[] { 1, options.MelChannels, 1 });
        var pair = model.PrepareAligned(mel, recording, new Random(4));
        Assert.Equal(options.UpsampleKernel, pair.Audio.Length);
        Assert.Equal(1, pair.Condition.Shape[2]);
        model.Train(mel, recording);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.2)]
    public void ActualCachedAdamHonorsConfiguredRateAndDecayWithoutExtraGradientClipping(double decay)
    {
        var options = Options(); options.LearningRate = 3e-4; options.WeightDecay = decay;
        var model = Create(options);
        Assert.True(double.IsFinite(model.Train(Sample(Audio(256)))));
        var optimizer = Assert.IsType<AdamOptimizer<double, Tensor<double>, Tensor<double>>>(
            typeof(SegmentVocoderBase<double>).GetField("_paperOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var actual = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(options.LearningRate, optimizer.GetCurrentLearningRate());
        Assert.Equal(decay > 0, actual.RegularizationExplicitlySet);
        if (decay > 0)
            Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(actual.Regularization).GetOptions().Strength);
        Assert.Equal(.9, actual.Beta1); Assert.Equal(.999, actual.Beta2); Assert.Equal(1e-8, actual.Epsilon);
        Assert.False(actual.UseAdaptiveBetas);
        Assert.False(actual.EnableGradientClipping);
        Assert.Equal(0, model.MaxGradNormValue);
    }

    [Fact]
    public void PositiveDecayChangesActualNativeRecordingUpdates()
    {
        var a = Options(); var b = Options(); b.WeightDecay = .2;
        var control = Create(a); var regularized = Create(b); var sample = Sample(Audio(256));
        Assert.Equal(control.GetParameters().ToArray(), regularized.GetParameters().ToArray());
        for (int i = 0; i < 2; i++) { control.Train(sample); regularized.Train(sample); }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(regularized.GetParameters().ToArray()));
    }
}
