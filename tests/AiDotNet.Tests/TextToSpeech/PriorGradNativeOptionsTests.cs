using System;
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

public class PriorGradNativeOptionsTests
{
    public sealed class DerivedOptions : PriorGradOptions { }

    private static PriorGrad<double> Create(PriorGradOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 8 }, options);

    private static Tensor<double> Audio(int length = 512, int divisor = 1)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = (((i * 37) % 211 - 105) / 256.0 + (i % 7 - 3) / 64.0) / divisor;
        return audio;
    }

    private static TtsTrainingSample<double> Sample(int length = 512) => new()
    { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio(length) };

    private static JObject Reference() => JObject.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "priorgrad_recording_reference.json")));

    [Fact]
    public void PublishedDefaults_KeepTheMelEnergyPriorAndBaseDenoiser()
    {
        var o = new PriorGradOptions();
        Assert.Equal((22050, 80, 1024, 1024, 256, 64, 64, 30, 10),
            (o.SampleRate, o.MelChannels, o.FftSize, o.WindowSize, o.HopSize, o.HiddenDim, o.ResChannels, o.NumResLayers, o.DilationCycle));
        Assert.Equal(new[] { 16, 16 }, o.UpsampleStrides);
        Assert.Equal((80.0, 7600.0, .1, 4.0, 2e-4, 0.0),
            (o.MelMinFrequency, o.MelMaxFrequency, o.MinStd, o.EnergyMax, o.LearningRate, o.DropoutRate));
        Assert.Equal(50, o.NoiseSchedule.Length);
        Assert.Equal(DiffWaveConditioner.MelSpectrogram, o.Conditioner); Assert.Equal(0, o.NumClasses);
        Assert.Equal(32, new FreGradOptions().HiddenDim);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(24)]
    [InlineData(32)]
    [InlineData(64)]
    public void SharedHop_ReconfiguresEveryRealEvenConditioningStage_AndTrains(int hop)
    {
        var o = PriorGradPaperTests.Options(fast: true); TtsModelOptions shared = o; shared.HopSize = hop;
        Assert.Equal(2, o.UpsampleStrides.Length);
        Assert.All(o.UpsampleStrides, s => Assert.True(s >= 2 && s % 2 == 0));
        Assert.Equal(hop, o.UpsampleStrides.Aggregate(1, (a, b) => a * b));
        var copy = new PriorGradOptions(o);
        var restored = JsonConvert.DeserializeObject<PriorGradOptions>(JsonConvert.SerializeObject(o))!;
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
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(12)]
    public void SharedWidth_ChangesTheActualResidualNetwork_AndSurvivesCopyAndJson(int width)
    {
        var o = PriorGradPaperTests.Options(fast: true); TtsModelOptions shared = o; shared.HiddenDim = width;
        Assert.Equal(width, o.ResChannels);
        foreach (var result in new[] { new PriorGradOptions(o), JsonConvert.DeserializeObject<PriorGradOptions>(JsonConvert.SerializeObject(o))! })
            Assert.Equal((width, width), (result.HiddenDim, result.ResChannels));
        using var model = Create(o);
        var baseline = PriorGradPaperTests.Options(fast: true); baseline.ResChannels = width + 4;
        using var wider = Create(baseline);
        Assert.True(wider.GetParameters().Length > model.GetParameters().Length);
        Assert.True(double.IsFinite(model.Train(Sample())));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeDimensionsAndStrides_WinOverBothLegacyAliases_InEitherOrderAndFreGrad(bool nativeFirst)
    {
        const string native = "\"ResChannels\":8,\"UpsampleStrides\":[4,4]";
        const string aliases = "\"HiddenDim\":256,\"HopSize\":256";
        string json = "{" + (nativeFirst ? native + "," + aliases : aliases + "," + native) + "}";
        foreach (var type in new[] { typeof(PriorGradOptions), typeof(DerivedOptions), typeof(FreGradOptions) })
        {
            var o = (PriorGradOptions)JsonConvert.DeserializeObject(json, type)!;
            Assert.Equal((8, 8), (o.HiddenDim, o.ResChannels));
            Assert.Equal(type == typeof(FreGradOptions) ? 32 : 16, o.HopSize);
        }
        var legacy = JsonConvert.DeserializeObject<PriorGradOptions>("{\"HiddenDim\":12,\"HopSize\":128}")!;
        Assert.Equal((12, 128), (legacy.ResChannels, legacy.HopSize));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(257)]
    public void InvalidHopsAndWidths_AreRejectedWithoutRemovingNativeStages(int hop)
    {
        var o = new PriorGradOptions();
        Assert.Throws<ArgumentOutOfRangeException>(() => o.HopSize = hop);
        Assert.Equal(new[] { 16, 16 }, o.UpsampleStrides);
        Assert.Throws<ArgumentOutOfRangeException>(() => o.HiddenDim = 0);
    }

    [Theory]
    [InlineData("unconditional", "Conditioner")]
    [InlineData("class", "Conditioner")]
    [InlineData("labels", "NumClasses")]
    [InlineData("negative-labels", "NumClasses")]
    [InlineData("dropout", "DropoutRate")]
    [InlineData("negative-dropout", "DropoutRate")]
    [InlineData("nan-dropout", "DropoutRate")]
    public void InheritedNonpaperTasksAndDropout_AreRejectedByTheRealConstructor(string invalid, string parameter)
    {
        var o = PriorGradPaperTests.Options();
        switch (invalid)
        {
            case "unconditional": o.Conditioner = DiffWaveConditioner.Unconditional; break;
            case "class": o.Conditioner = DiffWaveConditioner.ClassLabel; break;
            case "labels": o.NumClasses = 10; break;
            case "negative-labels": o.NumClasses = -1; break;
            case "dropout": o.DropoutRate = .1; break;
            case "negative-dropout": o.DropoutRate = -.1; break;
            case "nan-dropout": o.DropoutRate = double.NaN; break;
        }
        Assert.Equal(parameter, Assert.Throws<ArgumentException>(() => Create(o)).ParamName);
    }

    [Fact]
    public void DiffwaveUnconditionalLength_DoesNotChangeTheRealMelConditionedVocoder()
    {
        var a = PriorGradPaperTests.Options(fast: true); var b = new PriorGradOptions(a) { UtteranceSamples = 32000 };
        using var control = Create(a); using var candidate = Create(b);
        var mel = new Tensor<double>(new[] { 1, a.MelChannels, 4 });
        Assert.Equal(64, control.Predict(mel).Length);
        Assert.Equal(control.Predict(mel).ToArray(), candidate.Predict(mel).ToArray());
    }

    private sealed class SelectedStartRandom(int start, int maximum) : Random
    {
        public override int Next(int minValue, int maxValue)
        {
            Assert.Equal(0, minValue); Assert.Equal(maximum + 1, maxValue);
            Assert.InRange(start, minValue, maxValue - 1); return start;
        }
    }

    [Fact]
    public void RecordingAndPrior_MatchUnmodifiedReleasedLoaderMelAndStatisticsBodies()
    {
        var reference = Reference();
        Assert.Equal("8cf4fbcbf451c3affc68ecfa6c970327e927062f", (string)reference["commit"]!);
        Assert.Equal("c7d831235f19e804a966bb54cdf25f5c85191b10", (string)reference["git_blobs"]!["PriorGrad-vocoder/dataset.py"]!);
        Assert.Equal("e129e6ac4bdf66728016f20baaed6cb2b5586c61", (string)reference["git_blobs"]!["PriorGrad-vocoder/preprocess.py"]!);
        foreach (var c in reference["cases"]!)
        {
            var o = PriorGradPaperTests.Options(); o.EnergyMin = .02;
            using var model = Create(o);
            var pair = model.PrepareRecording(Audio((int)c["raw_length"]!),
                new SelectedStartRandom((int)c["start_sample"]!, (int)c["maximum_start"]!));
            Compare(c["audio"]!, pair.Audio.ToArray(), 1e-7);
            Compare(c["mel"]!, pair.Condition.ToArray(), 2e-6);
            var std = model.PriorStd(pair.Condition).ToArray();
            var expected = c["prior_std"]!.Select(x => (double)x).ToArray();
            Assert.Equal(expected.Length * o.HopSize, std.Length);
            for (int i = 0; i < std.Length; i++) Assert.InRange(Math.Abs(expected[i / o.HopSize] - std[i]), 0, 2e-6);
        }
        var settings = PriorGradPaperTests.Options(); using var fitted = Create(settings);
        fitted.FitEnergyStatistics(new[] { Audio(), Audio(divisor: 8) }, useDataMaximum: true);
        Assert.InRange(Math.Abs((double)reference["statistics"]!["minimum"]! - settings.EnergyMin!.Value), 0, 2e-6);
        Assert.InRange(Math.Abs((double)reference["statistics"]!["maximum"]! - settings.EnergyMax), 0, 2e-6);
        Assert.True((bool)reference["short_recording_rejected"]!);
        Assert.Throws<ArgumentException>(() => fitted.Train(Sample(37)));
    }

    private static void Compare(JToken expected, double[] actual, double tolerance)
    {
        var values = expected.Select(x => (double)x).ToArray(); Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++) Assert.InRange(Math.Abs(values[i] - actual[i]), 0, tolerance);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(.2)]
    public void CachedAdam_HonorsTheActualRateAndDecay_WithoutAdditionalClipping(double decay)
    {
        var o = PriorGradPaperTests.Options(); o.WeightDecay = decay; o.LearningRate = 3e-4;
        using var model = Create(o); Assert.True(double.IsFinite(model.Train(Sample())));
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
    [InlineData("EnergyMax")]
    [InlineData("WeightDecay")]
    public void RealRecordingCovarianceAndUpdates_RespondToConfiguredEnergyAndDecay(string option)
    {
        var a = PriorGradPaperTests.Options(fast: true); var b = new PriorGradOptions(a);
        if (option == "EnergyMax")
        {
            // The published energy range belongs to 1024-point features, not the
            // deliberately tiny frontend whose energies all hit the 0.1 lower clamp.
            a.FftSize = b.FftSize = 1024; a.WindowSize = b.WindowSize = 1024;
            a.HopSize = b.HopSize = 256; b.EnergyMax = 8;
        }
        else b.WeightDecay = .2;
        using var control = Create(a); using var candidate = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        if (option == "EnergyMax")
        {
            var mel = control.ComputeMel(Audio(4096));
            Assert.Contains(control.PriorStd(mel).ToArray(), x => x > a.MinStd && x < 1);
            Assert.False(control.PriorStd(mel).ToArray().SequenceEqual(candidate.PriorStd(mel).ToArray()));
            Assert.False(control.Predict(mel).ToArray().SequenceEqual(candidate.Predict(mel).ToArray()));
        }
        for (int i = 0; i < 2; i++) { control.Train(Sample(option == "EnergyMax" ? 4096 : 512)); candidate.Train(Sample(option == "EnergyMax" ? 4096 : 512)); }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(candidate.GetParameters().ToArray()));
    }
}
