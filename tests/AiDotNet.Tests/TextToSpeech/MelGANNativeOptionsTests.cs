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
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class MelGANNativeOptionsTests
{
    private sealed class RecordingModel(MelGANOptions options) : MelGAN<double>(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 7 }, options)
    {
        public (Tensor<double> Mel, Tensor<double> Audio) Pair(Tensor<double> audio, Random random)
            => RecordingPair(audio, random);
    }

    private static Tensor<double> Audio(int length = 3072)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = ((i * 37) % 211 - 105) / 256.0 + (i % 7 - 3) / 64.0;
        return audio;
    }

    private static JObject Reference() => JObject.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "melgan_recording_reference.json")));

    private sealed class SelectedRandom(double amplitude, int start, int maximum) : Random
    {
        public override double NextDouble() => (amplitude - .3) / .7;
        public override int Next(int minValue, int maxValue)
        {
            Assert.Equal(0, minValue); Assert.Equal(maximum + 1, maxValue);
            Assert.InRange(start, minValue, maxValue - 1);
            return start;
        }
    }

    [Fact]
    public void PublishedDefaults_KeepTheGeneratorAndPaperLossAndTrainingRecipe()
    {
        var options = new MelGANOptions();
        Assert.Equal(new[] { 8, 8, 2, 2 }, options.UpsampleRates);
        Assert.Equal((22050, 80, 256, 1024, 1024, 8192),
            (options.SampleRate, options.MelChannels, options.HopSize, options.FftSize, options.WindowSize, options.SegmentSize));
        Assert.Equal((32, 3, 3, 10.0, 1e-4, .5, .9, 0.0),
            (options.Ngf, options.ResidualLayers, options.NumDiscriminators, options.FeatureMatchingWeight,
                options.LearningRate, options.Beta1, options.Beta2, options.WeightDecay));
    }

    [Fact]
    public void Copies_DoNotShareNativeUpsamplingArrays()
    {
        var source = MelGANPaperTests.Options(); var copy = new MelGANOptions(source);
        Assert.Equal(source.UpsampleRates, copy.UpsampleRates);
        copy.UpsampleRates[0] = 8;
        Assert.Equal(4, source.UpsampleRates[0]);
    }

    [Fact]
    public void RecordingPair_MatchesUnmodifiedAuthorLoaderAndFrontend_IncludingNormalizationAndAugmentation()
    {
        var reference = Reference();
        Assert.Equal("6488045bfba1975602288de07a58570c7b4d66ea", (string)reference["commit"]!);
        Assert.Equal("0982af8a60f09e5c0e4c9e4e0990e41f2a848832", (string)reference["git_blobs"]!["mel2wav/dataset.py"]!);
        Assert.Equal("da39c5d61efcc5593413a3650edc5a57e2dd0319", (string)reference["git_blobs"]!["mel2wav/modules.py"]!);
        foreach (var c in reference["cases"]!)
        {
            var options = MelGANPaperTests.Options(); options.SegmentSize = (int)c["segment_samples"]!;
            using var model = new RecordingModel(options);
            var raw = (string)c["name"]! == "silence" ? new Tensor<double>(new[] { (int)c["raw_length"]! }) : Audio((int)c["raw_length"]!);
            var pair = model.Pair(raw, new SelectedRandom((double)c["amplitude"]!, (int)c["start_sample"]!, (int)c["maximum_start"]!));
            Assert.Equal(new[] { 1, options.MelChannels, 8 }, pair.Mel.Shape.ToArray());
            Compare(c["audio"]!, pair.Audio.ToArray(), 1e-7);
            Compare(c["mel"]!, pair.Mel.ToArray(), 2e-6);
        }
    }

    [Fact]
    public void Frontend_MatchesUnmodifiedFloat32AuthorAudio2Mel_AtNormalAndLowEnergyAndSilence()
    {
        using var model = new RecordingModel(MelGANPaperTests.Options());
        foreach (var c in Reference()["features"]!)
        {
            var audio = new Tensor<double>(new[] { 256 });
            var values = c["audio"]!.Select(x => (double)x).ToArray();
            for (int i = 0; i < audio.Length; i++) audio[i] = values[i];
            // Original Audio2Mel explicitly uses float32 Hann windows, banks and FFTs;
            // this is a float32 reference bound, including the low-energy epsilon regression.
            Compare(c["mel"]!, model.ComputeMel(audio).ToArray(), 2e-6);
        }
    }

    private static void Compare(JToken expected, double[] actual, double tolerance)
    {
        var values = expected.Select(x => (double)x).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++) Assert.InRange(Math.Abs(values[i] - actual[i]), 0, tolerance);
    }

    [Theory]
    [InlineData(0.0, .5, .9)]
    [InlineData(.2, .6, .8)]
    public void BothRealAdamUpdates_UseConfiguredRateBetasAndDecay_WithoutExtraClipping(double decay, double beta1, double beta2)
    {
        var options = MelGANPaperTests.Options(); options.SegmentSize = 128;
        options.LearningRate = 3e-4; options.WeightDecay = decay; options.Beta1 = beta1; options.Beta2 = beta2;
        using var model = new RecordingModel(options);
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio() })));
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
            if (decay > 0) Assert.Equal(decay,
                Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        }
        Assert.Equal(0, model.MaxGradNormValue);
    }

    [Theory]
    [InlineData("SegmentSize")]
    [InlineData("SamplingSeed")]
    [InlineData("WeightDecay")]
    public void CropSeedAndDecay_ChangeRealRecordingTraining_WithReproducibleControl(string option)
    {
        var a = MelGANPaperTests.Options(); a.SegmentSize = 128;
        var b = new MelGANOptions(a);
        if (option == "SegmentSize") b.SegmentSize = 256;
        else if (option == "SamplingSeed") b.SamplingSeed = 17;
        else b.WeightDecay = .2;
        using var control = new RecordingModel(a); using var repeated = new RecordingModel(new MelGANOptions(a));
        using var candidate = new RecordingModel(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        for (int step = 0; step < 2; step++)
        {
            var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio() };
            control.Train(sample); repeated.Train(sample); candidate.Train(sample);
        }
        Assert.Equal(control.GetParameters().ToArray(), repeated.GetParameters().ToArray());
        Assert.False(GanVocoderWeights.Of(control.GeneratorLayers).SequenceEqual(GanVocoderWeights.Of(candidate.GeneratorLayers)));
        Assert.False(GanVocoderWeights.Of(control.DiscriminatorLayers).SequenceEqual(GanVocoderWeights.Of(candidate.DiscriminatorLayers)));
    }
}
