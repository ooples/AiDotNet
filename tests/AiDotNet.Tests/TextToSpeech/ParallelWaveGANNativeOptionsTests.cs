using System;
using System.Collections;
using System.Collections.Generic;
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

public class ParallelWaveGANNativeOptionsTests
{
    public sealed class DerivedOptions : ParallelWaveGANOptions { }

    private static ParallelWaveGAN<double> Create(ParallelWaveGANOptions options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 3 }, options);

    private static Tensor<double> Audio(int length = 3077)
    {
        var audio = new Tensor<double>(new[] { length });
        for (int i = 0; i < length; i++) audio[i] = ((i * 37) % 211 - 105) / 256.0 + (i % 7 - 3) / 64.0;
        return audio;
    }

    private static TtsTrainingSample<double> Sample(int length = 3077) => new()
    { Tokens = new Tensor<double>(new[] { 1 }), Audio = Audio(length) };

    private static JObject Reference() => JObject.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "parallel_wavegan_recording_reference.json")));

    [Fact]
    public void PublishedDefaults_PreserveThePaperRatherThanTheDifferentLjspeechRecipe()
    {
        var o = new ParallelWaveGANOptions();
        Assert.Equal((24000, 80, 300, 1200, 70.0, 8000.0),
            (o.SampleRate, o.MelChannels, o.HopSize, o.WindowSize, o.MelMinFrequency, o.MelMaxFrequency));
        Assert.Equal(new[] { 4, 5, 3, 5 }, o.UpsampleRates);
        Assert.Equal((30, 3, 64, 64, 3, 10, 4.0, 100000L, 1e-4, 5e-5, 1e-6, 200000),
            (o.NumLayers, o.NumStacks, o.ResidualChannels, o.SkipChannels, o.KernelSize,
                o.DiscriminatorLayers, o.AdversarialWeight, o.DiscriminatorStartStep, o.LearningRate,
                o.DiscriminatorLearningRate, o.Epsilon, o.LearningRateHalvingSteps));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(17)]
    [InlineData(32)]
    public void SharedHop_ChangesRealConditioningGeometry_WhileKeepingEveryLearnedStage(int hop)
    {
        var o = ParallelWaveGANPaperTests.Options();
        TtsModelOptions shared = o; shared.HopSize = hop;
        Assert.Equal(2, o.UpsampleRates.Length);
        Assert.All(o.UpsampleRates, s => Assert.True(s >= 1));
        Assert.Equal(hop, o.UpsampleRates.Aggregate(1, (a, b) => a * b));
        var copy = new ParallelWaveGANOptions(o);
        var restored = JsonConvert.DeserializeObject<ParallelWaveGANOptions>(JsonConvert.SerializeObject(o))!;
        foreach (var result in new[] { copy, restored })
        {
            Assert.Equal(hop, result.HopSize); Assert.Equal(o.UpsampleRates, result.UpsampleRates);
            Assert.NotSame(o.UpsampleRates, result.UpsampleRates);
        }
        using var model = Create(o);
        Assert.Equal(5 * hop, model.Predict(new Tensor<double>(new[] { 1, o.MelChannels, 5 })).Length);
        Assert.True(double.IsFinite(model.Train(Sample())));
        Assert.Equal(5 * hop, model.Predict(new Tensor<double>(new[] { 1, o.MelChannels, 5 })).Length);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SavedNativeScales_WinOverLegacyHops_InEitherOrderAndDerivedTypes(bool nativeFirst)
    {
        const string native = "\"UpsampleRates\":[4,4]", alias = "\"HopSize\":300";
        string json = "{" + (nativeFirst ? native + "," + alias : alias + "," + native) + "}";
        foreach (var type in new[] { typeof(ParallelWaveGANOptions), typeof(DerivedOptions) })
            Assert.Equal(16, ((ParallelWaveGANOptions)JsonConvert.DeserializeObject(json, type)!).HopSize);
        Assert.Equal(128, JsonConvert.DeserializeObject<ParallelWaveGANOptions>("{\"HopSize\":128}")!.HopSize);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NonpositiveHop_IsRejectedWithoutChangingNativeStages(int hop)
    {
        var o = new ParallelWaveGANOptions();
        Assert.Throws<ArgumentOutOfRangeException>(() => o.HopSize = hop);
        Assert.Equal(new[] { 4, 5, 3, 5 }, o.UpsampleRates);
    }

    private sealed class SelectedFrameRandom(int start, int minimum, int exclusiveEnd) : Random
    {
        public override int Next(int minValue, int maxValue)
        {
            Assert.Equal(minimum, minValue); Assert.Equal(exclusiveEnd, maxValue);
            Assert.InRange(start, minValue, maxValue - 1); return start;
        }
    }

    [Fact]
    public void RecordingPair_MatchesUnmodifiedAuthorFrontendAndCollater_WithTrueContextAndRoundedSegments()
    {
        var reference = Reference();
        Assert.Equal("86740373ec609cb9fb192d472d2aea125041491a", (string)reference["commit"]!);
        Assert.Equal("5fab384adcf1f0461d918a021d33f9f2fc5f0295", (string)reference["git_blobs"]!["parallel_wavegan/bin/train.py"]!);
        Assert.Equal("1654c3d1212d08f98c2defe0dc51f5c591126eb2", (string)reference["git_blobs"]!["parallel_wavegan/bin/preprocess.py"]!);
        foreach (var c in reference["cases"]!)
        {
            var o = ParallelWaveGANPaperTests.Options(); o.SegmentSize = (int)c["segment_samples"]!;
            o.AuxContextWindow = (int)c["context"]!;
            o.MelMean = c["means"]!.Type == JTokenType.Null ? null : c["means"]!.Select(x => (double)x).ToArray();
            o.MelScale = c["scales"]!.Type == JTokenType.Null ? null : c["scales"]!.Select(x => (double)x).ToArray();
            using var model = Create(o);
            var pair = model.PrepareRecording(Audio((int)c["raw_length"]!),
                new SelectedFrameRandom((int)c["start_frame"]!, (int)c["minimum_start"]!, (int)c["exclusive_end"]!));
            Assert.Equal(c["audio"]!.Select(x => (double)x).ToArray(), pair.Audio.ToArray());
            Assert.Equal(new[] { 1, o.MelChannels, o.SegmentSize / o.HopSize + 2 * o.AuxContextWindow }, pair.Mel.Shape.ToArray());
            var expected = c["mel"]!.Select(x => (double)x).ToArray(); var actual = pair.Mel.ToArray();
            Assert.Equal(expected.Length, actual.Length);
            // Collater explicitly casts the unmodified released features to float32.
            for (int i = 0; i < expected.Length; i++) Assert.InRange(Math.Abs(expected[i] - actual[i]), 0, 2e-6);
            Assert.True(double.IsFinite(model.Train(Sample())));
        }
    }

    [Fact]
    public void TooShortRecording_IsRejectedAsInTheReleasedFilter_AndRestoresInferenceContext()
    {
        var o = ParallelWaveGANPaperTests.Options(); o.SegmentSize = 128;
        using var model = Create(o);
        Assert.True((bool)Reference()["short_recording_removed_by_author_length_filter"]!);
        Assert.Throws<ArgumentException>(() => model.Train(Sample(128)));
        Assert.Equal(48, model.Predict(new Tensor<double>(new[] { 1, o.MelChannels, 3 })).Length);
    }

    [Theory]
    [InlineData("mean-only")]
    [InlineData("scale-only")]
    [InlineData("mean-length")]
    [InlineData("scale-length")]
    [InlineData("mean-nan")]
    [InlineData("scale-zero")]
    [InlineData("scale-infinite")]
    public void FittedNormalization_RequiresCompleteFinitePerBandStatistics(string invalid)
    {
        var o = ParallelWaveGANPaperTests.Options(); o.MelMean = new double[o.MelChannels];
        o.MelScale = Enumerable.Repeat(1.0, o.MelChannels).ToArray();
        switch (invalid)
        {
            case "mean-only": o.MelScale = null; break;
            case "scale-only": o.MelMean = null; break;
            case "mean-length": o.MelMean = [0]; break;
            case "scale-length": o.MelScale = [1]; break;
            case "mean-nan": o.MelMean[0] = double.NaN; break;
            case "scale-zero": o.MelScale[0] = 0; break;
            case "scale-infinite": o.MelScale[0] = double.PositiveInfinity; break;
        }
        Assert.ThrowsAny<ArgumentException>(() => Create(o));
    }

    private static IDictionary Optimizers(ParallelWaveGAN<double> model) => (IDictionary)typeof(GanVocoderBase<double>)
        .GetField("_optimizers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private sealed class CountingRandom(int seed) : Random(seed)
    {
        public int Draws { get; private set; }
        public List<double> Uniforms { get; } = new();
        public override double NextDouble()
        {
            double value = base.NextDouble(); Draws++; Uniforms.Add(value); return value;
        }
    }

    private sealed class NoiseAuditModel(ParallelWaveGANOptions options) : ParallelWaveGAN<double>(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: options.MelChannels, outputSize: options.HopSize) { RandomSeed = 3 }, options)
    {
        public CountingRandom TrainingRandom { get; set; } = null!;
        public List<bool> TrainingModes { get; } = new();
        public List<double[]> GeneratorWeights { get; } = new();
        public double[]? DiscriminatorWaveform { get; private set; }
        public double[]? ReplayedWaveform { get; private set; }

        protected override Tensor<double> Generate(Tensor<double> mel)
        {
            var waveform = base.Generate(mel);
            TrainingModes.Add(IsTrainingMode);
            GeneratorWeights.Add(GanVocoderWeights.Of(GeneratorLayers));
            if (TrainingModes.Count == 2)
            {
                // Replay the first forward's actual Gaussian input through the real updated
                // native network: the author trainer calls generator(*x) twice on the same x.
                DiscriminatorWaveform = waveform.ToArray();
                var noise = new Tensor<double>(new[] { 1, 1, waveform.Length });
                for (int i = 0; i < noise.Length; i++)
                    noise[i] = Math.Sqrt(-2 * Math.Log(1 - TrainingRandom.Uniforms[2 * i]))
                        * Math.Cos(2 * Math.PI * TrainingRandom.Uniforms[2 * i + 1]);
                var network = (ParallelWaveGanGenerator<double>)typeof(ParallelWaveGAN<double>)
                    .GetField("_generator", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(this)!;
                ReplayedWaveform = network.Forward(noise, mel, hasContext: true).ToArray();
            }
            return waveform;
        }
    }

    [Fact]
    public void DiscriminatorAfterGenerator_RecomputesWithUpdatedWeightsAndTheSameBatchNoise_AsReleasedTrainer()
    {
        var o = ParallelWaveGANPaperTests.Options(0); o.SegmentSize = 128;
        var random = new CountingRandom(o.SamplingSeed + 1);
        using var model = new NoiseAuditModel(o) { TrainingRandom = random };
        typeof(ParallelWaveGAN<double>).GetField("_noise", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, random);
        Assert.True(double.IsFinite(model.Train(Sample())));
        Assert.Equal(new[] { true, true }, model.TrainingModes);
        Assert.False(model.GeneratorWeights[0].SequenceEqual(model.GeneratorWeights[1]));
        Assert.NotNull(model.DiscriminatorWaveform); Assert.NotNull(model.ReplayedWaveform);
        Assert.Equal(model.ReplayedWaveform, model.DiscriminatorWaveform);
        // Two uniforms per sample, once per batch; both real forwards use that same Gaussian input.
        Assert.Equal(2 * o.SegmentSize, random.Draws);
        var mel = new Tensor<double>(new[] { 1, o.MelChannels, 3 });
        Assert.Equal(model.Predict(mel).ToArray(), model.Predict(mel).ToArray());
        Assert.Equal(2 * o.SegmentSize, random.Draws);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(.2)]
    public void BothCachedRadams_HonorNativeSettingsAndExactReleasedTorchStepSchedule(double decay)
    {
        var o = ParallelWaveGANPaperTests.Options(0); o.SegmentSize = 128;
        o.WeightDecay = decay; o.LearningRateHalvingSteps = 2;
        using var model = Create(o);
        var rates = Reference()["rates_after_steps"]!;
        for (int step = 0; step < 6; step++)
        {
            Assert.True(double.IsFinite(model.Train(Sample())));
            var optimizers = Optimizers(model); Assert.Equal(2, optimizers.Count);
            foreach (string group in new[] { "generator", "discriminator" })
            {
                var optimizer = Assert.IsType<RAdamOptimizer<double, Tensor<double>, Tensor<double>>>(optimizers[group]);
                var settings = Assert.IsType<RAdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
                Assert.Equal((double)rates[group]![step]!, optimizer.GetCurrentLearningRate(), 14);
                Assert.Equal((.9, .999, 1e-6), (settings.Beta1, settings.Beta2, settings.Epsilon));
                Assert.False(settings.EnableGradientClipping);
                Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
                if (decay > 0) Assert.Equal(decay,
                    Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
            }
        }
    }

    [Theory]
    [InlineData("AdversarialWeight")]
    [InlineData("DiscriminatorLearningRate")]
    [InlineData("DiscriminatorGradientClipNorm")]
    [InlineData("Epsilon")]
    [InlineData("WeightDecay")]
    [InlineData("SegmentSize")]
    public void RealUpdates_RespondToConfiguredControls_InTheApplicableTrainingPhase(string option)
    {
        var a = ParallelWaveGANPaperTests.Options(0); a.SegmentSize = 128;
        var b = new ParallelWaveGANOptions(a);
        switch (option)
        {
            case "AdversarialWeight": b.AdversarialWeight = 8; break;
            case "DiscriminatorLearningRate": b.DiscriminatorLearningRate = 1e-3; break;
            case "DiscriminatorGradientClipNorm": b.DiscriminatorGradientClipNorm = 1e-5; break;
            case "Epsilon": b.Epsilon = .1; break;
            case "WeightDecay": b.WeightDecay = .2; break;
            case "SegmentSize": b.SegmentSize = 256; break;
        }
        using var control = Create(a); using var candidate = Create(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        int steps = option == "Epsilon" ? 6 : 2;
        for (int step = 0; step < steps; step++)
        {
            control.Train(Sample()); candidate.Train(Sample());
            if (option == "Epsilon" && step < 3)
                Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        }
        Assert.False(control.GetParameters().ToArray().SequenceEqual(candidate.GetParameters().ToArray()));
    }
}
