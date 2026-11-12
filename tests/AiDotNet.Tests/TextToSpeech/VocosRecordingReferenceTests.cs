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
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class VocosRecordingReferenceTests
{
    public sealed class DerivedOptions : VocosOptions { }
    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech", "ReferenceData", "vocos_recording_reference.json")));
    private static double[] Values(JsonElement item) => item.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static Tensor<double> Raw(JsonElement item) => new(new[] { item.GetProperty("raw_pcm").GetArrayLength() },
        new Vector<double>(Values(item.GetProperty("raw_pcm")).Select(x => x / 32768.0).ToArray()));

    private sealed class RecordedRandom : Random
    {
        private readonly JsonElement _item;
        public int GainCalls { get; private set; }
        public int CropCalls { get; private set; }
        public RecordedRandom(JsonElement item) => _item = item;
        public override double NextDouble() { GainCalls++; return _item.GetProperty("gain_draw").GetDouble(); }
        public override int Next(int minValue, int maxValue)
        {
            Assert.Equal(0, minValue);
            Assert.Equal(_item.GetProperty("raw_pcm").GetArrayLength() - _item.GetProperty("segment").GetInt32() + 1, maxValue);
            CropCalls++; return _item.GetProperty("start_sample").GetInt32();
        }
    }

    private sealed class Probe : Vocos<double>
    {
        public Probe(VocosOptions o) : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: o.MelChannels, outputSize: o.HopSize) { RandomSeed = 8 }, o) { }
        public (Tensor<double> Mel, Tensor<double> Audio) Pair(Tensor<double> audio, Random random) => RecordingPair(audio, random);
    }

    private static VocosOptions Options() => new()
    {
        MelChannels = 8, FftSize = 64, HopSize = 16, SampleRate = 4000, ConvNeXtDim = 8, NumBackboneBlocks = 2, IntermediateDim = 16,
        DiscriminatorPeriods = [2], DiscriminatorWidthDivisor = 32, ResolutionDiscriminatorChannels = 2,
        ResolutionFftSizes = [64], ResolutionHopSizes = [16], ResolutionWindowSizes = [64], SegmentSize = 128,
    };

    private static void Close(Tensor<double> actual, double[] expected, double tolerance, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
        Assert.True(TtsPaperNumerics.IsFinite(error) && error <= tolerance, $"{name}: maximum error {error:R} exceeds {tolerance:R}.");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    public void RecordingPipeline_MatchesUnmodifiedAuthorDatasetAndRealSox(int index)
    {
        using var document = Reference(); var item = document.RootElement.GetProperty("cases")[index];
        var o = Options(); o.SegmentSize = item.GetProperty("segment").GetInt32();
        using var model = new Probe(o); var random = new RecordedRandom(item);
        var pair = model.Pair(Raw(item), random);
        Close(pair.Audio, Values(item.GetProperty("audio")), 0, "actual SoX-normalized and cropped/repeated float32 waveform");
        Assert.Equal(1, random.GainCalls);
        Assert.Equal(item.GetProperty("start_sample").ValueKind == JsonValueKind.Null ? 0 : 1, random.CropCalls);
        Assert.Equal(item.GetProperty("mel_shape").EnumerateArray().Select(x => x.GetInt32()), pair.Mel.Shape.ToArray());
        Close(pair.Mel, Values(item.GetProperty("mel")), 1e-9, "author float64 MelSpectrogramFeatures of the actual SoX waveform");
    }

    private static double[] Learned(Probe model, string group)
        => ((IEnumerable<LayerBase<double>>)typeof(GanVocoderBase<double>).GetField(group,
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!)
            .SelectMany(l => l.GetTrainableParameters()).SelectMany(p => p.ToArray()).ToArray();

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(3)]
    public void ShortAndLongRecordings_TrainActualGeneratorAndDiscriminators(int index)
    {
        using var document = Reference(); var item = document.RootElement.GetProperty("cases")[index];
        using var model = new Probe(Options());
        var generator = Learned(model, "_generatorLayers"); var discriminator = Learned(model, "_discriminatorLayers");
        Assert.NotEmpty(generator); Assert.NotEmpty(discriminator);
        double loss = model.Train(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(item) });
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.False(generator.SequenceEqual(Learned(model, "_generatorLayers")));
        Assert.False(discriminator.SequenceEqual(Learned(model, "_discriminatorLayers")));
    }

    private static Dictionary<string, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Optimizers(Probe model)
        => (Dictionary<string, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)typeof(GanVocoderBase<double>)
            .GetField("_optimizers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    [Theory]
    [InlineData(.01)]
    [InlineData(.2)]
    public void BothActualAdamWOptimizers_KeepConfiguredSettingsWithoutAdaptiveBetasOrClipping(double decay)
    {
        using var document = Reference(); var o = Options(); o.LearningRate = 3e-4; o.WeightDecay = decay;
        using var model = new Probe(o);
        model.Train(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) });
        Assert.Equal(2, Optimizers(model).Count);
        foreach (var optimizer in Optimizers(model).Values)
        {
            var settings = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.Equal(o.LearningRate, settings.InitialLearningRate);
            Assert.Equal((o.Beta1, o.Beta2, 1e-8, decay), (settings.Beta1, settings.Beta2, settings.Epsilon, settings.WeightDecay));
            Assert.False(settings.UseAdaptiveBetas, "The paper uses fixed betas.");
            Assert.False(settings.EnableGradientClipping, "The reference has no optimizer clipping.");
        }
        Assert.Equal(0, model.MaxGradNormValue);
    }

    [Fact]
    public void PublishedDefaults_KeepThePrimaryPapersStatedRecipe()
    {
        var o = new VocosOptions();
        Assert.Equal((512, 512, 1536, 8), (o.HiddenDim, o.ConvNeXtDim, o.IntermediateDim, o.NumBackboneBlocks));
        Assert.Equal((24000, 100, 1024, 256, 16384), (o.SampleRate, o.MelChannels, o.FftSize, o.HopSize, o.SegmentSize));
        Assert.Equal((.9, .999, 2e-4, .01, 1000000, 0.0), (o.Beta1, o.Beta2, o.LearningRate, o.WeightDecay, o.CosineSteps, o.DropoutRate));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(32)]
    public void SharedWidth_ConfiguresTheActualConvNextEmbeddingAndLearning(int width)
    {
        var o = Options(); TtsModelOptions shared = o; shared.HiddenDim = width;
        using var model = new Probe(o);
        var generator = (VocosGenerator<double>)typeof(Vocos<double>).GetField("_generator", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        Assert.Equal(new[] { width, 8, 1, 7 }, generator.Layers[0].GetTrainableParameters()[0].Shape.ToArray());
        var copy = new VocosOptions(o);
        var restored = JsonConvert.DeserializeObject<VocosOptions>(JsonConvert.SerializeObject(o))!;
        Assert.Equal((width, width), (copy.HiddenDim, restored.ConvNeXtDim));
        Assert.NotSame(o.DiscriminatorPeriods, copy.DiscriminatorPeriods);
        using var document = Reference();
        Assert.True(TtsPaperNumerics.IsFinite(model.Train(new TtsTrainingSample<double>
        { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) })));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SavedNativeWidth_WinsOverLegacyWidth_InEitherOrder(bool nativeFirst)
    {
        string json = nativeFirst ? "{\"ConvNeXtDim\":16,\"HiddenDim\":256}" : "{\"HiddenDim\":256,\"ConvNeXtDim\":16}";
        foreach (var type in new[] { typeof(VocosOptions), typeof(DerivedOptions) })
        {
            var o = (VocosOptions)JsonConvert.DeserializeObject(json, type)!;
            Assert.Equal((16, 16), (o.ConvNeXtDim, o.HiddenDim));
        }
        Assert.Equal(8, JsonConvert.DeserializeObject<VocosOptions>("{\"HiddenDim\":8}")!.ConvNeXtDim);
    }

    [Fact]
    public void SegmentLength_ChangesActualCropFeaturesAndBothNetworksLearning()
    {
        using var document = Reference(); var audio = Raw(document.RootElement.GetProperty("cases")[2]);
        var a = Options(); var b = new VocosOptions(a) { SegmentSize = 256 };
        using var control = new Probe(a); using var candidate = new Probe(b);
        Assert.Equal(control.GetParameters().ToArray(), candidate.GetParameters().ToArray());
        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio };
        control.Train(sample); candidate.Train(sample);
        Assert.False(Learned(control, "_generatorLayers").SequenceEqual(Learned(candidate, "_generatorLayers")));
        Assert.False(Learned(control, "_discriminatorLayers").SequenceEqual(Learned(candidate, "_discriminatorLayers")));
        var small = control.Pair(audio, new Random(8)); var large = candidate.Pair(audio, new Random(8));
        Assert.Equal((128, 9, 128), (small.Audio.Length, small.Mel.Shape[2], control.MelToWaveform(small.Mel).Length));
        Assert.Equal((256, 17, 256), (large.Audio.Length, large.Mel.Shape[2], candidate.MelToWaveform(large.Mel).Length));
    }

    [Theory]
    [InlineData(.8, .9)]
    [InlineData(.9, .999)]
    public void ConfiguredBetas_RemainFixedInBothRealOptimizers(double beta1, double beta2)
    {
        using var document = Reference(); var o = Options(); o.Beta1 = beta1; o.Beta2 = beta2;
        using var model = new Probe(o);
        model.Train(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) });
        foreach (var optimizer in Optimizers(model).Values)
        {
            var settings = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.Equal((beta1, beta2), (settings.Beta1, settings.Beta2)); Assert.False(settings.UseAdaptiveBetas);
        }
    }

    [Fact]
    public void FeatureMatching_AveragesLayersAsThePrimaryPaperSpecifies_WithCorrectGradients()
    {
        using var document = Reference(); var item = document.RootElement.GetProperty("feature_matching");
        Tensor<double>[] Tensors(string property) => item.GetProperty(property).EnumerateArray()
            .Select(x => new Tensor<double>(new[] { x.GetArrayLength() }, new Vector<double>(Values(x)))).ToArray();
        var real = Tensors("real"); var fake = Tensors("fake");
        using var model = new Probe(Options()); using var tape = new GradientTape<double>();
        var loss = model.MeanFeatureMatching(real, fake);
        Assert.Equal(item.GetProperty("author_sum").GetDouble() / 3, item.GetProperty("paper_mean").GetDouble(), 14);
        Close(loss, new[] { item.GetProperty("paper_mean").GetDouble() }, 1e-12, "primary 1/L feature matching");
        var gradients = tape.ComputeGradients(loss, sources: fake.ToList());
        var expected = item.GetProperty("gradients").EnumerateArray().ToArray();
        for (int i = 0; i < fake.Length; i++) Close(gradients[fake[i]], Values(expected[i]), 1e-12, $"feature layer {i} gradient");
    }

    [Fact]
    public void SpectralFrontend_AtSilence_HasFiniteValuesAndZeroGradient()
    {
        using var model = new Probe(Options());
        var features = (CenteredLogMel<double>)typeof(Vocos<double>).GetField("_features", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var audio = new Tensor<double>(new[] { 128 });
        using var tape = new GradientTape<double>();
        var mel = features.Forward(audio);
        Assert.All(mel.ToArray(), x => Assert.Equal(Math.Log(1e-7), x, 12));
        var loss = AiDotNetEngine.Current.ReduceMean(mel, new[] { 0, 1, 2 }, false);
        var gradient = tape.ComputeGradients(loss, sources: new[] { audio })[audio];
        Assert.All(gradient.ToArray(), x => Assert.Equal(0, x));
    }

    [Fact]
    public void BothCosineSchedulers_MatchIndependentTorchAfterRealUpdates()
    {
        using var document = Reference(); var o = Options(); o.CosineSteps = 4;
        using var model = new Probe(o);
        var sample = new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = Raw(document.RootElement.GetProperty("cases")[2]) };
        foreach (var rate in document.RootElement.GetProperty("cosine_rates").EnumerateArray())
        {
            model.Train(sample); Assert.Equal(2, Optimizers(model).Count);
            foreach (var optimizer in Optimizers(model).Values) Assert.Equal(rate.GetDouble(), optimizer.GetCurrentLearningRate(), 14);
        }
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(-.1)]
    [InlineData(double.NaN)]
    public void UnsupportedDropout_IsRejectedByTheActualConstructor(double dropout)
    {
        var o = Options(); o.DropoutRate = dropout;
        Assert.Equal("DropoutRate", Assert.Throws<ArgumentException>(() => new Probe(o)).ParamName);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(15)]
    [InlineData(129)]
    public void InvalidTrainingSegments_AreRejectedBeforeBuildingTheNetwork(int samples)
    {
        var o = Options(); o.SegmentSize = samples;
        Assert.Equal("SegmentSize", Assert.Throws<ArgumentException>(() => new Probe(o)).ParamName);
    }

    [Fact]
    public void FullAuthorBackboneAndIstftHead_MatchOutputsAndAllGradients()
    {
        using var document = Reference(); var item = document.RootElement.GetProperty("generator");
        var engine = AiDotNetEngine.Current;
        var generator = new VocosGenerator<double>(engine, new Random(8), 8, 8, 16, 2, 64, 16);
        generator.LoadTorchWeights((name, shape) =>
        {
            Assert.Equal(item.GetProperty("shapes").GetProperty(name).EnumerateArray().Select(x => x.GetInt32()), shape);
            return Values(item.GetProperty("weights").GetProperty(name));
        }, 8, 16);
        var names = new List<string> { "backbone.embed.weight", "backbone.embed.bias", "backbone.norm.weight", "backbone.norm.bias" };
        for (int i = 0; i < 2; i++)
        {
            string prefix = $"backbone.convnext.{i}.";
            foreach (var layer in new[] { "dwconv", "norm", "pwconv1", "pwconv2" })
            { names.Add(prefix + layer + ".weight"); names.Add(prefix + layer + ".bias"); }
            names.Add(prefix + "gamma");
        }
        names.AddRange(new[] { "backbone.final_layer_norm.weight", "backbone.final_layer_norm.bias", "head.out.weight", "head.out.bias" });
        var parameters = generator.Layers.SelectMany(l => l.GetTrainableParameters()).ToArray();
        Assert.Equal(26, parameters.Length); Assert.Equal(parameters.Length, names.Count);
        Assert.Equal(1818, parameters.Sum(p => p.Length)); Assert.Equal(1818, item.GetProperty("parameter_scalars").GetInt32());
        for (int i = 0; i < parameters.Length; i++) Close(parameters[i], Values(item.GetProperty("weights").GetProperty(names[i])), 0, names[i]);
        var mel = new Tensor<double>(new[] { 1, 8, 9 }, new Vector<double>(Values(item.GetProperty("input_mel"))));
        using var tape = new GradientTape<double>();
        var hidden = generator.Backbone(mel); var authorHidden = Values(item.GetProperty("hidden"));
        Close(hidden, Enumerable.Range(0, 72).Select(i => authorHidden[(i % 9) * 8 + i / 9]).ToArray(), 1e-9, "author ConvNeXt backbone");
        var audio = generator.Forward(mel);
        Close(audio, Values(item.GetProperty("audio")), 1e-9, "author centered ISTFT waveform");
        var target = new Tensor<double>(audio.Shape.ToArray(), new Vector<double>(Values(item.GetProperty("target"))));
        var loss = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(audio, target)), Enumerable.Range(0, audio.Rank).ToArray(), false);
        Close(loss, new[] { item.GetProperty("loss").GetDouble() }, 1e-10, "diagnostic waveform MSE");
        var gradients = tape.ComputeGradients(loss, sources: parameters.Concat(new[] { mel }).ToList());
        for (int i = 0; i < parameters.Length; i++)
        {
            var expected = Values(item.GetProperty("parameter_gradients").GetProperty(names[i]));
            Assert.True(expected.Max(Math.Abs) > 1e-14, $"Oracle gradient for {names[i]} is uninformative.");
            Close(gradients[parameters[i]], expected, 1e-9, names[i] + " gradient");
        }
        Close(gradients[mel], Values(item.GetProperty("mel_gradient")), 1e-9, "author mel gradient");
    }
}
