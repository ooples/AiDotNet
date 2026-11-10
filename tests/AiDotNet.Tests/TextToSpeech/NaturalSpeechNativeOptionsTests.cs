using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class NaturalSpeechNativeOptionsTests
{
    private static NaturalSpeechOptions Tiny() => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32,
        DurationFilter = 8, DurationDropout = 0, DropoutRate = 0, UpsamplingFilter = 4,
        WeightDecay = 0, // A decoder update must come from the native objective's gradient.
        Couplings = 2, CouplingLayers = 1, UpsampleRates = new[] { 4, 4 },
        UpsampleKernels = new[] { 8, 8 }, UpsampleInitialChannels = 32,
        ResblockKernels = new[] { 3 }, ResblockDilations = new[] { new[] { 1, 3 } },
        DiscriminatorPeriods = new[] { 2, 3 }, DiscriminatorWidthDivisor = 16, MemorySize = 8,
        FftSize = 64, HopSize = 16, SampleRate = 4000, MelChannels = 8, SegmentFrames = 8,
    };

    private static NaturalSpeech<double> Create(NaturalSpeechOptions options)
        => new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 11 }, options);

    private static T Field<T>(object owner, string name) => (T)owner.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;

    private static NaturalSpeechCore<double> Core(NaturalSpeech<double> model) => Field<NaturalSpeechCore<double>>(model, "_core");

    private static void CheckDecode(NaturalSpeech<double> model, NaturalSpeechOptions options)
    {
        var latent = new Tensor<double>(new[] { 1, options.HiddenDim, 3 });
        for (int i = 0; i < latent.Length; i++) latent[i] = .1 * Math.Sin(i * .2);
        var core = Core(model);
        Assert.Equal(options.NumDecoderLayers, Field<ICollection>(core.Decoder, "_ups").Count);
        var waveform = core.Decode(latent);
        Assert.Equal(3 * options.HopSize, waveform.Length);
        Assert.All(waveform.ToVector().ToArray(), x => Assert.True(!double.IsNaN(x) && !double.IsInfinity(x)));
    }

    private static void Train(NaturalSpeech<double> model)
    {
        var ids = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 60, 70, 80 }));
        var audio = new Tensor<double>(new[] { 512 });
        for (int i = 0; i < audio.Length; i++) audio[i] = .3 * Math.Sin(2 * Math.PI * 300 * i / 4000.0);
        double loss = model.Train(new TtsTrainingSample<double> { Tokens = ids, Audio = audio });
        Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(17)]
    [InlineData(18)]
    [InlineData(32)]
    public void HopAlias_RebuildsPairedNativeStridesAndKernels_AndTrains(int hop)
    {
        var options = Tiny(); options.HopSize = hop;
        Assert.Equal(hop, options.UpsampleRates.Aggregate(1, (a, b) => a * b));
        using var model = Create(options);
        CheckDecode(model, options);
        Train(model);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void DecoderDepth_BuildsAndUpdatesTheActualNativeGenerator(int depth)
    {
        var options = Tiny(); options.NumDecoderLayers = depth;
        Assert.Equal(16, options.HopSize);
        using var model = Create(options);
        CheckDecode(model, options);
        var decoder = Core(model).Decoder;
        var before = decoder.Layers.SelectMany(layer => layer.GetParameters().ToArray()).ToArray();
        Train(model);
        Assert.False(before.SequenceEqual(decoder.Layers.SelectMany(layer => layer.GetParameters().ToArray())));
    }

    [Fact]
    public void AddBlank_ActsOnTheActualPreservedPunctuationFrontend()
    {
        Assert.False(new NaturalSpeechOptions().AddBlank);
        Assert.True(NaturalSpeechOptions.Heatz123().AddBlank);
        var options = Tiny();
        using var model = Create(options);
        var plain = model.TextToPhonemes("!");
        Assert.Single(plain);
        Assert.NotEqual(0, plain[0]);
        options.AddBlank = true;
        Assert.Equal(new[] { 0, plain[0], 0 }, model.TextToPhonemes("!"));
    }

    [Fact]
    public void NativeGeneratorAndDiscriminator_UseFixedAdamWWithoutAnUndeclaredGlobalClip()
    {
        var options = Tiny();
        using var model = Create(options);
        Train(model);
        var optimizers = Field<IDictionary>(model, "_optimizers");
        Assert.NotEmpty(optimizers.Values.Cast<object>());
        foreach (var optimizer in optimizers.Values.Cast<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>())
        {
            var settings = Assert.IsType<AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.False(settings.EnableGradientClipping);
            Assert.False(settings.UseAdaptiveBetas);
            Assert.Equal(options.LearningRate, settings.InitialLearningRate);
            Assert.Equal(0.8, settings.Beta1); Assert.Equal(0.99, settings.Beta2); Assert.Equal(1e-9, settings.Epsilon);
            Assert.Equal(0, settings.WeightDecay);
        }
    }

    public class DerivedNaturalOptions : NaturalSpeechOptions { public string Marker { get; set; } = "custom"; }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FullLegacyJson_NativeDecoderGeometryWinsSharedAliasesInEitherOrder(bool reverse)
    {
        var data = JObject.FromObject(Tiny());
        data["HopSize"] = 256; data["NumDecoderLayers"] = 6; data["Marker"] = "restored";
        if (reverse) data = new JObject(data.Properties().Reverse().Select(p => new JProperty(p.Name, p.Value.DeepClone())));
        var options = JsonConvert.DeserializeObject<DerivedNaturalOptions>(data.ToString())!;
        Assert.Equal("restored", options.Marker);
        Assert.Equal(16, options.HopSize); Assert.Equal(2, options.NumDecoderLayers);
        Assert.Equal(new[] { 4, 4 }, options.UpsampleRates); Assert.Equal(new[] { 8, 8 }, options.UpsampleKernels);
        using var model = Create(options);
        CheckDecode(model, options);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AliasOnlyJson_ComposesRealDecoderGeometryInEitherOrder(bool reverse)
    {
        var defaults = new NaturalSpeechOptions();
        Assert.Equal(256, defaults.HopSize); Assert.Equal(4, defaults.NumDecoderLayers);
        Assert.Equal(new[] { 8, 8, 2, 2 }, defaults.UpsampleRates);
        Assert.Equal(new[] { 16, 16, 4, 4 }, defaults.UpsampleKernels);
        string json = reverse ? "{ 'NumDecoderLayers':3, 'HopSize':17 }" : "{ 'HopSize':17, 'NumDecoderLayers':3 }";
        var restored = JsonConvert.DeserializeObject<NaturalSpeechOptions>(json)!;
        Assert.Equal(17, restored.HopSize); Assert.Equal(3, restored.NumDecoderLayers);
        var options = Tiny();
        options.UpsampleRates = restored.UpsampleRates; options.UpsampleKernels = restored.UpsampleKernels;
        using var model = Create(options);
        CheckDecode(model, options);
    }
}
