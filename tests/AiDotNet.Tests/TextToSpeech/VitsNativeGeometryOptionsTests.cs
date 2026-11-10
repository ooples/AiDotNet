using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class VitsNativeGeometryOptionsTests
{
    private static VitsModelOptions Defaults(string family) => family switch
    {
        "VITS" => new VITSOptions(), "VITS2" => new VITS2Options(), "Piper" => new PiperOptions(),
        "YourTTS" => new YourTTSOptions(), _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    private static VitsModelOptions Compact(string family)
    {
        var o = Defaults(family);
        o.VocabSize = 32; o.HiddenDim = 16; o.InterChannels = 8; o.FilterChannels = 32;
        o.NumHeads = 2; o.NumEncoderLayers = 3; o.DropoutRate = 0; o.PosteriorLayers = 1;
        o.FlowLayers = 1; o.NumFlowSteps = 2; o.DurationPredictorDropout = 0;
        o.UpsampleRates = [4, 4]; o.UpsampleKernelSizes = [8, 8]; o.UpsampleInitialChannels = 16;
        o.ResblockKernelSizes = [3]; o.ResblockDilationSizes = [[1, 3]];
        o.DiscriminatorPeriods = [2, 3]; o.DiscriminatorWidthDivisor = 32;
        o.FftSize = o.WindowSize = 64; o.HopSize = 16; o.SampleRate = 4000; o.MelChannels = 8; o.SegmentSize = 128;
        if (o is VITSOptions stochastic) stochastic.DurationPredictorFlows = 2;
        if (o is VITS2Options vits2) { vits2.DurationPredictorFilterChannels = 16; vits2.FlowTransformerDropout = 0; }
        if (o is YourTTSOptions your) { your.SpeakerEncoderDim = 8; your.SpeakerEncoderFilters = [8, 8, 8, 8]; }
        return o;
    }

    private static VitsTtsModelBase<double> Create(VitsModelOptions o)
    {
        var arch = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 4, outputSize: 64) { RandomSeed = 2294 };
        return o switch
        {
            YourTTSOptions your => new YourTTS<double>(arch, your),
            PiperOptions piper => new Piper<double>(arch, piper),
            VITS2Options vits2 => new VITS2<double>(arch, vits2),
            VITSOptions vits => new VITS<double>(arch, vits),
            _ => throw new ArgumentOutOfRangeException(nameof(o)),
        };
    }

    [Theory]
    [InlineData("VITS")]
    [InlineData("VITS2")]
    [InlineData("Piper")]
    [InlineData("YourTTS")]
    public void DefaultDecoderCountMatchesReleasedHifiGanStages(string family)
    {
        TtsModelOptions o = Defaults(family);
        // Piper medium/x-low use the released low_quality decoder; its high voice uses VITS's V1.
        Assert.Equal(family == "Piper" ? 3 : 4, o.NumDecoderLayers);
        Assert.Equal(family == "Piper" ? new[] { 8, 8, 4 } : new[] { 8, 8, 2, 2 }, ((VitsModelOptions)o).UpsampleRates);
        if (family == "Piper")
        {
            Assert.Equal(3, PiperOptions.XLow().NumDecoderLayers);
            Assert.Equal(4, PiperOptions.High().NumDecoderLayers);
        }
    }

    [Theory]
    [InlineData("VITS", 1)]
    [InlineData("VITS", 3)]
    [InlineData("VITS2", 1)]
    [InlineData("VITS2", 3)]
    [InlineData("Piper", 1)]
    [InlineData("Piper", 3)]
    [InlineData("YourTTS", 1)]
    [InlineData("YourTTS", 3)]
    public void SharedDecoderCountResizesActualUpsamplersAndPreservesWaveformHop(string family, int stages)
    {
        var original = Compact(family);
        using var control = Create(original);
        var copy = (VitsModelOptions)CloneEngine.CopyConfiguration(original);
        ((TtsModelOptions)copy).NumDecoderLayers = stages;
        Assert.Equal(2, original.NumDecoderLayers);
        Assert.Equal(stages, copy.UpsampleRates.Length);
        Assert.Equal(stages, copy.UpsampleKernelSizes.Length);
        Assert.Equal(16, copy.UpsampleRates.Aggregate(1, (a, b) => a * b));
        using var changed = Create(copy);
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var decoder = Assert.IsType<HiFiGanGenerator<double>>(
            typeof(VitsTtsModelBase<double>).GetField("_decoder", flags)!.GetValue(changed));
        Assert.Equal(stages, Assert.IsAssignableFrom<System.Collections.ICollection>(
            decoder.GetType().GetField("_ups", flags)!.GetValue(decoder)).Count);
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
        var latent = new Tensor<double>(new[] { 1, 8, 3 });
        for (int i = 0; i < latent.Length; i++) latent[i] = Math.Sin(i * 0.2);
        var waveform = decoder.Forward(latent);
        Assert.Equal(new[] { 1, 1, 48 }, waveform.Shape);
        Assert.All(waveform, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData("VITS", false)]
    [InlineData("VITS", true)]
    [InlineData("VITS2", false)]
    [InlineData("VITS2", true)]
    [InlineData("Piper", false)]
    [InlineData("Piper", true)]
    [InlineData("YourTTS", false)]
    [InlineData("YourTTS", true)]
    public void LegacyJsonKeepsNativeGeometryRegardlessOfPropertyOrder(string family, bool reverse)
    {
        var original = Compact(family);
        var json = JObject.FromObject(original);
        json["NumDecoderLayers"] = 6;
        if (original is YourTTSOptions) json["SpeakerEmbeddingDim"] = 256;
        if (reverse) json = new JObject(json.Properties().Reverse().Select(p => new JProperty(p.Name, p.Value.DeepClone())));
        var restored = (VitsModelOptions)JsonConvert.DeserializeObject(json.ToString(), original.GetType())!;
        Assert.Equal(2, restored.NumDecoderLayers);
        Assert.Equal(original.UpsampleRates, restored.UpsampleRates);
        if (restored is YourTTSOptions your) Assert.Equal(8, your.SpeakerEmbeddingDim);
        using var before = Create(original);
        using var after = Create(restored);
        Assert.Equal(before.GetParameters().Length, after.GetParameters().Length);
    }

    [Fact]
    public void SharedYourTtsSpeakerWidthChangesNativeHaspEmbeddingAndConditioning()
    {
        var source = (YourTTSOptions)Compact("YourTTS");
        using var control = (YourTTS<double>)Create(source);
        var copy = (YourTTSOptions)CloneEngine.CopyConfiguration(source);
        ((VitsModelOptions)copy).SpeakerEmbeddingDim = 16;
        Assert.Equal((8, 16, 16), (source.SpeakerEncoderDim, copy.SpeakerEncoderDim, copy.SpeakerEmbeddingDim));
        using var changed = (YourTTS<double>)Create(copy);
        var recording = new Tensor<double>(new[] { 4000 });
        for (int i = 0; i < recording.Length; i++) recording[i] = 0.3 * Math.Sin(2 * Math.PI * 180 * i / 4000.0);
        Assert.Equal(8, control.ComputeSpeakerEmbedding(recording).Length);
        var embedding = changed.ComputeSpeakerEmbedding(recording);
        Assert.Equal(16, embedding.Length);
        Assert.All(embedding, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
    }

    [Fact]
    public void PartialAliasesConfigureActualNativeGeometry()
    {
        var options = JsonConvert.DeserializeObject<YourTTSOptions>("{\"NumDecoderLayers\":3,\"SpeakerEmbeddingDim\":16}")!;
        Assert.Equal((3, 16, 16), (options.NumDecoderLayers, options.SpeakerEncoderDim, options.SpeakerEmbeddingDim));
        Assert.Equal(256, options.UpsampleRates.Aggregate(1, (a, b) => a * b));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(5)]
    [InlineData(31)]
    public void InvalidDecoderCountsCannotCreateZeroChannelStages(int stages)
    {
        var options = Compact("VITS");
        Assert.Throws<ArgumentOutOfRangeException>(() => options.NumDecoderLayers = stages);
        Assert.Equal(new[] { 4, 4 }, options.UpsampleRates);
    }

    [Fact]
    public void YourTtsRefusesLearnedSpeakerTablesBecauseItsVoicesAreExternal()
    {
        var options = (YourTTSOptions)Compact("YourTTS");
        options.NumSpeakers = 2;
        Assert.Throws<ArgumentException>(() => Create(options));
        Assert.Equal(512, new YourTTSOptions().SpeakerEmbeddingDim);
    }

    public sealed class CustomVitsOptions : VITSOptions
    {
        public string CustomMarker { get; set; } = "marker";
    }

    [Fact]
    public void ConverterPreservesDerivedVitsOptions()
    {
        var options = JsonConvert.DeserializeObject<CustomVitsOptions>("{\"CustomMarker\":\"custom\",\"NumDecoderLayers\":3}")!;
        Assert.Equal("custom", options.CustomMarker);
        Assert.Equal(3, options.UpsampleRates.Length);
    }
}
