using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Regularization;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Classic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class AcousticMelOptionAliasesTests
{
    private static AcousticModelOptions Options(string family)
    {
        AcousticModelOptions options = family switch
        {
            "FastSpeech" => new FastSpeechOptions { FftFilterSize = 32, DurationPredictorFilterSize = 16 },
            "FastSpeech2" => new FastSpeech2Options { FftFilterSize = 32, VariancePredictorFilterSize = 16 },
            "AlignTTS" => new AlignTTSOptions
            {
                FftFilterSize = 32, DurationPredictorDim = 8, DurationPredictorLayers = 1, MixDensityHiddenSize = 16,
            },
            "TransformerTTS" => new TransformerTTSOptions
            {
                FeedForwardDim = 32, EncoderPrenetLayers = 1, EncoderPrenetChannels = 16,
                DecoderPrenetSizes = [16, 16], PostnetLayers = 2, PostnetDim = 16, MaxDecoderSteps = 4,
            },
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
        options.EncoderDim = options.HiddenDim = 16;
        options.NumHeads = 2;
        options.NumEncoderLayers = options.NumDecoderLayers = 1;
        options.MelChannels = 8;
        options.MaxTextLength = 32;
        options.MaxMelLength = 64;
        return options;
    }

    private static TtsModelBase<double> Create(AcousticModelOptions options)
    {
        var arch = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: options.MelChannels) { RandomSeed = 2294 };
        return options switch
        {
            FastSpeechOptions fast => new FastSpeech<double>(arch, fast),
            FastSpeech2Options fast2 => new FastSpeech2<double>(arch, fast2),
            AlignTTSOptions align => new AlignTTS<double>(arch, align),
            TransformerTTSOptions transformer => new TransformerTTS<double>(arch, transformer),
            _ => throw new ArgumentException("Unsupported acoustic fixture.", nameof(options)),
        };
    }

    [Theory]
    [InlineData("FastSpeech")]
    [InlineData("FastSpeech2")]
    [InlineData("AlignTTS")]
    [InlineData("TransformerTTS")]
    public void BaseTypedDecoderWidthChangesNativeMelProjectionAndSynthesis(string family)
    {
        var options = Options(family);
        using var control = Create(options);
        var copy = (AcousticModelOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(options);
        copy.DecoderDim = 10;
        using var changed = Create(copy);
        Assert.Equal((8, 10, 10, 16), (options.MelChannels, copy.MelChannels, copy.DecoderDim, copy.HiddenDim));
        Assert.NotEqual(control.GetParameters().Length, changed.GetParameters().Length);
        var mel = changed switch
        {
            FastSpeech<double> fast => fast.Synthesize("hi"),
            FastSpeech2<double> fast2 => fast2.Synthesize("hi"),
            AlignTTS<double> align => align.Synthesize("hi"),
            TransformerTTS<double> transformer => transformer.Synthesize("hi"),
            _ => throw new InvalidOperationException(),
        };
        Assert.Equal(10, mel.Shape[1]);
        Assert.All(mel, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData("FastSpeech", false)]
    [InlineData("FastSpeech", true)]
    [InlineData("FastSpeech2", false)]
    [InlineData("FastSpeech2", true)]
    [InlineData("AlignTTS", false)]
    [InlineData("AlignTTS", true)]
    [InlineData("TransformerTTS", false)]
    [InlineData("TransformerTTS", true)]
    public void LegacyMelGeometryWinsOverStaleSharedWidthInEitherPropertyOrder(string family, bool reverse)
    {
        var source = Options(family);
        var json = JObject.FromObject(source);
        json["DecoderDim"] = 80;
        if (reverse) json = new JObject(json.Properties().Reverse().Select(p => new JProperty(p.Name, p.Value.DeepClone())));
        var restored = (AcousticModelOptions)JsonConvert.DeserializeObject(json.ToString(), source.GetType())!;
        Assert.Equal((8, 8), (restored.MelChannels, restored.DecoderDim));
        var partial = (AcousticModelOptions)JsonConvert.DeserializeObject("{\"DecoderDim\":12}", source.GetType())!;
        Assert.Equal((12, 12), (partial.MelChannels, partial.DecoderDim));
        using var originalModel = Create(source);
        using var restoredModel = Create(restored);
        Assert.Equal(originalModel.GetParameters().Length, restoredModel.GetParameters().Length);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.2)]
    public void TransformerConfiguredRateAndOptionalCoupledL2ReachItsActualAdam(double decay)
    {
        var options = (TransformerTTSOptions)Options("TransformerTTS");
        options.LearningRate = 0.002;
        options.WeightDecay = decay;
        using var model = Create(options);
        var optimizer = Assert.IsAssignableFrom<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>(
            typeof(TransformerTTS<double>).GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
        var configured = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
        Assert.Equal(0.002, configured.InitialLearningRate);
        Assert.Equal(decay > 0, configured.RegularizationExplicitlySet);
        if (decay > 0)
            Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(configured.Regularization).GetOptions().Strength);
    }

    public sealed class CustomDeepVoiceOptions : DeepVoice3Options
    {
        public string CustomMarker { get; set; } = "marker";
    }

    public sealed class CustomFastSpeechOptions : FastSpeechOptions
    {
        public string CustomMarker { get; set; } = "marker";
    }

    [Fact]
    public void InheritedOptionsConvertersPreserveDerivedConfigurationTypesAndMembers()
    {
        var deep = JsonConvert.DeserializeObject<CustomDeepVoiceOptions>("{\"CustomMarker\":\"deep\",\"EmbeddingDim\":18,\"EncoderDim\":256}")!;
        Assert.Equal("deep", deep.CustomMarker);
        Assert.Equal(18, deep.EncoderDim);
        var fast = JsonConvert.DeserializeObject<CustomFastSpeechOptions>("{\"CustomMarker\":\"fast\",\"MelChannels\":10,\"DecoderDim\":80}")!;
        Assert.Equal("fast", fast.CustomMarker);
        Assert.Equal(10, fast.DecoderDim);
    }
}
