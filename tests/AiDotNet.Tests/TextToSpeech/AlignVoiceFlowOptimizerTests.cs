using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.TextToSpeech.Classic;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.EndToEnd;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class AlignVoiceFlowOptimizerTests
{
    private static NeuralNetworkArchitecture<double> Architecture() => new(
        InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 8) { RandomSeed = 2294 };

    private static AlignTTS<double> Align(double rate) => new(Architecture(), new AlignTTSOptions
    {
        EncoderDim = 16, HiddenDim = 16, MelChannels = 8, NumEncoderLayers = 1, NumDecoderLayers = 1,
        NumHeads = 2, FftFilterSize = 32, DurationPredictorDim = 8, DurationPredictorLayers = 1,
        MixDensityHiddenSize = 16, DropoutRate = 0, LearningRate = rate,
    });

    private static VoiceFlowOptions FlowOptions(double rate, double decay) => new()
    {
        HiddenDim = 16, NumHeads = 2, NumEncoderLayers = 1, FilterChannels = 32, PrenetDropout = 0,
        DropoutRate = 0, DurationPredictorFilterChannels = 16, FlowDim = 8, MelChannels = 8,
        SegmentFrames = 12, NumFlowSteps = 3, LearningRate = rate, WeightDecay = decay,
        MaxMelLength = 32, MaxTextLength = 16,
    };

    private static VoiceFlow<double> Flow(double rate, double decay) => new(Architecture(), FlowOptions(rate, decay));

    private static AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> Adam(object model, string field) =>
        Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(
            Assert.IsAssignableFrom<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>(
                model.GetType().GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)).GetOptions());

    [Theory]
    [InlineData(1e-4)]
    [InlineData(0.002)]
    public void AlignUsesPublishedNoamForEarlyStagesAndConfiguredFixedRateForLateStages(double rate)
    {
        // Zeng et al., Sec. 4.1 adopts Vaswani Sec. 5.3 for phases 1/2; phases 3/4 use fixed Adam.
        using var model = Align(rate);
        var early = Adam(model, "_optimizer");
        var late = Adam(model, "_fineTuningOptimizer");
        foreach (var settings in new[] { early, late })
        {
            Assert.Equal(0.9, settings.Beta1);
            Assert.Equal(0.98, settings.Beta2);
            Assert.Equal(1e-9, settings.Epsilon);
            Assert.False(settings.UseAdaptiveBetas);
        }
        var schedule = Assert.IsType<NoamSchedule>(early.LearningRateScheduler);
        foreach (int completed in new[] { 0, 1, 1999, 3999, 4000, 15999, 39999 })
        {
            double step = completed + 1.0;
            double expected = Math.Pow(16, -0.5) * Math.Min(Math.Pow(step, -0.5), step * Math.Pow(4000, -1.5));
            Assert.Equal(expected, schedule.GetLearningRateAtStep(completed), 14);
        }
        Assert.Null(late.LearningRateScheduler);
        Assert.Equal(rate, late.InitialLearningRate);
    }

    [Theory]
    [InlineData(5e-5, 0)]
    [InlineData(0.002, 0.2)]
    public void VoiceFlowUsesAuthorAdamWithConfiguredRateAndOptionalCoupledL2(double rate, double decay)
    {
        // X-LANCE/VoiceFlow-TTS train.py uses torch.optim.Adam, whose decay is coupled L2.
        using var model = Flow(rate, decay);
        var settings = Adam(model, "_optimizer");
        Assert.Equal(rate, settings.InitialLearningRate);
        Assert.Equal(0.9, settings.Beta1);
        Assert.Equal(0.999, settings.Beta2);
        Assert.Equal(1e-8, settings.Epsilon);
        Assert.False(settings.UseAdaptiveBetas);
        Assert.Null(settings.LearningRateScheduler);
        Assert.Equal(decay > 0, settings.RegularizationExplicitlySet);
        if (decay > 0)
            Assert.Equal(decay, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidRatesAreRejectedBeforeNativeAllocation(double rate)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Align(rate));
        Assert.Throws<ArgumentOutOfRangeException>(() => Flow(rate, 0));
    }

    [Fact]
    public void VoiceFlowDefaultWidthsMatchIndependentAuthorEncoderAndUnetConfiguration()
    {
        // configs/lj_16k_gt_dur.yaml: n_enc_channels=192 and dec_dim=128.
        EndToEndTtsOptions options = new VoiceFlowOptions();
        Assert.Equal((192, 128), (options.EncoderDim, options.DecoderDim));
    }

    [Fact]
    public void SharedDecoderAliasResizesNativeUnetWithoutChangingTextEncoder()
    {
        var original = FlowOptions(5e-5, 0);
        original.HiddenDim = 24;
        using var control = new VoiceFlow<double>(Architecture(), original);
        EndToEndTtsOptions copy = (VoiceFlowOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(original);
        copy.DecoderDim = 16;
        using var variant = new VoiceFlow<double>(Architecture(), (VoiceFlowOptions)copy);
        Assert.Equal((24, 16, 16), (copy.HiddenDim, copy.DecoderDim, ((VoiceFlowOptions)copy).FlowDim));
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var embedding = typeof(VoiceFlow<double>).GetField("_embedding", flags)!.GetValue(variant)!;
        var estimator = typeof(VoiceFlow<double>).GetField("_estimator", flags)!.GetValue(variant)!;
        Assert.Equal(24, embedding.GetType().GetField("_embeddingDimension", flags)!.GetValue(embedding));
        Assert.Equal(16, estimator.GetType().GetField("_dim", flags)!.GetValue(estimator));
        Assert.NotEqual(control.GetParameters().Length, variant.GetParameters().Length);
        var mel = variant.Synthesize("hi");
        Assert.Equal(8, mel.Shape[1]);
        Assert.All(mel, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LegacyVoiceFlowJsonRetainsNativeEncoderAndFlowWidthsInEitherOrder(bool reverse)
    {
        var source = FlowOptions(5e-5, 0);
        source.HiddenDim = 24;
        var json = JObject.FromObject(source);
        json["DecoderDim"] = 24; // Previous alias pointed at the text encoder.
        if (reverse) json = new JObject(json.Properties().Reverse().Select(p => new JProperty(p.Name, p.Value.DeepClone())));
        var restored = JsonConvert.DeserializeObject<VoiceFlowOptions>(json.ToString())!;
        Assert.Equal((24, 8, 8), (restored.HiddenDim, restored.FlowDim, restored.DecoderDim));
        using var model = new VoiceFlow<double>(Architecture(), restored);
        using var original = new VoiceFlow<double>(Architecture(), source);
        Assert.Equal(original.GetParameters().Length, model.GetParameters().Length);
        var partial = JsonConvert.DeserializeObject<VoiceFlowOptions>("{\"DecoderDim\":16}")!;
        Assert.Equal((192, 16, 16), (partial.HiddenDim, partial.FlowDim, partial.DecoderDim));
    }

    [Theory]
    [InlineData(AlignTTSTrainingPhase.Alignment, true)]
    [InlineData(AlignTTSTrainingPhase.Decoder, true)]
    [InlineData(AlignTTSTrainingPhase.JointFineTuning, false)]
    [InlineData(AlignTTSTrainingPhase.DurationPredictor, false)]
    public void AlignTrainingStepsOnlyTheOptimizerForItsPublishedPhase(AlignTTSTrainingPhase phase, bool scheduled)
    {
        using var model = Align(0.002);
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        var early = Assert.IsAssignableFrom<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(
            typeof(AlignTTS<double>).GetField("_optimizer", flags)!.GetValue(model));
        var late = Assert.IsAssignableFrom<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(
            typeof(AlignTTS<double>).GetField("_fineTuningOptimizer", flags)!.GetValue(model));
        double firstRate = Math.Pow(16, -0.5) * Math.Pow(4000, -1.5);
        Assert.Equal(firstRate, early.GetCurrentLearningRate(), 14);
        Assert.Equal(0.002, late.GetCurrentLearningRate());
        model.CurrentPhase = phase;
        var tokens = new Tensor<double>(new[] { 3 });
        for (int i = 0; i < 3; i++) tokens[i] = i + 1;
        var mel = new Tensor<double>(new[] { 6, 8 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(0.13 * i);
        var before = model.GetParameters().ToArray();
        model.Train(tokens, mel);
        Assert.False(before.SequenceEqual(model.GetParameters().ToArray()));
        Assert.Equal((scheduled ? 2 : 1) * firstRate, early.GetCurrentLearningRate(), 14);
        Assert.Equal(0.002, late.GetCurrentLearningRate());
    }

    public sealed class CustomFlowOptions : VoiceFlowOptions
    {
        public string CustomMarker { get; set; } = "marker";
    }

    [Fact]
    public void VoiceFlowConverterPreservesDerivedOptions()
    {
        var options = JsonConvert.DeserializeObject<CustomFlowOptions>("{\"CustomMarker\":\"custom\",\"FlowDim\":8,\"DecoderDim\":192}")!;
        Assert.Equal("custom", options.CustomMarker);
        Assert.Equal((192, 8), (options.HiddenDim, options.DecoderDim));
    }
}
