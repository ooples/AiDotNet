using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class CosyVoiceOptimizerOptionsTests
{
    private static CosyVoiceFamilyOptions Options(string family)
    {
        if (family == "v3") return CosyVoice3Fixture.Options();
        if (family == "v2") return new CosyVoice2Options
        {
            LLMDim = 16, NumLLMLayers = 1, NumHeads = 2, KeyValueHeads = 1, LmFeedForwardDim = 32,
            FlowDim = 16, FlowHeads = 2, FlowFeedForwardDim = 32, FlowEncoderLayers = 1,
            DecoderChannels = [16], DecoderHeads = 2, DecoderHeadDim = 8, DecoderBlocks = 1, DecoderMidBlocks = 1,
            FlowTimesteps = 2, HiftBaseChannels = 16, HiftHarmonics = 1, HiftUpsampleRates = [2],
            HiftUpsampleKernels = [4], HiftSourceKernels = [3], HopSize = 8, FftSize = 32,
            SpeechTokenizerState = 64, SpeechTokenizerHeads = 1, SpeechTokenizerLayers = 1,
            SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 4, SpeakerEmbeddingDim = 8,
            MelChannels = 8, DropoutRate = 0, IncludeDiscriminators = false,
        };
        CosyVoiceOptions o = family switch
        {
            "v1" => new CosyVoiceOptions(),
            "clone" => new CosyVoiceCloneOptions(), _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
        o.LLMDim = 16; o.NumLLMLayers = 1; o.NumHeads = 2; o.LmFeedForwardDim = 32;
        o.FlowDim = 16; o.FlowHeads = 2; o.FlowFeedForwardDim = 32; o.FlowEncoderLayers = 1;
        o.DecoderChannels = [16]; o.DecoderHeads = 2; o.DecoderHeadDim = 8;
        o.DecoderBlocks = o.DecoderMidBlocks = 1; o.FlowTimesteps = 2;
        o.HiftBaseChannels = 16; o.HiftHarmonics = 1; o.HiftUpsampleRates = [2];
        o.HiftUpsampleKernels = [4]; o.HiftSourceKernels = [3]; o.HopSize = 8; o.FftSize = 32;
        o.SpeechTokenizerState = 16; o.SpeechTokenizerHeads = 2;
        o.SpeechTokenizerLayers = 1; o.SpeakerEncoderChannels = 8; o.SpeakerEncoderGrowthRate = 4;
        o.SpeakerEmbeddingDim = 8; o.MelChannels = 8; o.DropoutRate = 0; o.IncludeDiscriminators = false;
        o.TextEncoderDim = 16; o.NumEncoderLayers = 1; o.CodebookSize = 16;
        return o;
    }

    private static CosyVoiceBase<double> Create(CosyVoiceFamilyOptions o)
    {
        var arch = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1) { RandomSeed = 2294 };
        return o switch
        {
            CosyVoiceCloneOptions clone => new CosyVoiceClone<double>(arch, clone),
            CosyVoiceOptions v1 => new CosyVoice<double>(arch, v1),
            CosyVoice2Options v2 => new CosyVoice2<double>(arch, v2),
            CosyVoice3Options v3 => new CosyVoice3<double>(arch, v3),
            _ => throw new ArgumentOutOfRangeException(nameof(o)),
        };
    }

    private static GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> Optimizer(CosyVoiceBase<double> model, string component)
    {
        // Ask the real stage cache for the optimizer used by Train, including the GAN's separate cache.
        const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
        object? optimizer;
        if (component is "language model" or "flow")
        {
            model.CurrentStage = component == "flow" ? CosyVoiceTrainingStage.Flow : CosyVoiceTrainingStage.LanguageModel;
            optimizer = typeof(CosyVoiceBase<double>).GetMethod("StageOptimizer", flags)!.Invoke(model, null);
        }
        else
        {
            model.CurrentStage = CosyVoiceTrainingStage.Vocoder;
            optimizer = typeof(CosyVoiceBase<double>).GetMethod("VocoderOptimizer", flags)!.Invoke(model, [component]);
        }
        return Assert.IsAssignableFrom<GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>>(optimizer);
    }

    [Theory]
    [InlineData("v1", "language model")]
    [InlineData("v1", "flow")]
    [InlineData("v1", "vocoder")]
    [InlineData("v1", "discriminator")]
    [InlineData("v2", "language model")]
    [InlineData("v2", "flow")]
    [InlineData("v2", "vocoder")]
    [InlineData("v2", "discriminator")]
    [InlineData("v3", "language model")]
    [InlineData("v3", "flow")]
    [InlineData("v3", "vocoder")]
    [InlineData("v3", "discriminator")]
    [InlineData("clone", "language model")]
    [InlineData("clone", "flow")]
    [InlineData("clone", "vocoder")]
    [InlineData("clone", "discriminator")]
    public void NativeStageAdamMatchesReleasedRecipeDefaults(string family, string component)
    {
        // FunAudioLLM/CosyVoice examples/libritts/{cosyvoice,cosyvoice2,cosyvoice3}/conf/*.yaml.
        var options = Options(family);
        using var model = Create(options);
        var actual = Optimizer(model, component);
        var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(actual.GetOptions());
        bool lmOrFlow = component is "language model" or "flow";
        double expected = lmOrFlow ? family is "v1" or "clone" ? 1e-3 : 1e-5 : 2e-4;
        Assert.Equal(expected, settings.InitialLearningRate);
        Assert.Equal(family is "v1" or "clone" ? 1e-3 : 1e-5, options.LearningRate);
        Assert.Equal(0, options.WeightDecay);
        Assert.Equal(0.9, settings.Beta1); Assert.Equal(0.999, settings.Beta2); Assert.Equal(1e-8, settings.Epsilon);
        Assert.False(settings.UseAdaptiveBetas);
        Assert.False(settings.RegularizationExplicitlySet);
        Assert.True(settings.EnableGradientClipping);
        Assert.Equal(5, settings.MaxGradientNorm);
        Assert.Equal(5.0, typeof(NeuralNetworkBase<double>).GetField("MaxGradNorm",
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model));
    }

    [Theory]
    [InlineData("v1")]
    [InlineData("v2")]
    [InlineData("v3")]
    [InlineData("clone")]
    public void ConfiguredRateAndCoupledDecayReachActualNativeStageOptimizers(string family)
    {
        var options = Options(family);
        options.LearningRate = 0.002; options.WeightDecay = 0.2;
        using var model = Create(options);
        foreach (string component in new[] { "language model", "flow", "vocoder", "discriminator" })
        {
            var settings = Assert.IsType<AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>>(Optimizer(model, component).GetOptions());
            Assert.Equal(component is "language model" or "flow" ? 0.002 : 2e-4, settings.InitialLearningRate);
            Assert.True(settings.RegularizationExplicitlySet);
            Assert.Equal(0.2, Assert.IsType<L2Regularization<double, Tensor<double>, Tensor<double>>>(settings.Regularization).GetOptions().Strength);
        }
    }

    [Theory]
    [InlineData("v1", 0)]
    [InlineData("v1", 3)]
    [InlineData("v1", 2500)]
    [InlineData("clone", 0)]
    [InlineData("clone", 3)]
    [InlineData("clone", 2500)]
    [InlineData("v2", 0)]
    [InlineData("v2", 1)]
    [InlineData("v2", 3)]
    [InlineData("v2", 2500)]
    [InlineData("v3", 0)]
    [InlineData("v3", 1)]
    [InlineData("v3", 3)]
    [InlineData("v3", 2500)]
    public void NativeWarmupUsesAuthorUpdateCountsAndUnboundedCurve(string family, int warmup)
    {
        // cosyvoice/utils/scheduler.py WarmupLR.get_lr uses t=last_epoch+1; it is not a 100-step Optimize loop.
        var options = Options(family);
        options.LearningRate = 0.002; options.WarmupSteps = warmup;
        using var model = Create(options);
        var optimizer = Optimizer(model, "language model");
        bool inverseRoot = family is "v1" or "clone";
        foreach (int completed in new[] { 0, 1, 2, 99, 2499, 2500, 9999, 99999 })
        {
            double t = completed + 1.0;
            double factor = inverseRoot
                ? warmup == 0 ? Math.Pow(t, -0.5) : Math.Sqrt(warmup) * Math.Min(Math.Pow(t, -0.5), t * Math.Pow(warmup, -1.5))
                : warmup == 0 ? 1 : Math.Min(t / warmup, 1);
            double expected = options.LearningRate * factor;
            double actual = optimizer.LearningRateScheduler?.GetLearningRateAtStep(completed) ?? optimizer.GetCurrentLearningRate();
            Assert.Equal(expected, actual, 14);
            if (completed == 0) Assert.Equal(expected, optimizer.GetCurrentLearningRate(), 14);
        }
    }

    [Theory]
    [InlineData("v1")]
    [InlineData("v2")]
    [InlineData("v3")]
    [InlineData("clone")]
    public void CoupledDecayTrainsOnlyTheActiveNativeComponent(string family)
    {
        var options = Options(family);
        options.LearningRate = 0.002; options.WeightDecay = 0.2; options.WarmupSteps = 0;
        using var model = Create(options);
        var tokens = new Tensor<double>(new[] { 3 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = 2 * i + 1;
        var speech = new Tensor<double>(new[] { 20 });
        for (int i = 0; i < speech.Length; i++) speech[i] = (7 * i + 3) % options.CodebookSize;
        var mel = new Tensor<double>(new[] { 43, 8 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(0.13 * i) - 4;
        var speaker = new Tensor<double>(new[] { 8 });
        for (int i = 0; i < speaker.Length; i++) speaker[i] = Math.Cos(i + 1.0);
        var sample = new TtsTrainingSample<double> { Tokens = tokens, CodecTokens = speech, Mel = mel, SpeakerReference = speaker };
        static double[] Snapshot(IEnumerable<AiDotNet.NeuralNetworks.Layers.LayerBase<double>> layers) =>
            AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList(), -1)
                .SelectMany(p => p.ToArray()).ToArray();
        foreach (var stage in new[] { CosyVoiceTrainingStage.LanguageModel, CosyVoiceTrainingStage.Flow })
        {
            model.CurrentStage = stage;
            var lm = Snapshot(model.LanguageModelLayers);
            var flow = Snapshot(model.FlowLayers);
            var frozen = Snapshot(model.FrozenLayers);
            for (int i = 0; i < 2; i++) Assert.True(TtsPaperNumerics.IsFinite(model.Train(sample)));
            Assert.Equal(stage == CosyVoiceTrainingStage.LanguageModel, !lm.SequenceEqual(Snapshot(model.LanguageModelLayers)));
            Assert.Equal(stage == CosyVoiceTrainingStage.Flow, !flow.SequenceEqual(Snapshot(model.FlowLayers)));
            Assert.Equal(frozen, Snapshot(model.FrozenLayers));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void NativeConstructorRefusesInvalidRates(double rate)
    {
        var options = CosyVoice3Fixture.Options();
        options.LearningRate = rate;
        Assert.Throws<ArgumentOutOfRangeException>(() => Create(options));
    }
}
