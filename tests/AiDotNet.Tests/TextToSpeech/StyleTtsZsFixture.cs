using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.StyleEmotion;

namespace AiDotNet.Tests.TextToSpeech;

internal static class StyleTtsZsFixture
{
    internal static StyleTTSZSOptions Options() => new()
    {
        ModelDim = 16, StyleDim = 16, Heads = 2, FeedForwardDim = 32,
        PromptTextKernels = new[] { 3 }, ProsodyEncoderKernels = new[] { 3 }, ProsodyDecoderKernels = new[] { 3 },
        DenoiserKernels = new[] { 3 }, DiscriminatorKernels = new[] { 3 },
        MaxPhonemes = 64, StyleLength = 4, Quantizers = 2, CodebookSize = 8, CodeDim = 2, SamplingSteps = 4,
        ResidualDim = 8, AlignerHidden = 16, AlignerTokenEmbedding = 8, AlignerConvLayers = 1,
        PitchExtractorChannels = new[] { 4, 8, 8, 8 }, PitchExtractorLstm = 8, TmaStartStep = 1000,
        DecoderBottleneck = 16, UpsampleInitialChannels = 16, UpsampleRates = new[] { 2, 4 },
        UpsampleKernels = new[] { 4, 8 }, HopSize = 8, FftSize = 64, WindowLength = 64, WavePadding = 40, PromptFrames = 96,
        BertHidden = 16, BertLayers = 2, BertHeads = 2, BertIntermediate = 32, BertEmbedding = 8, BertMaxPositions = 64,
        SlmHidden = 8, SlmLayers = 3, SlmHeads = 2, SlmIntermediate = 16, SlmConvChannels = 8,
        SlmPositionalConvKernel = 16, SlmPositionalConvGroups = 2,
        SpeakerChannels = 2, SpeakerBlocks = new[] { 1, 1, 1, 1 }, SpeakerEmbedding = 8,
    };

    internal static StyleTTSZS<T> Create<T>(StyleTTSZSOptions options) => new(
        new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    internal static TtsTrainingSample<T> Sample<T>() => StyleTts2Fixture.Sample<T>();

    internal static bool IsDistillation(string option) => option is "DistillationGuidanceMin" or "DistillationGuidanceMax";

    internal static void PrepareOptions(StyleTTSZSOptions options, string option)
    {
        if (option == "MaxTextLength") options.MaxTextLength = 4;
        if (option is "MonotonicLossWeight" or "SlmSampleRate") options.TmaStartStep = 0;
        if (option == "TmaStartStep") options.TmaStartStep = 1;
    }

    internal static void PrepareModel<T>(StyleTTSZS<T> model, string option) => model.CurrentStage = IsDistillation(option)
        ? StyleTtsZsTrainingStage.Distillation : StyleTtsZsTrainingStage.Training;

    internal static void Train<T>(StyleTTSZS<T> model, string option)
    {
        PrepareModel(model, option);
        var loss = model.Train(Sample<T>());
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (!double.IsFinite(ops.ToDouble(loss))) throw new InvalidOperationException("The native StyleTTS-ZS loss is not finite.");
        if (model.SkippedSamples != 0) throw new InvalidOperationException("The StyleTTS-ZS control recording was skipped.");
    }

    internal static long? Observe<T>(StyleTTSZS<T> model, string option)
    {
        PrepareModel(model, option);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option == "MaxTextLength")
            return TtsPaperNumerics.Fingerprint(model.TextToTokens("hello world, this is a longer sentence").Select(ops.ToDouble));
        if (option == "SlmSampleRate")
            return AiDotNet.Tests.ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(new[]
            {
                ops.ToDouble(model.EvaluateTrainingObjective(Sample<T>(), adversarialTerms: true)),
            });
        if (option == "MinTrainingFrames")
        {
            var source = Sample<T>();
            var sample = new TtsTrainingSample<T>
            {
                Tokens = source.Tokens,
                Audio = new Tensor<T>(new[] { 1024 }, new Vector<T>(source.Audio!.Take(1024).ToArray())),
            };
            try
            {
                return AiDotNet.Tests.ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(new[]
                {
                    ops.ToDouble(model.EvaluateTrainingObjective(sample)),
                });
            }
            catch (ArgumentException ex) when (ex.Message.Contains("evaluates segments of at least", StringComparison.Ordinal))
            {
                return long.MinValue;
            }
        }
        return null;
    }
}
