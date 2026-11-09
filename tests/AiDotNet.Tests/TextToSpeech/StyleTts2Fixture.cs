using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.StyleEmotion;

namespace AiDotNet.Tests.TextToSpeech;

internal static class StyleTts2Fixture
{
    internal static StyleTTS2Options Options() => new()
    {
        HiddenDim = 16, StyleDim = 8, TextLayers = 2, StyleEncoderDim = 4, MaxConvDim = 16,
        ResidualDim = 8, AlignerHidden = 16, AlignerTokenEmbedding = 8, AlignerConvLayers = 1,
        PitchExtractorChannels = new[] { 4, 8, 8, 8 }, PitchExtractorLstm = 8, TmaStartStep = 1000,
        DurationBins = 6, DecoderBottleneck = 16, UpsampleInitialChannels = 16,
        UpsampleRates = new[] { 2, 4 }, UpsampleKernels = new[] { 4, 8 }, HopSize = 8,
        FftSize = 64, WindowLength = 64, WavePadding = 40, TrimEndSamples = 0,
        BertHidden = 16, BertLayers = 2, BertHeads = 2, BertIntermediate = 32, BertEmbedding = 8, BertMaxPositions = 64,
        DiffusionLayers = 1, DiffusionHeads = 2, DiffusionHeadFeatures = 4, DiffusionMaxLength = 64,
        SlmHidden = 8, SlmLayers = 3, SlmHeads = 2, SlmIntermediate = 16, SlmConvChannels = 8,
        SlmPositionalConvKernel = 16, SlmPositionalConvGroups = 2, SlmDiscriminatorChannels = 4,
        DiscriminatorWidthDivisor = 16, MaxTrainingFrames = 128,
    };

    internal static StyleTTS2<double> Create(StyleTTS2Options options) => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1) { RandomSeed = 7 }, options);

    internal static TtsTrainingSample<double> Sample() => Sample<double>();

    internal static bool IsSlmAdversarialOption(string option) => option is
        "SlmGradientThreshold" or "SlmGradientScale" or "SlmDurationSigma"
        or "SlmDiscriminatorInterval" or "SlmAdversarialMinFrames" or "SlmAdversarialMaxFrames";

    internal static void PrepareSlmAdversarialOptions(StyleTTS2Options options, string option)
    {
        // Keep the native decoder and frozen WavLM. Its 400-sample receptive field must fit
        // in the resampled predicted crop, rather than silently skipping the adversary.
        options.UpsampleRates = new[] { 4, 8 };
        options.UpsampleKernels = new[] { 8, 16 };
        options.HopSize = 32;
        options.SlmAdversarialMinFrames = option == "SlmAdversarialMinFrames" ? 80 : 16;
        options.SlmAdversarialMaxFrames = option == "SlmAdversarialMaxFrames" ? 48 : 128;
        options.SlmDiscriminatorInterval = 3;
    }

    internal static TtsTrainingSample<T> SlmAdversarialSample<T>(int hopSize)
    {
        var sample = Sample<T>(hopSize);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        // Twenty-four phonemes produce long enough predicted durations at six bins/token.
        var tokens = new Tensor<T>(new[] { 24 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = ops.FromDouble(i % 7 + 1);
        return new TtsTrainingSample<T> { Tokens = tokens, Audio = sample.Audio };
    }

    internal static TtsTrainingSample<T> Sample<T>(int hopSize = 8)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var audio = new Tensor<T>(new[] { Math.Max(2048, checked(hopSize * 256)) });
        for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.047) + 0.1 * Math.Cos(i * 0.073));
        return new TtsTrainingSample<T>
        {
            Tokens = new Tensor<T>(new[] { 3 }, new Vector<T>(new[] { ops.One, ops.FromDouble(2), ops.FromDouble(3) })), Audio = audio,
        };
    }
}
