using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>All five native components at framework-test scale; full released sampling is audited separately.</summary>
internal static class TortoiseNativeFixture
{
    internal static TortoiseTTS<T> Create<T>() => new(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
        NeuralNetworkTaskType.Generative, inputSize: 4, outputSize: 256) { RandomSeed = 2294 }, new TortoiseTTSOptions
    {
        SampleRate = 1600, TokenizerSampleRate = 1600, MelChannels = 8, AutoregressiveMelChannels = 8,
        FftSize = 32, AutoregressiveFftSize = 32, HopSize = 8, AutoregressiveHopSize = 8,
        ConditioningSamples = 64, DiffusionConditioningSamples = 64, VocabSize = 256, MaxTextLength = 128,
        TextStartToken = 255, TextStopToken = 0, CodebookSize = 8, SpeechStartToken = 8, SpeechStopToken = 9,
        LLMDim = 8, NumLLMLayers = 1, NumEncoderLayers = 1, NumHeads = 2,
        AutoregressiveFeedForwardDim = 16, MaxCodecFrames = 32, MaximumAutoregressiveTokens = 4,
        NumAutoregressiveSamples = 2, MinimumAutoregressiveTokens = 1,
        ClvpDim = 8, ClvpLayers = 1, ClvpHeads = 2, ClvpTextLength = 128, ClvpSpeechLength = 64,
        DiffusionDim = 8, DiffusionLayers = 2, DiffusionHeads = 2,
        DiffusionTrainingSteps = 1000, DiffusionInferenceSteps = 2,
        TokenizerLatentDim = 8, TokenizerCodeDim = 2, UnivNetNoiseDim = 4, UnivNetChannels = 4,
        UnivNetUpsampleRates = new[] { 2, 2, 2 }, UnivNetDilations = new[] { 1, 3 },
        UnivNetKernelPredictorHidden = 4, UnivNetPaddingFrames = 1,
        VocoderTraining = new UnivNetOptions
        {
            StftFftSizes = new[] { 16, 32 }, StftHopSizes = new[] { 4, 8 }, StftWindowSizes = new[] { 16, 24 },
            ResolutionDiscriminatorChannels = 2, DiscriminatorPeriods = new[] { 2, 3 },
            PeriodDiscriminatorChannels = new[] { 2, 2, 2, 2, 2 }, PretrainingSteps = 0
        }
    });
}

internal static class TortoiseOptionFixture
{
    internal static bool Handles(string option) => option is "UnivNetUpsampleRates" or "AutoregressiveMelNormalization"
        or "ClvpSpeechLength" or "ClvpTokenDropout" or "DiffusionConditioningDropout" or "TokenizerCommitmentWeight";

    internal static void Prepare(TortoiseTTSOptions options, string option)
    {
        if (option == "UnivNetUpsampleRates")
        {
            options.UnivNetUpsampleRates = new[] { 2, 2, 4 };
            options.HopSize = 16;
        }
        if (option == "AutoregressiveMelNormalization")
            options.AutoregressiveMelNormalization = Enumerable.Repeat(1.0, options.AutoregressiveMelChannels).ToArray();
        if (option == "ClvpSpeechLength") options.ClvpSpeechLength = 4;
        if (option == "ClvpTokenDropout") options.TrainingComponent = TortoiseTrainingComponent.Clvp;
        if (option == "DiffusionConditioningDropout") options.TrainingComponent = TortoiseTrainingComponent.Diffusion;
        if (option == "TokenizerCommitmentWeight") options.TrainingComponent = TortoiseTrainingComponent.SpeechTokenizer;
    }

    internal static long? Observe<T>(TortoiseTTS<T> model, string option)
    {
        var options = (TortoiseTTSOptions)model.GetOptions();
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option is "ClvpSpeechLength" or "ClvpTokenDropout")
        {
            var text = Enumerable.Range(0, 16).Select(i => 1 + i % (options.VocabSize - 2)).ToArray();
            var speech = Enumerable.Range(0, 16).Select(i => i % options.CodebookSize).ToArray();
            var score = model.ClvpModel!.ScoreTensor(text, speech, option == "ClvpTokenDropout", new Random(1337));
            return OptionSensitivityHarness.Fingerprint(score.Select(value => numeric.ToDouble(value)));
        }
        if (option is not ("AutoregressiveMelNormalization" or "DiffusionConditioningDropout" or "TokenizerCommitmentWeight"))
            return null;
        var audio = new Tensor<T>(new[] { 128 });
        for (int i = 0; i < audio.Length; i++) audio[i] = numeric.FromDouble(0.2 * Math.Sin(i * 0.19) + 0.1 * Math.Cos(i * 0.37));
        if (option == "AutoregressiveMelNormalization")
        {
            var mel = (Tensor<T>)typeof(TortoiseTTS<T>).GetMethod("AutoregressiveMel",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.Invoke(model, new object[] { audio })!;
            return OptionSensitivityHarness.Fingerprint(mel.Select(value => numeric.ToDouble(value)));
        }
        var tokens = new Tensor<T>(new[] { 4 });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = numeric.FromDouble(i + 1);
        var loss = model.EvaluateTrainingObjective(new TtsTrainingSample<T>
        {
            Tokens = tokens, Audio = audio, CodecTokens = tokens,
        });
        return OptionSensitivityHarness.Fingerprint(new[] { numeric.ToDouble(loss) });
    }
}
