using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class Mars5OptionFixture
{
    internal static bool IsNarOption(string option) => option is "NarWeightDecay" or "NarEndSteps" or "NarBeta1"
        or "NarBeta2" or "NarEndLearningRate" or "NarLearningRate" or "NarPromptMinSeconds" or "NarPromptMaxSeconds"
        or "IncludeKlPrior" or "ConditionDropProbability";

    internal static void PrepareOptions(MARS5TTSOptions options, string option)
    {
        if (!IsNarOption(option)) return;
        options.CurrentStage = Mars5Stage.NonAutoregressive;
        options.HopSize = 320;
        options.Codec.Ratios = new[] { 8, 5, 4, 2 };
        options.Codec.TargetBandwidthKbps = 24000.0 / options.HopSize * options.NumCodebooks * Math.Log(options.CodebookSize, 2) / 1000;
        options.WarmupSteps = 0;
        options.NarEndSteps = 4;
        options.NarPromptMinSeconds = 2.0 * options.HopSize / options.SampleRate;
        options.NarPromptMaxSeconds = 6.0 * options.HopSize / options.SampleRate;
    }

    private static TtsTrainingSample<T> Sample<T>(MARS5TTS<T> model)
    {
        var options = (MARS5TTSOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var audio = new Tensor<T>(new[] { 12 * options.HopSize });
        for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(.3 * Math.Sin(2 * Math.PI * 220 * i / options.SampleRate));
        return new TtsTrainingSample<T> { Tokens = new Tensor<T>(new[] { 1 }), Text = "short", Audio = audio };
    }

    internal static void Train<T>(MARS5TTS<T> model) => model.Train(Sample(model));

    internal static long? Observe<T>(MARS5TTS<T> model, string option)
    {
        if (option != "IncludeKlPrior") return null;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return TtsPaperNumerics.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(Sample(model))) });
    }
}
