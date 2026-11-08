using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class Mars5OptionFixture
{
    internal static bool IsNarOption(string option) => option is "NarWeightDecay" or "NarEndSteps" or "NarBeta1"
        or "NarBeta2" or "NarEndLearningRate" or "NarLearningRate" or "NarPromptMinSeconds" or "NarPromptMaxSeconds"
        or "IncludeKlPrior" or "ConditionDropProbability";

    private static bool IsGenerationOption(string option) => option is "EosPenaltyDecay" or "PresencePenalty"
        or "FrequencyPenalty" or "TrimDb" or "JumpLength" or "ReferencePadSeconds" or "DeepClone";

    internal static void PrepareOptions(MARS5TTSOptions options, string option)
    {
        if (!IsNarOption(option) && !IsGenerationOption(option)) return;
        options.CurrentStage = IsNarOption(option) ? Mars5Stage.NonAutoregressive : Mars5Stage.Autoregressive;
        options.HopSize = 320;
        options.Codec.Ratios = new[] { 8, 5, 4, 2 };
        options.Codec.TargetBandwidthKbps = 24000.0 / options.HopSize * options.NumCodebooks * Math.Log(options.CodebookSize, 2) / 1000;
        options.WarmupSteps = 0;
        options.NarEndSteps = 4;
        options.NarPromptMinSeconds = 2.0 * options.HopSize / options.SampleRate;
        options.NarPromptMaxSeconds = 6.0 * options.HopSize / options.SampleRate;
        if (IsGenerationOption(option))
        {
            options.MaxGenerationLength = 32;
            options.SlidingWindow = 64;
            options.TopK = 0;
            options.TopP = 1;
            options.TypicalP = 1;
        }
        if (option == "JumpLength") options.JumpSamples = 2;
        if (option == "PresencePenalty") options.FrequencyPenalty = 0;
        if (option == "FrequencyPenalty") { options.PresencePenalty = 0; options.FrequencyPenalty = .5; }
        if (option == "EosPenaltyDecay") options.EosPenaltyFactor = .5;
        if (option == "TrimDb") options.TrimDb = 1;
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
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        if (option == "IncludeKlPrior")
            return TtsPaperNumerics.Fingerprint(new[] { ops.ToDouble(model.EvaluateTrainingObjective(Sample(model))) });
        if (!IsGenerationOption(option)) return null;
        var options = (MARS5TTSOptions)model.GetOptions();
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        if (option is "EosPenaltyDecay" or "PresencePenalty" or "FrequencyPenalty")
        {
            var reference = new int[6, options.NumCodebooks];
            for (int f = 0; f < 6; f++)
                for (int q = 0; q < options.NumCodebooks; q++) reference[f, q] = (f + q) % options.CodebookSize;
            var sequence = model.TextToTokens("hi").Select(value => (int)ops.ToDouble(value)).ToList();
            var generate = typeof(MARS5TTS<T>).GetMethod("Generate", flags)!;
            var values = new List<double>();
            for (int seed = 0; seed < 16; seed++)
            {
                var tokens = (List<int>)generate.Invoke(model, new object[] { sequence, reference, sequence.Count + 1, 8, new Random(1337 + seed) })!;
                values.Add(tokens.Count);
                values.AddRange(tokens.Select(token => (double)token));
            }
            return TtsPaperNumerics.Fingerprint(values);
        }
        var voice = model.CreateVoice(Sample(model).Audio!, options.SampleRate, "hi");
        if (option == "ReferencePadSeconds")
        {
            var prompt = typeof(MARS5TTS<T>).GetMethod("Prompt", flags)!.Invoke(model, new object[] { voice })!;
            var codes = (int[,])prompt.GetType().GetField("Codes")!.GetValue(prompt)!;
            return TtsPaperNumerics.Fingerprint(new[] { (double)codes.GetLength(0) }.Concat(codes.Cast<int>().Select(code => (double)code)));
        }
        var previousVoice = model.Voice;
        try
        {
            model.Voice = voice;
            return TtsPaperNumerics.Fingerprint(model.Synthesize("hi").Select(ops.ToDouble));
        }
        finally { model.Voice = previousVoice; }
    }
}
