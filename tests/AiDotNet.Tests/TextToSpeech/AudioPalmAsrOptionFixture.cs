using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

internal static class AudioPalmAsrOptionFixture
{
    internal static bool IsTrainingOption(string option) => option is "LearningRate" or "Beta1" or "DropoutRate"
        or "ZLossWeight" or "MaxGradientNorm" or "EndId" or "RampWarmupSteps" or "RampDecaySteps"
        or "RampPeakLearningRate" or "RampFinalLearningRate";

    internal static void PrepareOptions(AudioPaLMOptions options, string option)
    {
        if (option == "SpeechTokenizer") options.W2vBert2Layer = 0;
        if (option == "W2vBert2Layer")
        {
            options.SpeechTokenizer = AudioPalmSpeechTokenizer.W2vBert2;
            options.W2vBert2Layer = 0;
            ConfigureTokenizerRate(options, 50);
        }
        if (option.StartsWith("Ramp", StringComparison.Ordinal))
        {
            options.RampWarmupSteps = 2;
            options.RampDecaySteps = 4;
            options.RampPeakLearningRate = .01;
            options.RampFinalLearningRate = .001;
        }
        if (option == "SamplingSeed") options.Temperature = .9;
        if (option == "MaxGradientNorm") options.MaxGradientNorm = .001;
    }

    internal static void ConfigureTokenizerRate(AudioPaLMOptions options, int rate)
    {
        int hop = options.SampleRate / rate;
        options.SoundStorm.SemanticFrameRate = options.AudioLM.SemanticFrameRate = rate;
        options.SoundStorm.HopSize = hop;
        foreach (var codec in new[] { options.SoundStorm.SoundStream, options.AudioLM.SoundStream })
        {
            codec.Ratios = hop == 320 ? new[] { 8, 5, 4, 2 } : new[] { 8, 5, 4, 4 };
            codec.TargetBandwidthKbps = codec.NumQuantizers * rate * Math.Log(codec.CodebookSize, 2) / 1000.0;
        }
    }

    internal static long? ObserveSynthesis<T>(AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T> model, string option)
    {
        if (option != "VoicePromptSeconds") return null;
        var prompt = (Tensor<T>)model.Core.GetType().GetMethod("VoicePrompt",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
            .Invoke(model.Core, new object[] { AudioPaLMFixture.Tones<T>(.25) })!;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return TtsPaperNumerics.Fingerprint(new[] { (double)prompt.Length }.Concat(prompt.Select(ops.ToDouble)));
    }

    internal static void Train<T>(AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var ids = model.TokenizeText("hello");
        var target = new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(id => ops.FromDouble(id)).ToArray()));
        model.Train(AudioPaLMFixture.Tones<T>(.4), target);
    }

    internal static long? Observe<T>(AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> model, string option)
    {
        if (option != "SamplingSeed") return null;
        var prefix = model.Core.TaskPrefix("[ASR English]", model.Core.AudioIds(model.Core.AudioTokensOf(AudioPaLMFixture.Tones<T>(.4))));
        bool training = model.IsTrainingMode;
        model.SetTrainingMode(false);
        try
        {
            using var scope = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            var output = model.Core.Generate(prefix, audioOutput: false);
            return TtsPaperNumerics.Fingerprint(new[] { (double)output.Length }.Concat(output.Select(id => (double)id)));
        }
        finally { model.SetTrainingMode(training); }
    }
}
