using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

internal static class AudioPalmAsrOptionFixture
{
    internal static bool IsTrainingOption(string option) => option is "LearningRate" or "Beta1" or "DropoutRate"
        or "ZLossWeight" or "MaxGradientNorm" or "EndId" or "RampWarmupSteps" or "RampDecaySteps"
        or "RampPeakLearningRate" or "RampFinalLearningRate";

    internal static void PrepareOptions(AudioPaLMOptions options, string option)
    {
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
