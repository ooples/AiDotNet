using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Tests.TextToSpeech;

internal static class LlasaOptionFixture
{
    internal static bool IsScheduler(string option) => option is "Epochs" or "WarmupEpochFraction" or "FinalLearningRateFraction";
    internal static bool Handles(string option) => IsScheduler(option) || option == "IncludePromptAudio";

    internal static void Prepare(LlasaOptions options, string option)
    {
        if (!IsScheduler(option)) return;
        // TrainingSteps=0 intentionally selects a constant rate. Use a finite run where a valid
        // change to epochs or warm-up fraction can change the integer number of warm-up updates.
        options.TrainingSteps = 8;
        options.Epochs = 2;
        options.WarmupEpochFraction = .5;
        if (option == "FinalLearningRateFraction")
        {
            options.TrainingSteps = 4;
            options.WarmupEpochFraction = 0;
        }
    }

    private static Tensor<T> Recording<T>(int samples)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(new[] { samples }, new Vector<T>(Enumerable.Range(0, samples)
            .Select(i => ops.FromDouble(.3 * Math.Sin(.02 * i) + .1 * Math.Sin(.11 * i))).ToArray()));
    }

    internal static long ObservePrompt<T>(Llasa<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var wave = model.SynthesizeWithPrompt("General Kenobi.", Recording<T>(1000), "Hello there.");
        return OptionSensitivityHarness.Fingerprint(wave.ToArray().Select(ops.ToDouble));
    }

    internal static void TrainSchedule<T>(Llasa<T> model, int steps)
    {
        if (steps == 0) return;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var ids = model.Tokenizer.Encode("a");
        var sample = new TtsTrainingSample<T>
        {
            Text = "a", Audio = Recording<T>(1000),
            Tokens = new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(id => ops.FromDouble(id)).ToArray())),
        };
        for (int step = 0; step < steps; step++) model.Train(sample);
    }
}
