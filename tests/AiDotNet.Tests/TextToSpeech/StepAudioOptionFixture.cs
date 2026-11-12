using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

internal static class StepAudioOptionFixture
{
    internal static bool IsChatSamplingOption(string option) =>
        option is "ChatTemperature" or "ChatTopP" or "ChatMaxNewTokens";

    internal static bool IsPpoOption(string option) =>
        option is "PpoResponseLength" or "PpoClip" or "CriticWarmupSteps" or "Gamma" or "Lambda" or "KlCoefficient";

    internal static void PreparePpoOptions(StepAudioOptions options, string option)
    {
        options.IncludeRlhfModels = true;
        options.CriticWarmupSteps = option == "CriticWarmupSteps" ? 1 : 0;
        options.WarmupSteps = 0;
        if (option == "PpoClip") options.PpoClip = 0;
    }

    internal static long ObserveChat<T>(StepAudio<T> model)
    {
        var answers = new[] { model.Answer("你好"), model.Answer("今天是星期几？"), model.Answer("Tell me a story.") };
        return OptionSensitivityHarness.Fingerprint(string.Join("\n", answers).Select(character => (double)character));
    }

    internal static long ObserveLinguisticDither<T>(StepAudio<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var audio = Enumerable.Range(0, 6400).Select(i => 0.25 * Math.Sin(2 * Math.PI * 180 * i / 16000)).ToArray();
        return OptionSensitivityHarness.Fingerprint(model.Core.Linguistic.Encode(audio).ToArray().Select(ops.ToDouble));
    }

    internal static void TrainPpo<T>(StepAudio<T> model, int steps)
    {
        var options = (StepAudioOptions)model.GetOptions();
        var recording = StepAudioFixture.Tones<T>(0.4);
        options.Stage = StepAudioTrainingStage.RewardModel;
        model.InitializeRlhfFromChat();
        model.TrainReward(recording, StepAudioFixture.SampleRate, "好的", "不知道");
        model.InitializeCriticFromReward();
        options.Stage = StepAudioTrainingStage.Ppo;
        for (int step = 0; step < Math.Max(2, steps); step++) model.TrainPpo(recording, StepAudioFixture.SampleRate);
    }
}
