using System;
using System.Linq;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Exercises Moshi's actual inference, conversation and codec-training paths.</summary>
internal static class MoshiOptionFixture
{
    internal static bool IsSampling(string option) => option is
        "AudioTemperature" or "AudioTopK" or "TextTemperature" or "TextTopK"
        or "TtsTemperature" or "TtsPaddingTarget" or "TtsPaddingBonus" or "SamplingSeed";

    private static bool IsChat(string option) => option is
        "AudioTemperature" or "AudioTopK" or "TextTemperature" or "TextTopK";

    private static bool IsConversation(string option) => option is "TextMaskProbability" or "TextDelaySeconds";

    private static bool IsMimi(string option) => option.StartsWith("WavLm", StringComparison.Ordinal) || option is
        "MimiLearningRate" or "MimiWeightDecay" or "MimiWeightAverageDecay"
        or "MimiQuantizationRate" or "MimiQuantizerDropout" or "MimiAdversarialWeight"
        or "MimiFeatureWeight" or "MimiDistillationWeight" or "MimiDiscriminatorWindows" or "MimiDiscriminatorFilters";

    internal static bool Handles(string option) => IsSampling(option) || IsConversation(option) || IsMimi(option);

    internal static void Prepare(MoshiOptions options, string option)
    {
        if (IsChat(option) || IsConversation(option)) options.Variant = MoshiVariant.Dialogue;
        if (IsConversation(option)) options.Stage = MoshiTrainingStage.PreTraining;
        if (option == "TextMaskProbability") options.TextDelaySeconds = 0;
        if (option == "TextDelaySeconds") options.TextMaskProbability = 0;
        if (!IsMimi(option)) return;
        options.IncludeMimiTraining = true;
        // Two acoustic quantizers are required for dropout to select a proper subset.
        options.Codebooks = 3;
        // Keep the first numeric perturbations inside the EMA coefficient's valid range.
        if (option == "MimiWeightAverageDecay") options.MimiWeightAverageDecay = .5;
    }

    internal static long Observe<T>(Moshi<T> model, string option)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (IsChat(option))
        {
            var result = model.Chat(MoshiFixture.Tones<T>(.32));
            return OptionSensitivityHarness.Combine(
                OptionSensitivityHarness.Fingerprint(result.Audio.Select(ops.ToDouble)),
                OptionSensitivityHarness.Fingerprint(result.Text.Select(c => (double)c)));
        }
        if (option == "MimiWeightAverageDecay") model.ApplyMimiWeightAverage();
        var spoken = model.Speak("the cat sat");
        return OptionSensitivityHarness.Combine(
            OptionSensitivityHarness.Fingerprint(spoken.Audio.Select(ops.ToDouble)),
            OptionSensitivityHarness.Fingerprint(spoken.Words.Select(word => word.Time)));
    }

    internal static bool Train<T>(Moshi<T> model, string option)
    {
        if (IsMimi(option))
        {
            model.TrainMimi(MoshiFixture.Tones<T>(1.0));
            return true;
        }
        if (!IsConversation(option)) return false;
        bool jitter = option == "TextDelaySeconds";
        var words = jitter
            ? new[] { new MoshiWord("the", .65, .75), new MoshiWord("cat", .8, .9), new MoshiWord("sat", .95, 1.05) }
            : MoshiFixture.Words("the cat sat", .32);
        model.TrainConversation(new MoshiConversation<T>
        {
            Moshi = MoshiFixture.Tones<T>(jitter ? 1.5 : .32), Words = words,
        });
        return true;
    }
}
