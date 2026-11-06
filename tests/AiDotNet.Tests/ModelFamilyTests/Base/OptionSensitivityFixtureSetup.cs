using AiDotNet.Interfaces;
using AiDotNet.Tests.TextToSpeech;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Replays the fitted assets installed by the generated fixture after configuration-only reconstruction.</summary>
internal static class OptionSensitivityFixtureSetup
{
    internal static void PrepareOptions(AiDotNet.Models.Options.ModelOptions options)
    {
        if (options is not AiDotNet.TextToSpeech.TtsModelOptions) return;
        // Cross an epoch boundary within the probe's two training steps.
        var perEpoch = options.GetType().GetProperty("UpdatesPerEpoch");
        if (perEpoch?.PropertyType == typeof(int) && perEpoch.CanWrite)
            perEpoch.SetValue(options, 1);
    }

    internal static void UseConfiguredOptimizer<T>(INeuralNetworkModel<T> network)
    {
        if (network is not AiDotNet.TextToSpeech.TtsModelBase<T>) return;
        // The generated fixtures inject constant-rate optimizers for convergence invariants.
        // Reconstruct option probes without that override so the paper optimizer reads its settings.
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly;
        for (var type = network.GetType(); type is not null; type = type.BaseType)
        {
            foreach (var field in type.GetFields(flags))
                if (field.Name is "_suppliedOptimizer" or "_optimizer" or "_vocoderBaseOptimizer"
                    && field.FieldType.IsInterface && field.FieldType.Name.StartsWith("IGradientBasedOptimizer", StringComparison.Ordinal))
                    field.SetValue(network, null);
        }
    }

    internal static void Prepare<T>(INeuralNetworkModel<T> network)
    {
        switch (network)
        {
            case AiDotNet.TextToSpeech.Latest.KaniTTS<T> model:
                KaniTTSFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.Latest.KaniTTS2<T> model:
                KaniTTSFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T> model:
                MetaVoiceNativeFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T> model:
                AudioPaLMFixture.Prepare(model);
                break;
            case AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> model:
                AudioPaLMFixture.Prepare(model);
                break;
        }
    }
}
