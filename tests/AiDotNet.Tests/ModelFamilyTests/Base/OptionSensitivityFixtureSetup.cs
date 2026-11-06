using AiDotNet.Interfaces;
using AiDotNet.Tests.TextToSpeech;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Replays the fitted assets installed by the generated fixture after configuration-only reconstruction.</summary>
internal static class OptionSensitivityFixtureSetup
{
    internal static bool UsesFittedAssets<T>(INeuralNetworkModel<T> network)
        => network is AiDotNet.TextToSpeech.Latest.KaniTTSBase<T>
            or AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T>
            or AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T>
            or AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T>;

    internal static long ParameterCountBeforeFitting<T>(INeuralNetworkModel<T> network)
        => UsesFittedAssets(network) ? network.GetParameters().Length : 0;

    internal static string ControlGroup<T>(INeuralNetworkModel<T> network, string option)
        => network is AiDotNet.TextToSpeech.Latest.KaniTTSBase<T> ? KaniControlGroup(option) : string.Empty;

    private static string KaniControlGroup(string option) => option switch
    {
        "LoraRank" or "LoraAlpha" or "LoraDropout" or "LoraTargets" or "RankStabilizedLora" => "lora",
        "TopK" or "TopP" or "SamplingSeed" => "sampling",
        "WarmupRatio" => "warmup-ratio",
        "WarmupSteps" => "warmup-steps",
        "TrainingSteps" => "cosine",
        "RopeAlphaMin" or "RopeAlphaMax" or "RopeAlphaLearningRateRatio" => "learnable-rope",
        "AudioStep" => "frame-positions",
        "LenientSpeechSpans" or "StreamingChunkFrames" or "StreamingLookbackFrames" => "streaming",
        "MaxReferenceSeconds" => "reference-crop",
        _ => string.Empty,
    };

    internal static void PrepareOptions(AiDotNet.Models.Options.ModelOptions options, string option)
    {
        if (options is AiDotNet.TextToSpeech.Latest.KaniTTSOptions kani)
        {
            switch (KaniControlGroup(option))
            {
                case "lora":
                    kani.UseLora = true;
                    break;
                case "sampling":
                    kani.Temperature = 2;
                    kani.LenientSpeechSpans = true;
                    kani.StreamingChunkFrames = 1;
                    kani.StreamingLookbackFrames = 0;
                    break;
                case "warmup-ratio":
                    kani.TrainingSteps = 20;
                    kani.WarmupSteps = 0;
                    break;
                case "warmup-steps":
                    kani.TrainingSteps = 20;
                    kani.WarmupSteps = 2;
                    break;
                case "cosine":
                    kani.TrainingSteps = 20;
                    kani.WarmupSteps = 0;
                    kani.WarmupRatio = 0;
                    break;
                case "learnable-rope":
                    kani.LearnableRope = true;
                    kani.UseLora = false;
                    break;
                case "frame-positions":
                    kani.FrameLevelPositions = true;
                    break;
                case "streaming":
                    kani.LenientSpeechSpans = true;
                    kani.StreamingChunkFrames = 1;
                    kani.StreamingLookbackFrames = 0;
                    break;
                case "reference-crop":
                    if (kani is AiDotNet.TextToSpeech.Latest.KaniTTS2Options kani2) kani2.MaxReferenceSeconds = 0.5;
                    break;
            }
        }
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
