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
        => network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> && IsCosyFlowOption(option) ? "cosy-flow"
            : network is AiDotNet.TextToSpeech.Classic.ForwardTacotron<T> ? option switch
            {
                "SecondStageLearningRate" or "SecondStageStartStep" => "forward-second-stage",
                "GradientClipNorm" => "forward-clipping",
                "PitchMinHz" => "forward-pitch-bound",
                _ => string.Empty,
            }
            : network is AiDotNet.TextToSpeech.Latest.KaniTTSBase<T> ? KaniControlGroup(option)
            : network is AiDotNet.SpeechRecognition.Foundation.SPIRAL<T> && IsSpiralPretrainingOption(option)
                ? "spiral-pretraining" : option switch
                {
                    "FrequencyMaskWidth" => "frequency-masking",
                    "TimeMaskRatio" => "time-masking",
                    "MaxAudioLengthSeconds" => "audio-crop",
                    _ => string.Empty,
                };

    internal static Tensor<T> ObservationInput<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option)
    {
        if (option != "MaxAudioLengthSeconds" || input.Length == 0
            || network.GetType().Namespace?.StartsWith("AiDotNet.SpeechRecognition", StringComparison.Ordinal) != true)
            return input;
        var options = network.GetType().GetMethod("GetOptions", Type.EmptyTypes)?.Invoke(network, null);
        if (options?.GetType().GetProperty("SampleRate")?.GetValue(options) is not int sampleRate || sampleRate < 1)
            return input;
        var preprocess = network.GetType().GetMethod("PreprocessAudio", System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic,
            binder: null, types: new[] { typeof(Tensor<T>) }, modifiers: null);
        if (preprocess is null) return input;
        // Exercise the same clipping front end as Transcribe, beyond the one-second control limit.
        network.SetTrainingMode(false);
        var waveform = new Tensor<T>(new[] { checked(2 * sampleRate) });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < waveform.Length; i++)
            waveform[i] = i < sampleRate ? input[i % input.Length] : ops.Negate(input[i % input.Length]);
        return (Tensor<T>)(preprocess.Invoke(network, new object[] { waveform })
            ?? throw new InvalidOperationException("The audio front end returned no input."));
    }

    private static bool IsSpiralPretrainingOption(string option) => option is
        "TeacherMomentumStart" or "TeacherMomentumEnd" or "PretrainingSteps" or "MaxTeacherPaddingUnits"
        or "TimeMaskProbability" or "TimeMaskLength" or "FrequencyMaskProbability" or "FrequencyMaskLength"
        or "Temperature" or "Negatives";

    private static bool IsCosyFlowOption(string option) => option is
        "TrainingCfgRate" or "FlowLoss" or "FlowAttentionDropoutRate";

    internal static int TrainingSteps<T>(INeuralNetworkModel<T> network, string option, int steps)
        => network is AiDotNet.SpeechRecognition.Foundation.SPIRAL<T> && IsSpiralPretrainingOption(option)
            ? Math.Max(3, steps) : steps;

    internal static void PrepareOptionTraining<T>(INeuralNetworkModel<T> network, string option)
    {
        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> cosy && IsCosyFlowOption(option))
            cosy.CurrentStage = AiDotNet.TextToSpeech.CodecBased.CosyVoiceTrainingStage.Flow;
        if (option != "CurrentStage" || network is not AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> natural
            || natural.GetOptions() is not AiDotNet.TextToSpeech.CodecBased.NaturalSpeechOptions options
            || options.CurrentStage != AiDotNet.TextToSpeech.CodecBased.NaturalSpeechStage.Joint) return;
        // Joint training has a documented warm-up prerequisite: cluster posterior frames into the bank.
        int samples = checked(Math.Max(options.MemorySize, 32) * options.HopSize);
        var recording = new Tensor<T>(new[] { samples });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < samples; i++)
            recording[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180 * i / options.SampleRate)
                + 0.2 * Math.Sin(2 * Math.PI * 470 * i / options.SampleRate));
        natural.InitializeMemoryBank(new[] { recording });
    }

    internal static bool TryTrainOption<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option)
    {
        if (network is not AiDotNet.SpeechRecognition.Foundation.SPIRAL<T> spiral || !IsSpiralPretrainingOption(option))
            return false;
        spiral.Pretrain(input);
        return true;
    }

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
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoiceFamilyOptions cosy && option == "TrainingCfgRate")
            cosy.TrainingCfgRate = 0.5;
        if (options is AiDotNet.TextToSpeech.Classic.ForwardTacotronOptions forward)
        {
            if (option is "SecondStageLearningRate" or "SecondStageStartStep")
                forward.SecondStageStartStep = 1;
            if (option == "GradientClipNorm") forward.GradientClipNorm = 1e-8;
            // Synthetic supervision spans 100–200 Hz; cross its voiced/unvoiced boundary.
            if (option == "PitchMinHz") forward.PitchMinHz = 125;
        }
        string? dependency = option switch
        {
            "FrequencyMaskWidth" => "FrequencyMasks",
            "TimeMaskRatio" => "TimeMasks",
            "MaxAudioLengthSeconds" => "MaxAudioLengthSeconds",
            _ => null,
        };
        if (dependency is not null)
        {
            var property = options.GetType().GetProperty(dependency);
            if (property?.PropertyType == typeof(int) && property.CanWrite) property.SetValue(options, 1);
        }
        if (options is AiDotNet.SpeechRecognition.Foundation.SPIRALOptions spiral && IsSpiralPretrainingOption(option))
        {
            spiral.PretrainingSteps = 4; // The third student step observes the teacher's changing EMA schedule.
            spiral.TeacherMomentumStart = 0.5;
            spiral.MaxTeacherPaddingUnits = 1; // The tiny fixture caps larger values to the same padding limit.
        }
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
