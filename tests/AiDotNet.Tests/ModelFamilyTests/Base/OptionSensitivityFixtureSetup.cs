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
        => UsesFittedAssets(network) || network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T>
            ? network.GetParameters().Length : 0;

    internal static string ControlGroup<T>(INeuralNetworkModel<T> network, string option)
        => network is AiDotNet.Audio.Speaker.CAMPlusPlus<T> && option == "FrameShiftMilliseconds" ? "cam-audio-frontend"
            : network is AiDotNet.VisionLanguage.Grounding.GLaMM<T> && option == "RegionGroupNormGroups" ? "glamm-region-conversation"
            : network is AiDotNet.VisionLanguage.Grounding.GLaMM<T> && option == "TokenizerPieces" ? "glamm-unloaded-tokenizer"
            : network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> && option == "LatentFusionProbability" ? "index-latent-fusion"
            : network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> && IsIndexFlowOption(option) ? "index-semantic-to-mel"
            : network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> && option == "SpeakerAdversaryWeight" ? "index-emotion-adversary"
            : network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> && IsIndexEmotionArchitecture(option) ? "index-text-emotion"
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> && option == "MemoryKMeansIterations" ? "natural-memory-clustering"
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> && IsNaturalJointOption(option) ? "natural-joint"
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> && option == "MaskRatio" ? "natural-phoneme-pretraining"
            : network is AiDotNet.TextToSpeech.Vocoders.DiffusionVocoderBase<T> && option == "CropFrames" ? "diffusion-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.DiffusionVocoderBase<T> && option == "InferenceNoiseSchedule" ? "diffusion-fast-sampling"
            : network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option == "AdversarialWeight" ? "multi-band-adversarial"
            : network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option == "SegmentSize" ? "multi-band-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option is "MelMean" or "MelScale" ? "vocoder-normalization"
            : IsVocoderFrontendOption(option) && network is AiDotNet.TextToSpeech.Vocoders.SegmentVocoderBase<T>
                or AiDotNet.TextToSpeech.Vocoders.GanVocoderBase<T> ? "vocoder-frontend"
            : network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> && option == "MaximumAudioFrames" ? "voicebox-audio-chunk"
            : network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> && option == "TopK" ? "cosy-top-k"
            : network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> && IsCosyFlowOption(option) ? "cosy-flow"
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

    internal static long Observe<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option,
        Func<INeuralNetworkModel<T>, Tensor<T>, long> observe)
    {
        if (network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> fish && option == "ConstrainToSemanticTokens")
            return ObserveFishFrames(fish, input);
        long fingerprint = observe(network, ObservationInput(network, input, option));
        if (network is not AiDotNet.VisionLanguage.Grounding.GLaMM<T> glamm || option != "RegionGroupNormGroups")
            return fingerprint;
        // Predict encodes the whole image; only a conversation naming a box runs the region encoder.
        var sample = new AiDotNet.VisionLanguage.Grounding.GlammSample<T>(input,
            new[] { new AiDotNet.VisionLanguage.Grounding.GlammTurn("Describe <bbox>.", "A region.") },
            new[] { new[] { 0.1, 0.1, 0.8, 0.8 } });
        double loss = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble(glamm.EvaluateConversationLoss(sample));
        return OptionSensitivityHarness.Combine(fingerprint, OptionSensitivityHarness.Fingerprint(new[] { loss }));
    }

    private static long ObserveFishFrames<T>(AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> model, Tensor<T> input)
    {
        model.SetTrainingMode(false);
        // Observe the actual generated frame rows before codec decoding: an untrained unconstrained
        // model can emit non-speech frames that a waveform decoder correctly refuses to interpret.
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
        var type = typeof(AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T>);
        var format = type.GetProperty("Format", flags)!.GetValue(model)!;
        var transformer = type.GetProperty("Transformer", flags)!.GetValue(model)!;
        var options = (AiDotNet.TextToSpeech.CodecBased.FishSpeechFamilyOptions)model.GetOptions();
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var ids = Enumerable.Range(0, input.Length).Select(i => (int)ops.ToDouble(input[i])).ToArray();
        var prompt = (int[,])format.GetType().GetMethod("Segment", new[] { typeof(int[]), typeof(int[,]) })!
            .Invoke(format, new object?[] { ids, null })!;
        var generator = type.Assembly.GetType("AiDotNet.TextToSpeech.CodecBased.FishSpeechGenerator`1")!.MakeGenericType(typeof(T));
        var frames = (int[,])generator.GetMethod("Generate", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!
            .Invoke(null, new object[] { AiDotNet.Tensors.Engines.AiDotNetEngine.Current, transformer, format, prompt,
                4, options.Temperature, options.TopP, options.RepetitionPenalty,
                new Random(options.SamplingSeed), options.ConstrainToSemanticTokens })!;
        return OptionSensitivityHarness.Combine(
            OptionSensitivityHarness.Fingerprint(frames.Cast<int>().Select(value => (double)value)),
            OptionSensitivityHarness.Fingerprint(model.GetParameters().Select(value => ops.ToDouble(value))));
    }

    internal static Tensor<T> ObservationInput<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option)
    {
        if (network is AiDotNet.Audio.Speaker.CAMPlusPlus<T> cam && option == "FrameShiftMilliseconds")
        {
            var settings = (AiDotNet.Audio.Speaker.CAMPlusPlusOptions)cam.GetOptions();
            var camWaveform = new Tensor<T>(new[] { settings.SampleRate });
            var frontendOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < camWaveform.Length; i++)
                camWaveform[i] = frontendOps.FromDouble(0.4 * Math.Sin(i * 0.13) + 0.2 * Math.Cos(i * 0.37));
            return (Tensor<T>)typeof(AiDotNet.Audio.Speaker.CAMPlusPlus<T>).GetMethod("PreprocessAudio",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.Invoke(cam, new object[] { camWaveform })!;
        }
        if (network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> index && IsIndexEmotionArchitecture(option))
            index.DetectEmotion("I feel happy."); // The text-to-emotion model is built lazily by its public entry point.
        if (IsVocoderFrontendOption(option) && network is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> native
            && native.GetOptions() is AiDotNet.TextToSpeech.Vocoders.VocoderOptions vocoder
            && network is AiDotNet.TextToSpeech.Vocoders.SegmentVocoderBase<T> or AiDotNet.TextToSpeech.Vocoders.GanVocoderBase<T>)
        {
            var recording = VocoderRecording<T>(vocoder);
            return network is AiDotNet.TextToSpeech.Vocoders.SegmentVocoderBase<T> segment
                ? segment.ComputeMel(recording) : ((AiDotNet.TextToSpeech.Vocoders.GanVocoderBase<T>)network).ComputeMel(recording);
        }
        if (option == "MaximumAudioFrames" && network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> && input.Length > 0)
        {
            // Keep synthesis below the smallest valid budget; training retains the longer utterance
            // so its real phone-aligned chunking path can respond to that budget.
            var phone = new Tensor<T>(new[] { 1 });
            phone[0] = input[0];
            return phone;
        }
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

    private static bool IsNaturalJointOption(string option) => option is
        "MemoryKMeansIterations" or "MemoryHeads" or "DurationLossAfterWarmup" or "ForwardKlWeight";

    private static bool IsIndexEmotionArchitecture(string option) => option is
        "EmotionHidden" or "EmotionLayers" or "EmotionHeads" or "EmotionKeyValueHeads" or "EmotionHeadDim"
        or "EmotionIntermediate" or "EmotionVocabulary" or "EmotionLoraRank";

    private static bool IsIndexFlowOption(string option) => option is
        "LatentFusionProbability" or "ClassDropout" or "WavenetDropout";

    private static bool IsVocoderFrontendOption(string option) => option is
        "MelMinFrequency" or "MelMaxFrequency" or "WindowSize" or "MelMean" or "MelScale" or "EnergyMin";

    private static Tensor<T> VocoderRecording<T>(AiDotNet.TextToSpeech.Vocoders.VocoderOptions options)
    {
        var recording = new Tensor<T>(new[] { Math.Max(2 * options.FftSize, 8 * options.HopSize) });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < recording.Length; i++)
            recording[i] = ops.FromDouble(0.4 * Math.Sin(i * 0.13) + 0.2 * Math.Cos(i * 0.37));
        return recording;
    }

    internal static int TrainingSteps<T>(INeuralNetworkModel<T> network, string option, int steps)
        => network is AiDotNet.SpeechRecognition.Foundation.SPIRAL<T> && IsSpiralPretrainingOption(option)
            ? Math.Max(3, steps) : steps;

    internal static void PrepareOptionTraining<T>(INeuralNetworkModel<T> network, string option)
    {
        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> cosy && IsCosyFlowOption(option))
            cosy.CurrentStage = AiDotNet.TextToSpeech.CodecBased.CosyVoiceTrainingStage.Flow;
        if ((option != "CurrentStage" && !IsNaturalJointOption(option))
            || network is not AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> natural
            || natural.GetOptions() is not AiDotNet.TextToSpeech.CodecBased.NaturalSpeechOptions options
            || options.CurrentStage != AiDotNet.TextToSpeech.CodecBased.NaturalSpeechStage.Joint) return;
        // Joint training has a documented warm-up prerequisite: cluster posterior frames into the bank.
        int frames = option == "MemoryKMeansIterations" ? Math.Max(8 * options.MemorySize, 128) : Math.Max(options.MemorySize, 32);
        int samples = checked(frames * options.HopSize);
        var recording = new Tensor<T>(new[] { samples });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < samples; i++)
            recording[i] = ops.FromDouble(0.4 * Math.Sin(2 * Math.PI * 180 * i / options.SampleRate)
                + 0.2 * Math.Sin(2 * Math.PI * 470 * i / options.SampleRate));
        natural.InitializeMemoryBank(new[] { recording });
    }

    internal static bool TryTrainOption<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option)
    {
        if ((network is AiDotNet.TextToSpeech.Vocoders.DiffusionVocoderBase<T> && option == "CropFrames"
                || network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option == "SegmentSize")
            && network is AiDotNet.TextToSpeech.TtsModelBase<T> vocoder
            && vocoder.GetOptions() is AiDotNet.TextToSpeech.Vocoders.VocoderOptions options)
        {
            vocoder.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            {
                Tokens = input,
                Audio = VocoderRecording<T>(options),
            });
            return true;
        }
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
        if (options is AiDotNet.VisionLanguage.Grounding.GLaMMOptions glamm && option == "TokenizerPieces")
        {
            glamm.TokenizerModel = null;
            glamm.TokenizerPath = null; // Vocabulary sizing applies before a tokenizer asset is supplied.
        }
        if (options is AiDotNet.TextToSpeech.Latest.IndexTTS2Options stagedIndex)
        {
            if (IsIndexFlowOption(option))
            {
                stagedIndex.CurrentStage = AiDotNet.TextToSpeech.Latest.IndexTts2Stage.SemanticToMel;
                stagedIndex.ClassDropout = option == "LatentFusionProbability" ? 0 : 0.5;
            }
            if (option == "SpeakerAdversaryWeight")
            {
                stagedIndex.CurrentStage = AiDotNet.TextToSpeech.Latest.IndexTts2Stage.Emotion;
                stagedIndex.SpeakerCount = 2; // One class has identically zero cross-entropy.
            }
        }
        if (options is AiDotNet.TextToSpeech.Latest.IndexTTS2Options index && IsIndexEmotionArchitecture(option))
        {
            index.EmotionHidden = 16;
            index.EmotionLayers = 1;
            index.EmotionHeads = 2;
            index.EmotionKeyValueHeads = 1;
            index.EmotionHeadDim = 8;
            index.EmotionIntermediate = 32;
            index.EmotionLoraRank = 2;
            index.EmotionMaxTokens = 1;
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.NaturalSpeechOptions natural)
        {
            if (IsNaturalJointOption(option)) natural.CurrentStage = AiDotNet.TextToSpeech.CodecBased.NaturalSpeechStage.Joint;
            if (option == "MaskRatio")
            {
                natural.CurrentStage = AiDotNet.TextToSpeech.CodecBased.NaturalSpeechStage.PhonemePretraining;
                natural.MaskRatio = 0.5;
            }
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.DiffWaveOptions diffusion)
        {
            if (option == "CropFrames") diffusion.CropFrames = 2;
            if (option == "InferenceNoiseSchedule") diffusion.UseFastSampling = true;
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGANOptions gan)
        {
            if (option == "AdversarialWeight") gan.PretrainingSteps = 0;
            if (option == "SegmentSize") gan.SegmentSize = checked(2 * gan.HopSize);
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGANOptions multiBand && option is "MelMean" or "MelScale")
        {
            multiBand.MelMean = new double[multiBand.MelChannels];
            multiBand.MelScale = Enumerable.Repeat(1.0, multiBand.MelChannels).ToArray();
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.VoiceboxOptions voicebox && option == "MaximumAudioFrames")
            voicebox.MaximumPhoneDuration = 1;
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoiceFamilyOptions cosy && IsCosyFlowOption(option))
            cosy.TrainingCfgRate = 0.5;
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoiceFamilyOptions sampling && option == "TopK")
        {
            sampling.TopK = 2;
            sampling.TopP = 1;
        }
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
        if (perEpoch is not null && perEpoch.CanWrite
            && (perEpoch.PropertyType == typeof(int) || Nullable.GetUnderlyingType(perEpoch.PropertyType) == typeof(int)))
            perEpoch.SetValue(options, 1);
    }

    internal static void UseConfiguredOptimizer<T>(INeuralNetworkModel<T> network)
    {
        if (network is not AiDotNet.TextToSpeech.TtsModelBase<T>
            and not AiDotNet.Audio.Codecs.NeuralAudioCodecBase<T>) return;
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
