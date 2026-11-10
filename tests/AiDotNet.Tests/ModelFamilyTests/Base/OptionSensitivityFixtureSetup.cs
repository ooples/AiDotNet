using AiDotNet.Interfaces;
using AiDotNet.Tests.TextToSpeech;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>Replays the fitted assets installed by the generated fixture after configuration-only reconstruction.</summary>
internal static class OptionSensitivityFixtureSetup
{
    private static bool IsGooglePretrainingOption(string option) => option is
        "MaskNoiseStd" or "MaskLength" or "MaskProbability" or "CodeDim" or "StackedFrames";

    private static bool IsMaskGctDurationOption(string option) => option is
        "DurationMode" or "DurationSteps" or "DurationGuidance" or "DurationHeads";

    private static bool IsMaskGctSamplingOption(string option) => option is
        "TextToSemanticSteps" or "TextToSemanticTemperature" or "SemanticToAcousticTemperature" or "SamplingSeed";

    private static bool IsFishScheduleOption(string option) => option is
        "TrainingSteps" or "CosineDecay" or "FinalLearningRateRatio" or "WarmupSteps";

    private sealed class RepetitionBoundaryRandom : Random
    {
        private int _draws;
        public override double NextDouble() => _draws++ == 0 ? 0.1 : 0.99;
    }

    private sealed class NucleusBoundaryRandom : Random
    {
        public override double NextDouble() => 0.99;
    }

    internal static bool UsesFittedAssets<T>(INeuralNetworkModel<T> network)
        => network is AiDotNet.TextToSpeech.Latest.KaniTTSBase<T>
            or AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T>
            or AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T>
            or AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T>
            or AiDotNet.TextToSpeech.MultiModal.SpiritLM<T>
            or AiDotNet.TextToSpeech.Latest.MegaTTS3<T>
            or AiDotNet.TextToSpeech.CodecBased.SparkTTS<T>;

    internal static long ParameterCountBeforeFitting<T>(INeuralNetworkModel<T> network)
        => network is AiDotNet.TextToSpeech.EndToEnd.Kokoro<T> kokoro
            ? kokoro.ParameterCount
            : network is AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T> metaVoice ? MetaVoiceOptionFixture.ParameterCount(metaVoice)
            : network.GetParameters().Length;

    internal static string ControlGroup<T>(INeuralNetworkModel<T> network, string option)
        => AiDotNet.Tests.VisionLanguage.NormalizationOptionFixture.Handles(network, option) ? "image-normalization"
            : network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> && option == "Watermark" ? "chatterbox-published-perth"
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> && option == "AddBlank" ? "natural-phoneme-frontend"
            : network is AiDotNet.TextToSpeech.Vocoders.WaveGlow<T> && option == "SegmentSamples" ? "waveglow-recording-crop"
            : network is AiDotNet.TextToSpeech.CodecBased.SPEARTTS<T> && SpearOptionFixture.Handles(option) ? "spear-" + option
            : network is AiDotNet.TextToSpeech.Latest.OuteTTS<T> && OuteOptionFixture.Handles(option) ? "oute-" + option
            : network is AiDotNet.TextToSpeech.FlowDiffusion.F5TTS<T> or AiDotNet.TextToSpeech.FlowDiffusion.E2TTS<T> ? "infilling-" + option
            : network is AiDotNet.Video.Enhancement.Upscale4KAgent<T> && option == "RollbackThreshold" ? "4k-reflection"
            : network is AiDotNet.TextToSpeech.CodecBased.CosyVoice3<T> && option == "StaticChunkSize" ? "cosy-streaming-flow"
            : network is AiDotNet.TextToSpeech.CodecBased.UniAudio<T> ? "uniaudio-" + option
        : option == nameof(AiDotNet.Models.Options.ModelOptions.Seed) ? "model-options-seed"
            : network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> && option == "MaximumPhoneDuration" ? "voicebox-duration-cap"
            : network is AiDotNet.TextToSpeech.MultiModal.MinMo<T> && MinMoOptionFixture.Handles(option) ? "minmo-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.CSM<T> && CsmOptionFixture.Handles(option) ? "csm-" + option
            : network is AiDotNet.TextToSpeech.EndToEnd.Kokoro<T> ? "kokoro-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.TortoiseTTS<T> && TortoiseOptionFixture.Handles(option) ? "tortoise-" + option
            : network is AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T> && MetaVoiceOptionFixture.Handles(option) ? "metavoice-" + option


            : network is AiDotNet.TextToSpeech.Classic.DeepVoice3<T> && option == "GradientClipValue" ? "deepvoice-value-clip"

            : network is AiDotNet.TextToSpeech.FlowDiffusion.DiTToTTS<T> && option is "TextPadId" or "NumHeads" ? "ditto-" + option

            : network is AiDotNet.TextToSpeech.CodecBased.Bark<T> && BarkOptionFixture.Handles(option) ? "bark-" + option

            : network is AiDotNet.TextToSpeech.CodecBased.Vevo<T> && option == "MaxLength" ? "vevo-autoregressive-budget"

            : network is AiDotNet.Finance.Forecasting.StateSpace.Hippo<T> && option == "TimescaleMax" ? "hippo-timescale-clamp"
            : network is AiDotNet.SpeechRecognition.CTCVariants.CTCSegmentation<T> && option == "ConfidenceWindow" ? "ctc-segmentation-confidence"
            : network is AiDotNet.TextToSpeech.Latest.MegaTTS3<T> ? "mega-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.SpiritLM<T> ? "spirit-" + option
            : network is AiDotNet.Audio.MusicGen.MusicGenModel<T> ? "musicgen-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.GLM4Voice<T> ? "glmvoice-" + option
            : network is AiDotNet.TextToSpeech.VoiceCloning.XTTSv2<T> && option is "DvaeFftSize" or "DvaeHopSize" or "DvaeWindowSize" ? "xtts-codec"
            : network is AiDotNet.TextToSpeech.VoiceCloning.XTTSv2<T> && XttsOptionFixture.IsSamplerOption(option) ? "xtts-sampler"
            : AiDotNet.Tests.IntegrationTests.SyntheticData.SyntheticGanOptionFixture.Handles(network) ? "synthetic-" + option
            : network is AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> ? "audiopalm-asr-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T> ? "audiopalm-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.SoundStorm<T> && (SoundStormOptionFixture.IsTrainingOption(option) || SoundStormOptionFixture.IsInferenceOption(option)) ? "soundstorm-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.Pheme<T> && (PhemeOptionFixture.IsTrainingOption(option) || PhemeOptionFixture.IsInferenceOption(option)) ? "pheme-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.MARS5TTS<T> ? "mars5-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.SeedTTS<T> ? "seedtts-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech3<T> ? "naturalspeech3-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.VoiceCraft<T> ? "voicecraft-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.VALLE2<T> ? "valle2-" + option
            : network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTSZS<T> ? "stylezs-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.WhisperSpeech<T> ? "whisperspeech-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.SpeechT5<T> ? "speecht5-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.SparkTTS<T> ? "spark-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.FireRedTTS<T> ? "firered-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.VALLEX<T> ? "vallex-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.GPTSoVITS<T> ? "gptsovits-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.Zonos<T> ? "zonos-" + option



            : network is AiDotNet.TextToSpeech.FlowDiffusion.E3TTS<T> ? option switch
            {
                "DefaultSpeakerId" => "e3-speaker-selection",
                "TrainingBetaEnd" => "e3-training-schedule",
                "PaddingLossWeight" => "e3-padded-training",
                "SilenceBlockSize" or "SilenceMagnitudeThreshold" => "e3-silence-trimming",
                _ => string.Empty,
            }
            : network is AiDotNet.SpeechRecognition.ProprietaryAPI.GoogleSpeechV2<T> && IsGooglePretrainingOption(option) ? "google-best-rq-masking"
            : network is AiDotNet.SpeechRecognition.Foundation.BESTRQ<T> && IsGooglePretrainingOption(option) ? "best-rq-masking"
            : network is AiDotNet.Audio.Generation.AudioLM<T> && option is "SemanticCropSeconds" or "CoarseCropSeconds" or "FineCropSeconds" ? "audiolm-stage-crop"
            : network is AiDotNet.Audio.Generation.AudioLM<T> && option is "SemanticTemperature" or "CoarseTemperature" or "FineTemperature" or "Seed" ? "audiolm-code-sampling"
            : network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> && IsFishScheduleOption(option) ? "fish-training-schedule"
            : network is AiDotNet.Audio.Effects.DAC<T> && option == "QuantizerDropout" ? "dac-quantizer-dropout"
            : network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> && IsMaskGctSamplingOption(option) ? "maskgct-token-sampling"
            : network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> && option == "DurationGuidance" ? "maskgct-prompted-duration"

            : network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> && IsMaskGctDurationOption(option) ? "maskgct-duration-predictor"
            : network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> && option == "SemanticToAcousticHeads" ? "maskgct-acoustic-heads"
            : network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> && option == "GuidanceRescale" ? "maskgct-guided-logits"
            : network is AiDotNet.PointCloud.Models.DGCNN<T> && option is "DropoutRate" or "UseDropout" ? "dgcnn-dropout"
            : network is AiDotNet.Audio.Speaker.CAMPlusPlus<T> && option is "FrameShiftMilliseconds" or "FrameLengthMilliseconds" ? "cam-audio-frontend"
            : network is AiDotNet.VisionLanguage.Grounding.GLaMM<T> && option == "RegionGroupNormGroups" ? "glamm-region-conversation"
            : network is AiDotNet.VisionLanguage.Grounding.GLaMM<T> && option == "TokenizerPieces" ? "glamm-unloaded-tokenizer"
            : network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> && option is "EmotionGroupSizes" or "EmotionMode" ? "index-emotion-table"
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
            : network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option is "SegmentSize" or "SamplingSeed" ? "multi-band-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.BigVGAN<T> && option == "SegmentSize" ? "bigvgan-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option is "MelMean" or "MelScale" ? "vocoder-normalization"
            : IsVocoderFrontendOption(option) && network is AiDotNet.TextToSpeech.Vocoders.SegmentVocoderBase<T>
                or AiDotNet.TextToSpeech.Vocoders.GanVocoderBase<T> ? "vocoder-frontend"
            : network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> && option is "UseNativeEnglishFallback" or "SpectrogramMaskMinimum" ? "voicebox-" + option

            : network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> && option == "MaximumAudioFrames" ? "voicebox-audio-chunk"
            : network is AiDotNet.TextToSpeech.EndToEnd.VITS2<T>
                && option is "SpeakerEmbeddingDim" or "LanguageEmbeddingDim" or "SpeakerConditionedEncoderBlock"
                    or "NumSpeakers" or "AlignmentNoiseDecay" ? "vits2-conditioning-" + option
            : network is AiDotNet.TextToSpeech.EndToEnd.YourTTS<T>
                && option is "LanguageEmbeddingDim" or "SpeakerConsistencyLossWeight" ? "yourtts-conditioning-" + option
            : network is AiDotNet.TextToSpeech.EndToEnd.VITS<T> or AiDotNet.TextToSpeech.EndToEnd.Piper<T>
                && option is "SpeakerEmbeddingDim" or "LanguageEmbeddingDim" ? "vits-conditioning-" + option
            : network is AiDotNet.TextToSpeech.FlowDiffusion.CoMoSpeech<T>
                && option is "DistillationSteps" or "EmaDecay" ? "comospeech-distillation"
            : network is AiDotNet.TextToSpeech.Classic.GradTTS<T> && option == "NumDecoderLayers" ? "gradtts-native-depth"

            : network is AiDotNet.TextToSpeech.Classic.AlignTTS<T> && option == "LearningRate" ? "aligntts-fixed-rate-training"
            : network is AiDotNet.TextToSpeech.CodecBased.OrpheusTTS<T>
                && option is "MaxGradNorm" or "RemoveDuplicateFrames" ? "orpheus-training-" + option
            : network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T>
                ? option == "DecoderType" ? "styletts2-decoder-switch"
                : option == "IstftFftSize" ? "styletts2-istft"
                : option is "MonotonicLossWeight" or "S2sLossWeight" ? "styletts2-active-aligner"
                : StyleTts2Fixture.IsSlmAdversarialOption(option) ? "styletts2-slm-" + option
                : IsStyleTts2JointOption(option) ? "styletts2-joint" : "styletts2-pretraining"
            : network is AiDotNet.VisionLanguage.Grounding.GroundedSAM2<T> && option is "MaskThreshold" or "SamDecoderHeads" ? "grounded-sam2-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.CosyVoice<T> && option == "SpeechTokenizerHeads" ? "cosy-native-s3-heads"
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
        if (network is AiDotNet.TextToSpeech.EndToEnd.Kokoro<T> kokoro)
            return KokoroOptionFixture.Observe(kokoro, option);
        var task = ObserveTask(network, input, option, observe);
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        // Vector.Select materializes another full vector (and a double copy for float models).
        // Stream conversions so a paper-scale model needs only the existing parameter snapshot.
        var parameters = network.GetParameters();
        return OptionSensitivityHarness.Combine(task,
            OptionSensitivityHarness.Fingerprint(Enumerable.Range(0, parameters.Length).Select(i => ops.ToDouble(parameters[i]))));
    }

    private static long ObserveTask<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option,
        Func<INeuralNetworkModel<T>, Tensor<T>, long> observe)
    {
        if (network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> marked && option == "Watermark")
            // Run the real final synthesis stage with the hash-verified released checkpoint configured below.
            return OptionSensitivityHarness.Fingerprint(marked.Watermarked(ChatterboxFixture.Recording<T>())
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.CodecBased.SPEARTTS<T> spear && SpearOptionFixture.Handles(option))
            return OptionSensitivityHarness.Fingerprint(SpearOptionFixture.Observe(spear, option));
        if (network is AiDotNet.TextToSpeech.Latest.OuteTTS<T> oute && OuteOptionFixture.Handles(option))
            return OptionSensitivityHarness.Fingerprint(OuteOptionFixture.Observe(oute, option));
        if (network is AiDotNet.Video.Enhancement.Upscale4KAgent<T> agent && option == "RollbackThreshold")
            return AiDotNet.Tests.Video.Upscale4KOptionFixture.Observe(agent);
        if (network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> naturalFrontend && option == "AddBlank")
            // This is the actual text preprocessing path; Predict(Tensor) already receives phoneme IDs.
            // A punctuation token is phonemized without loading the optional eSpeak library.
            return OptionSensitivityHarness.Fingerprint(naturalFrontend.TextToPhonemes("!").Select(id => (double)id));

        if (network is AiDotNet.TextToSpeech.CodecBased.UniAudio<T> uniAudio && option == "UseReleasedTextFrontend")
            return UniAudioOptionFixture.ObserveFrontend(uniAudio);
        if (AiDotNet.Tests.VisionLanguage.NormalizationOptionFixture.Handles(network, option))
            return AiDotNet.Tests.VisionLanguage.NormalizationOptionFixture.Observe(network);
        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoice3<T> streamingCosy && option == "StaticChunkSize")
            return CosyVoiceStreamingOptionFixture.Observe(streamingCosy);
        if (network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> durationVoicebox && option == "MaximumPhoneDuration")
            return VoiceboxDurationOptionFixture.Observe(durationVoicebox);

        if (network is AiDotNet.TextToSpeech.MultiModal.MinMo<T> minMo && MinMoOptionFixture.Handles(option))
            return MinMoOptionFixture.Observe(minMo, option);

        if (network is AiDotNet.TextToSpeech.CodecBased.CSM<T> csm && CsmOptionFixture.Handles(option))
            return CsmOptionFixture.Observe(csm, input, option);

        if (network is AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T> metaVoice && MetaVoiceOptionFixture.Handles(option))
            return MetaVoiceOptionFixture.Observe(metaVoice, input, option);

        if (network is AiDotNet.TextToSpeech.CodecBased.TortoiseTTS<T> tortoise && TortoiseOptionFixture.Handles(option)
            && TortoiseOptionFixture.Observe(tortoise, option) is long tortoiseObservation)
            return tortoiseObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.Voicebox<T> voicebox)
        {
            if (option == "UseNativeEnglishFallback")
            {
                try
                {
                    return OptionSensitivityHarness.Fingerprint(string.Join("|", voicebox.Phonemize("test")).Select(value => (double)value));
                }
                catch (InvalidOperationException error) when (error.Message.StartsWith("Voicebox raw text requires an MFA-compatible", StringComparison.Ordinal))
                {
                    return OptionSensitivityHarness.Fingerprint(Array.Empty<double>());
                }
            }
            if (option == "SpectrogramMaskMinimum")
            {
                var settings = (AiDotNet.TextToSpeech.CodecBased.VoiceboxOptions)voicebox.GetOptions();
                var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
                var mel = new Tensor<T>(new[] { 16, settings.MelChannels });
                for (int i = 0; i < mel.Length; i++) mel[i] = numeric.FromDouble(-5 + 0.2 * Math.Sin(i * 0.13));
                var tokens = new Tensor<T>(new[] { 4 });
                for (int i = 0; i < tokens.Length; i++) tokens[i] = numeric.FromDouble(i + 1);
                var objective = voicebox.EvaluateTrainingObjective(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
                {
                    Tokens = tokens, Mel = mel, Durations = new[] { 4, 4, 4, 4 },
                });
                return OptionSensitivityHarness.Fingerprint(new[] { numeric.ToDouble(objective) });
            }
        }

        if (network is AiDotNet.VisionLanguage.Grounding.GroundedSAM2<T> groundedSam && option is "MaskThreshold" or "SamDecoderHeads")
            return AiDotNet.Tests.VisionLanguage.GroundedSam2OptionFixture.Observe(groundedSam, input);

        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoice<T> cosyTokenizer && option == "SpeechTokenizerHeads")
            return CosyVoiceTokenizerOptionFixture.Observe(cosyTokenizer);

        if (network is AiDotNet.TextToSpeech.FlowDiffusion.DiTToTTS<T> ditto && option is "TextPadId" or "NumHeads")
            return option == "NumHeads" ? DiTToOptionFixture.ObserveHeads(ditto) : DiTToOptionFixture.ObserveTextPadding(ditto);

        if (network is AiDotNet.TextToSpeech.CodecBased.Bark<T> bark && BarkOptionFixture.Handles(option))
            return BarkOptionFixture.Observe(bark, option);

        if (network is AiDotNet.TextToSpeech.CodecBased.Vevo<T> vevo && option == "MaxLength")
            return VevoOptionFixture.Observe(vevo);

        if (network is AiDotNet.SpeechRecognition.CTCVariants.CTCSegmentation<T> segmentation && option == "ConfidenceWindow")
            return AiDotNet.Tests.Audio.CtcSegmentationOptionFixture.Observe(segmentation);
        if (AiDotNet.Tests.IntegrationTests.SyntheticData.SyntheticGanOptionFixture.Handles(network))
            return AiDotNet.Tests.IntegrationTests.SyntheticData.SyntheticGanOptionFixture.Observe(network);

        if (network is AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> audioPalm
            && AudioPalmAsrOptionFixture.Observe(audioPalm, option) is { } audioPalmObservation)
            return audioPalmObservation;

        if (network is AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T> synthesisPalm
            && AudioPalmAsrOptionFixture.ObserveSynthesis(synthesisPalm, option) is { } promptObservation)
            return promptObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.SoundStorm<T> soundStorm
            && SoundStormOptionFixture.Observe(soundStorm, option) is { } soundStormObservation)
            return soundStormObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.Pheme<T> pheme
            && PhemeOptionFixture.Observe(pheme, option) is { } phemeObservation)
            return phemeObservation;

        if (network is AiDotNet.TextToSpeech.VoiceCloning.XTTSv2<T> xtts
            && option is "DvaeFftSize" or "DvaeHopSize" or "DvaeWindowSize")
            return XttsOptionFixture.ObserveCodec(xtts);

        if (network is AiDotNet.TextToSpeech.VoiceCloning.XTTSv2<T> samplerXtts && XttsOptionFixture.IsSamplerOption(option))
            return XttsOptionFixture.ObserveSampler(samplerXtts);

        if (network is AiDotNet.TextToSpeech.CodecBased.MARS5TTS<T> mars5
            && Mars5OptionFixture.Observe(mars5, option) is { } marsObservation)
            return marsObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.SeedTTS<T> seedTts
            && SeedTtsOptionFixture.Observe(seedTts, option) is { } seedObservation)
            return seedObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.VoiceCraft<T> voiceCraft
            && VoiceCraftOptionFixture.Observe(voiceCraft, option) is { } voiceObservation)
            return voiceObservation;

        if (network is AiDotNet.TextToSpeech.MultiModal.GLM4Voice<T> glmVoice
            && GlmVoiceOptionFixture.Observe(glmVoice, option) is { } glmObservation)
            return glmObservation;

        if (network is AiDotNet.Audio.MusicGen.MusicGenModel<T> musicGen
            && AiDotNet.Tests.Audio.MusicGenOptionFixture.Observe(musicGen, input, option) is { } musicGenObservation)
            return musicGenObservation;

        if (network is AiDotNet.TextToSpeech.Latest.MegaTTS3<T> mega
            && MegaTts3Fixture.Observe(mega, option) is { } megaObservation)
            return megaObservation;

        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTSZS<T> styleZs
            && StyleTtsZsFixture.Observe(styleZs, option) is { } styleZsObservation)
            return styleZsObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.VALLE2<T> valle2
            && Valle2OptionFixture.Observe(valle2, option) is { } valle2Observation)
            return valle2Observation;

        if (network is AiDotNet.TextToSpeech.MultiModal.WhisperSpeech<T> whisperSpeech
            && WhisperSpeechFixture.Observe(whisperSpeech, option) is { } whisperObservation)
            return whisperObservation;

        if (network is AiDotNet.TextToSpeech.MultiModal.SpeechT5<T> speechT5
            && SpeechT5Fixture.Observe(speechT5, option) is { } speechObservation)
            return speechObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.OrpheusTTS<T> orpheus && option == "RemoveDuplicateFrames")
        {
            // Pre-tokenized SNAC frames are a supported training input. Include a repeated coarse
            // code so the release's deduplication path changes the actual language-model sequence.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var codes = new Tensor<T>(new[] { 3, 7 });
            for (int frame = 0; frame < 3; frame++)
                for (int slot = 0; slot < 7; slot++) codes[frame, slot] = ops.FromDouble(slot + (frame < 2 ? 1 : 2));
            var ids = orpheus.TrainingIds(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            {
                Tokens = input, Text = "A repeated codec frame.", CodecTokens = codes,
            });
            return OptionSensitivityHarness.Fingerprint(ids.Select(id => (double)id));
        }

        // Unlike the joint prosody losses, these two objectives belong to stage-1 alignment.
        // Observe the real objective as well as weights: Adam can absorb a scalar loss rescaling.
        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T> alignerProbe
            && option is "MonotonicLossWeight" or "S2sLossWeight")
        {
            var settings = (AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2Options)alignerProbe.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var objective = alignerProbe.EvaluateTrainingObjective(StyleTts2Fixture.Sample<T>(settings.HopSize), adversarialTerms: true);
            return OptionSensitivityHarness.Combine(observe(network, input),
                OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(objective) }));
        }
        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T> styleProbe && IsStyleTts2JointOption(option))
        {
            var stage = styleProbe.CurrentStage;
            try
            {
                styleProbe.CurrentStage = AiDotNet.TextToSpeech.StyleEmotion.StyleTts2TrainingStage.Joint;
                var settings = (AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2Options)styleProbe.GetOptions();
                var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
                var objective = styleProbe.EvaluateTrainingObjective(StyleTts2Fixture.Sample<T>(settings.HopSize), adversarialTerms: true);
                return OptionSensitivityHarness.Combine(observe(network, input),
                    OptionSensitivityHarness.Fingerprint(new[] { ops.ToDouble(objective) }));
            }
            finally { styleProbe.CurrentStage = stage; }
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.Zonos<T> zonos
            && ZonosFixture.Observe(zonos, input, option) is { } zonosObservation)
            return zonosObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.GPTSoVITS<T> gptSoVits)
            return GptSoVitsFixture.Observe(gptSoVits, input, option);

        if (network is AiDotNet.TextToSpeech.CodecBased.VALLEX<T> vallEX
            && VallEXCloneFixture.Observe(vallEX, option, input) is { } vallEXObservation)
            return vallEXObservation;

        if (network is AiDotNet.TextToSpeech.CodecBased.FireRedTTS<T> fireRed
            && FireRedTtsFixture.Observe(fireRed, option) is { } fireRedObservation)
            return fireRedObservation;
        if (network is AiDotNet.TextToSpeech.CodecBased.SparkTTS<T> spark
            && SparkNativeFixture.Observe(spark, option) is { } sparkObservation)
            return sparkObservation;
        if (network is AiDotNet.TextToSpeech.MultiModal.SpiritLM<T> spirit
            && SpiritLmFixture.Observe(spirit, option) is { } spiritObservation)
            return spiritObservation;
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> guidanceMask && option == "GuidanceRescale")
        {
            guidanceMask.SetTrainingMode(false);
            var settings = (AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions)guidanceMask.GetOptions();
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var module = guidanceMask.GetType().GetProperty("TextToSemantic", flags)!.GetValue(guidanceMask)!;
            var decoding = guidanceMask.GetType().GetMethod("Decoding", flags)!.Invoke(guidanceMask,
                new object[] { settings.TextToSemanticTemperature })!;
            var logits = (Tensor<T>)module.GetType().GetMethod("GuidedLogits", flags)!.Invoke(module,
                new object[] { new[] { 1, 2, 3, 4 }, new[] { 0, 1 }, new[] { 1, 2, 3, 4, 5, 6 },
                    new[] { true, false, true, false, true, false }, 0.37, decoding })!;
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            return OptionSensitivityHarness.Fingerprint(logits.Select(value => ops.ToDouble(value)));
        }
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> acousticMask && option == "SemanticToAcousticHeads")
        {
            acousticMask.SetTrainingMode(false);
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var first = acousticMask.GetType().GetProperty("FirstLayerModel", flags)!.GetValue(acousticMask)!;
            var semantic = Enumerable.Range(0, 8).Select(i => i % 4).ToArray();
            var codes = new int[8, 1];
            for (int i = 0; i < 8; i++) codes[i, 0] = i % 4;
            var masked = Enumerable.Range(0, 8).Select(i => i >= 2 && i % 2 == 0).ToArray();
            var logits = (Tensor<T>)first.GetType().GetMethod("TrainingLogits")!.Invoke(first,
                new object[] { semantic, codes, 0, 2, masked, 0.37 })!;
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            return OptionSensitivityHarness.Fingerprint(logits.Select(value => ops.ToDouble(value)));
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> fish && option == "ConstrainToSemanticTokens")
            return ObserveFishFrames(fish, input);
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> durationMask && IsMaskGctDurationOption(option))
        {
            durationMask.SetTrainingMode(false);
            var settings = (AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions)durationMask.GetOptions();
            if (settings.TargetDurationSeconds is not null) throw new InvalidOperationException("Duration predictor probes require no fixed-duration override.");
            var phones = Enumerable.Range(0, 8).Select(i => 1 + i % 4).ToArray();
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var random = new Random(settings.SamplingSeed);
            int frames = (int)durationMask.GetType().GetMethod("TargetLength", flags)!.Invoke(durationMask,
                new object?[] { new[] { 1, 2 }, phones, 640, random, option == "DurationGuidance" ? new[] { 1, 1 } : null })!;
            long result = OptionSensitivityHarness.Fingerprint(new[] { (double)frames });
            if (option != "DurationMode")
            {
                var predictor = durationMask.GetType().GetProperty("DurationPredictor", flags)!.GetValue(durationMask)!;
                random = new Random(settings.SamplingSeed);
                var noise = phones.Select(_ => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble())).ToArray();
                var durations = (double[])predictor.GetType().GetMethod("Generate")!.Invoke(predictor,
                    new object[] { option == "DurationGuidance" ? new[] { 1, 2 } : Array.Empty<int>(),
                        option == "DurationGuidance" ? new[] { 1, 1 } : Array.Empty<int>(), phones, settings.DurationSteps, settings.DurationGuidance, noise })!;
                result = OptionSensitivityHarness.Combine(result, OptionSensitivityHarness.Fingerprint(durations));
            }
            return result;
        }
        long fingerprint = observe(network, ObservationInput(network, input, option));
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCT<T> mask && IsMaskGctSamplingOption(option))
        {
            var settings = (AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions)mask.GetOptions();
            var random = new Random(settings.SamplingSeed);
            Func<int, double[]> uniforms = count => Enumerable.Range(0, count).Select(_ => random.NextDouble()).ToArray();
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var semantic = (int[])mask.GetType().GetMethod("Generate", flags)!.Invoke(mask,
                new object[] { new[] { 1, 2 }, new[] { 3, 4 }, new[] { 0, 1 }, 32, uniforms })!;
            fingerprint = OptionSensitivityHarness.Combine(fingerprint,
                OptionSensitivityHarness.Fingerprint(semantic.Select(code => (double)code)));
            if (option == "SemanticToAcousticTemperature")
            {
                var codes = (int[,])mask.GetType().GetMethod("Acoustic", flags)!.Invoke(mask,
                    new object[] { semantic, new int[2, settings.NumQuantizers], uniforms })!;
                fingerprint = OptionSensitivityHarness.Combine(fingerprint,
                    OptionSensitivityHarness.Fingerprint(codes.Cast<int>().Select(code => (double)code)));
            }
            return fingerprint;
        }
        if (network is AiDotNet.Audio.Generation.AudioLM<T> sampledAudioLm && option is "SemanticTemperature" or "CoarseTemperature" or "FineTemperature" or "Seed")
        {
            var settings = (AiDotNet.Audio.Generation.AudioLMOptions)sampledAudioLm.GetOptions();
            // Observe generated tokens before frozen codec decoding: an unfitted codec can
            // reconstruct different sampled codes as the same waveform.
            var codes = sampledAudioLm.ContinueCodes(new int[settings.NumCodebooks, 0], Array.Empty<int>(), 0.16);
            return OptionSensitivityHarness.Combine(fingerprint,
                OptionSensitivityHarness.Fingerprint(codes.Cast<int>().Select(code => (double)code)));
        }
        if (network is AiDotNet.SpeechRecognition.CTCVariants.CIFDecoder<T> cif && option == "TailThreshold")
        {
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            foreach (int samples in new[] { 1600, 3200, 6400, 16000 })
            {
                var waveform = new Tensor<T>(new[] { samples });
                for (int i = 0; i < samples; i++) waveform[i] = ops.FromDouble(0.4 * Math.Sin(i * .13) + 0.2 * Math.Cos(i * .37));
                string text = cif.Transcribe(waveform, includeTimestamps: false).Text;
                fingerprint = OptionSensitivityHarness.Combine(fingerprint,
                    OptionSensitivityHarness.Fingerprint(text.Select(character => (double)character)));
            }
            return fingerprint;
        }
        if (network is AiDotNet.Audio.Generation.AudioLM<T> audioLm && option is "SemanticCropSeconds" or "CoarseCropSeconds" or "FineCropSeconds")
        {
            var settings = (AiDotNet.Audio.Generation.AudioLMOptions)audioLm.GetOptions();
            audioLm.CurrentStage = option switch
            {
                "SemanticCropSeconds" => AiDotNet.Audio.Generation.AudioLMStage.Semantic,
                "FineCropSeconds" => AiDotNet.Audio.Generation.AudioLMStage.FineAcoustic,
                _ => AiDotNet.Audio.Generation.AudioLMStage.CoarseAcoustic,
            };
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var semantic = new Tensor<T>(new[] { 8 });
            var codes = new Tensor<T>(new[] { 16, settings.NumCodebooks });
            for (int i = 0; i < semantic.Length; i++) semantic[i] = ops.FromDouble(i % settings.SemanticCodes);
            for (int frame = 0; frame < 16; frame++)
                for (int q = 0; q < settings.NumCodebooks; q++) codes[frame, q] = ops.FromDouble((frame + q) % settings.CodebookSize);
            double cropLoss = ops.ToDouble(audioLm.EvaluateTrainingObjective(semantic, codes));
            return OptionSensitivityHarness.Combine(fingerprint, OptionSensitivityHarness.Fingerprint(new[] { cropLoss }));
        }
        if (network is AiDotNet.SpeechRecognition.ProprietaryAPI.GoogleSpeechV2<T> google && IsGooglePretrainingOption(option))
        {
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            double pretrainingLoss = ops.ToDouble(google.PretrainingLoss(input, new Random(1337)));
            return OptionSensitivityHarness.Combine(fingerprint, OptionSensitivityHarness.Fingerprint(new[] { pretrainingLoss }));
        }
        if (network is AiDotNet.SpeechRecognition.Foundation.BESTRQ<T> bestRq && IsGooglePretrainingOption(option))
        {
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            double pretrainingLoss = ops.ToDouble(bestRq.PretrainingLoss(input, new Random(1337)));
            return OptionSensitivityHarness.Combine(fingerprint, OptionSensitivityHarness.Fingerprint(new[] { pretrainingLoss }));
        }
        if (network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> chatter && option == "VoiceEncoderPartialRate")
            return OptionSensitivityHarness.Fingerprint(chatter.VoiceEncoderModel!.Embed(ChatterboxFixture.Recording<T>(2), 24000)
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> chunked && option == "StaticChunkSize")
            return OptionSensitivityHarness.Fingerprint(chunked.FlowModel!.Encode(Enumerable.Range(0, 12).ToArray(),
                training: false, new Random(1337), streaming: true)
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> vocoder && option == "VocoderSegmentLength")
        {
            var targets = vocoder.VocoderTargets(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            {
                Tokens = input, Audio = ChatterboxFixture.Recording<T>(),
            }, new Random(1337));
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            return OptionSensitivityHarness.Combine(OptionSensitivityHarness.Fingerprint(targets.Audio.Select(ops.ToDouble)),
                OptionSensitivityHarness.Fingerprint((vocoder is AiDotNet.TextToSpeech.CodecBased.CosyVoice3<T> causal
                    ? causal.CausalVocoder : vocoder.Vocoder)!.Forward(targets.Mel, new Random(1337)).Select(ops.ToDouble)));
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoice3<T> dit
            && option is "DiTTrainingCfgRate" or "DiTSigmaMin")
        {
            var settings = (AiDotNet.TextToSpeech.CodecBased.CosyVoice3Options)dit.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var mel = new Tensor<T>(new[] { 1, settings.MelChannels, 6 });
            for (int i = 0; i < mel.Length; i++) mel[i] = ops.FromDouble(0.2 * Math.Sin(i * 0.17));
            var speaker = new Tensor<T>(new[] { settings.SpeakerEmbeddingDim }); speaker[0] = ops.One;
            var ditLoss = dit.DiTFlowModel!.TrainingLoss(new[] { 1, 2, 3 }, mel, speaker, false, new Random(1337));
            return OptionSensitivityHarness.Fingerprint(ditLoss.Select(ops.ToDouble));
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoiceBase<T> cosySampler
            && cosySampler.GetOptions() is AiDotNet.TextToSpeech.CodecBased.CosyVoiceAutoregressiveOptions
                or AiDotNet.TextToSpeech.CodecBased.CosyVoice3Options
            && option is "TopP" or "TopK" or "RepetitionWindow" or "RepetitionRatio")
        {
            const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Instance
                | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic;
            object? languageModel = null;
            for (var type = network.GetType(); type is not null && languageModel is null; type = type.BaseType)
                languageModel = type.GetProperty("LanguageModel", flags)?.GetValue(network);
            var configuration = languageModel!.GetType().GetProperty("Configuration", flags)!.GetValue(languageModel)!;
            object Setting(string name) => configuration.GetType().GetProperty(name)!.GetValue(configuration)!;
            var sampler = languageModel.GetType().Assembly.GetType("AiDotNet.TextToSpeech.CodecBased.CosyVoiceLm`1")!.MakeGenericType(typeof(T));
            var method = sampler.GetMethods(System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)
                .Single(candidate => candidate.Name == "SampleRepetitionAware" && candidate.GetParameters().Length == 8);
            // One recent repeat is exactly the default 10 * .1 boundary. A second draw
            // chooses the other token, exposing whether the real configured sampler resampled.
            int token = (int)method.Invoke(null, new object[] { new[] { Math.Log(.6), Math.Log(.4) },
                option is "TopP" or "TopK" ? Array.Empty<int>() : new[] { 0 },
                Setting("TopP"), Setting("TopK"), Setting("RepetitionWindow"),
                Setting("RepetitionRatio"), option is "TopP" or "TopK" ? new NucleusBoundaryRandom() : new RepetitionBoundaryRandom(),
                configuration.GetType().GetProperty("ExcludeRepeatedTokenOnFallback")?.GetValue(configuration) ?? true })!;
            return OptionSensitivityHarness.Combine(fingerprint, token);
        }
        if (network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> emotionIndex && option is "EmotionGroupSizes" or "EmotionMode")
        {
            var settings = (AiDotNet.TextToSpeech.Latest.IndexTTS2Options)emotionIndex.GetOptions();
            int rows = settings.EmotionGroupSizes!.Sum();
            // Nonconstant rows distinguish group boundaries, means and nearest-speaker selection.
            var emotions = Enumerable.Range(0, rows).Select(r =>
                Enumerable.Range(0, settings.HiddenDim).Select(k => (r + 1) * 0.1 + k * 0.01).ToArray()).ToArray();
            var speakers = Enumerable.Range(0, rows).Select(r =>
                Enumerable.Range(0, settings.StyleDim).Select(k => k == 0 ? 1.0 : (r + 1) * 0.1).ToArray()).ToArray();
            emotionIndex.LoadEmotionTable(emotions, speakers);
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var style = new Tensor<T>(new[] { 1, settings.StyleDim });
            style[0] = ops.FromDouble(1);
            var weights = new double[settings.EmotionLabels.Length];
            weights[0] = 1;
            var mixed = (Tensor<T>)typeof(AiDotNet.TextToSpeech.Latest.IndexTTS2<T>).GetMethod("EmotionTableMix",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .Invoke(emotionIndex, new object[] { weights, style, new Random(1337) })!;
            return OptionSensitivityHarness.Combine(fingerprint,
                OptionSensitivityHarness.Fingerprint(mixed.Select(value => ops.ToDouble(value))));
        }
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.E3TTS<T> e3 && option is "SilenceBlockSize" or "SilenceMagnitudeThreshold")
        {
            var waveform = new Tensor<T>(new[] { 16 });
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < 9; i++) waveform[i] = ops.FromDouble(0.075);
            var trimmed = (Tensor<T>)typeof(AiDotNet.TextToSpeech.FlowDiffusion.E3TTS<T>).GetMethod("TrimTrailingSilence",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.Invoke(e3, new object[] { waveform })!;
            return OptionSensitivityHarness.Combine(fingerprint,
                OptionSensitivityHarness.Fingerprint(trimmed.Select(value => ops.ToDouble(value))));
        }
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
        if (network is AiDotNet.Audio.Speaker.CAMPlusPlus<T> cam && option is "FrameShiftMilliseconds" or "FrameLengthMilliseconds")
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

    private static bool IsStyleTts2JointOption(string option) => StyleTts2Fixture.IsSlmAdversarialOption(option) || option is
        "DropoutRate" or "EmbeddingMaskProbability" or "FineTuneLearningRate" or "DurationCrossEntropyWeight"
        or "LogSigmaMean" or "LogSigmaStd" or "DiffusionStartStep" or "BertWeightDecay" or "BertLearningRate"
        or "DiffusionLossWeight" or "DurationLossWeight" or "JointStartStep" or "EstimateSigmaData"
        or "BertBeta1" or "SlmSampleRate" or "GeneratorLossWeight" or "F0LossWeight"
        or "NormLossWeight" or "StyleLossWeight" or "SlmLossWeight";

    private static bool IsCosyFlowOption(string option) => option is
        "TrainingCfgRate" or "FlowLoss" or "FlowAttentionDropoutRate" or "DropoutRate"
        or "DiTTrainingCfgRate" or "DiTSigmaMin" or "StaticChunkSize";

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
        => AiDotNet.Tests.VisionLanguage.NormalizationOptionFixture.Handles(network, option) ? 0
        : network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> && option == "Watermark" ? 0
        : network is AiDotNet.TextToSpeech.CodecBased.SPEARTTS<T> && SpearOptionFixture.Handles(option) ? 0
        : network is AiDotNet.TextToSpeech.Latest.OuteTTS<T> && OuteOptionFixture.Handles(option) ? 0
        : network is AiDotNet.Video.Enhancement.Upscale4KAgent<T> && option == "RollbackThreshold" ? 0
        // Graph, inference and dropout probes already observe their native path directly; reserve the
        // full 100M-parameter backward pass for the optimizer settings that require it.
        : network is AiDotNet.TextToSpeech.EndToEnd.Kokoro<T> && option is not ("LearningRate" or "WeightDecay")
            ? 0
            : network is AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> or AiDotNet.TextToSpeech.MultiModal.AudioPaLM<T>
            && option.StartsWith("Ramp", StringComparison.Ordinal)
            ? Math.Max(4, steps)
            : network is AiDotNet.Audio.MusicGen.MusicGenModel<T> && option is "WarmupSteps" or "TrainingSteps"
            ? Math.Max(4, steps)
            : network is AiDotNet.TextToSpeech.FlowDiffusion.CoMoSpeech<T> && option is "DistillationSteps" or "EmaDecay"
            ? Math.Max(2, steps)
            : network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T> && option == "SlmDiscriminatorInterval"
            ? Math.Max(3, steps)
            : network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T> && option == "BertBeta1"
            ? Math.Max(2, steps)
            : network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTSZS<T> && option == "TmaStartStep"
            ? Math.Max(2, steps)
            : network is AiDotNet.SpeechRecognition.Foundation.SPIRAL<T> && IsSpiralPretrainingOption(option)
            || network is AiDotNet.TextToSpeech.MultiModal.WhisperSpeech<T>
            || network is AiDotNet.TextToSpeech.MultiModal.SpeechT5<T> && SpeechT5Fixture.IsPretraining(option)
            || network is AiDotNet.TextToSpeech.FlowDiffusion.E3TTS<T> && option == "TrainingBetaEnd"
            ? Math.Max(3, steps) : steps;

    internal static void PrepareOptionTraining<T>(INeuralNetworkModel<T> network, string option)
    {
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.CoMoSpeech<T> como
            && option is "DistillationSteps" or "EmaDecay"
            && como.CurrentPhase == AiDotNet.TextToSpeech.FlowDiffusion.CoMoSpeechTrainingPhase.Teacher)
            como.BeginDistillation();

        if (network is AiDotNet.TextToSpeech.Classic.AlignTTS<T> align && option == "LearningRate")
            align.CurrentPhase = AiDotNet.TextToSpeech.Classic.AlignTTSTrainingPhase.JointFineTuning;
        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T> style && IsStyleTts2JointOption(option))
            style.CurrentStage = AiDotNet.TextToSpeech.StyleEmotion.StyleTts2TrainingStage.Joint;
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
        if (network is AiDotNet.TextToSpeech.Vocoders.WaveGlow<T> waveGlow && option == "SegmentSamples")
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.WaveGlowOptions)waveGlow.GetOptions();
            waveGlow.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            { Tokens = new Tensor<T>(new[] { 1 }), Audio = VocoderRecording<T>(settings) });
            return true;
        }
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.F5TTS<T> or AiDotNet.TextToSpeech.FlowDiffusion.E2TTS<T>
            && option == "AudioDropProbability")
        {
            var flow = (AiDotNet.TextToSpeech.FlowDiffusion.FlowMatchingTtsModelBase<T>)network;
            var infillingOptions = (AiDotNet.TextToSpeech.TtsModelOptions)flow.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            // A real partially observed mel span: the minimum recording may leave no audio condition to drop.
            var mel = new Tensor<T>(new[] { 16, infillingOptions.MelChannels });
            for (int i = 0; i < mel.Length; i++) mel[i] = ops.FromDouble(0.4 * Math.Sin(i * 0.17) + 0.2 * Math.Cos(i * 0.31));
            flow.Train(input, mel);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.EndToEnd.VITS2<T> alignment && option == "AlignmentNoiseDecay")
        {
            var settings = (AiDotNet.TextToSpeech.EndToEnd.VITS2Options)alignment.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            // The generic minimum-length recording can give each token exactly one frame:
            // MAS then has only one valid path, regardless of the actual noise schedule.
            var tokens = new Tensor<T>(new[] { 4 });
            for (int i = 0; i < tokens.Length; i++) tokens[i] = ops.FromDouble((i + 1) % settings.VocabSize);
            int hop = settings.UpsampleRates.Aggregate(1, (a, b) => a * b);
            var audio = new Tensor<T>(new[] { hop * 4 * tokens.Length });
            for (int i = 0; i < audio.Length; i++) audio[i] = ops.FromDouble(0.3 * Math.Sin(i * 0.17) + 0.1 * Math.Sin(i * 0.31));
            alignment.Train(tokens, audio);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.UniAudio<T> uniAudio && UniAudioOptionFixture.IsCodebookOption(option))
        {
            UniAudioOptionFixture.TrainCodebooks(uniAudio);
            return true;
        }
        if (network is AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T> audioPalm)
        {
            AudioPalmAsrOptionFixture.Train(audioPalm);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.SoundStorm<T> soundStorm && SoundStormOptionFixture.IsTrainingOption(option))
        {
            SoundStormOptionFixture.Train(soundStorm, option);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.Pheme<T> pheme && PhemeOptionFixture.IsTrainingOption(option))
        {
            PhemeOptionFixture.Train(pheme, option);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.MARS5TTS<T> mars5 && Mars5OptionFixture.IsNarOption(option))
        {
            Mars5OptionFixture.Train(mars5);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.SeedTTS<T> seedTts)
        {
            SeedTtsOptionFixture.Train(seedTts);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech3<T> naturalSpeech && NaturalSpeech3OptionFixture.IsCodecOption(option))
        {
            NaturalSpeech3OptionFixture.Train(naturalSpeech);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.VoiceCraft<T> voiceCraft)
        {
            VoiceCraftOptionFixture.Train(voiceCraft);
            return true;
        }
        if (network is AiDotNet.Audio.MusicGen.MusicGenModel<T> musicGen && AiDotNet.Tests.Audio.MusicGenOptionFixture.TryTrain(musicGen, input, option))
            return true;
        if (network is AiDotNet.TextToSpeech.MultiModal.GLM4Voice<T> glmVoice && GlmVoiceOptionFixture.TryTrain(glmVoice, option))
            return true;
        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTSZS<T> styleZs)
        {
            StyleTtsZsFixture.Train(styleZs, option);
            return true;
        }

        if (network is AiDotNet.TextToSpeech.MultiModal.WhisperSpeech<T> whisperSpeech)
        {
            whisperSpeech.Train(WhisperSpeechFixture.Sample<T>());
            return true;
        }

        if (network is AiDotNet.TextToSpeech.MultiModal.SpeechT5<T> speechT5)
        {
            speechT5.Train(SpeechT5Fixture.Sample<T>(option));
            return true;
        }

        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T> style)
        {
            int hop = ((AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2Options)style.GetOptions()).HopSize;
            style.Train(StyleTts2Fixture.IsSlmAdversarialOption(option)
                ? StyleTts2Fixture.SlmAdversarialSample<T>(hop) : StyleTts2Fixture.Sample<T>(hop));
            return true;
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.Zonos<T> zonos && ZonosFixture.TryTrain(zonos, input, option))
            return true;

        if (network is AiDotNet.TextToSpeech.CodecBased.SparkTTS<T> spark && SparkNativeFixture.IsAnnotatedOption(option))
        {
            spark.TrainAnnotatedSample(new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, Audio = SparkNativeFixture.Recording<T>() },
                true, AiDotNet.TextToSpeech.CodecBased.SparkTTSControlLevel.Moderate,
                AiDotNet.TextToSpeech.CodecBased.SparkTTSControlLevel.High, 2, 3);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.MultiModal.SpiritLM<T> spirit)
        {
            if (option is "VocoderLearningRate" or "VocoderLearningRateDecay" or "VocoderUpdatesPerEpoch")
            {
                spirit.TrainPitchQuantizer(SpiritLmFixture.Tones<T>(1.0));
                return true;
            }
            if (option == "DropoutRate")
            {
                int width = ((AiDotNet.TextToSpeech.MultiModal.SpiritLMOptions)spirit.GetOptions()).SonarWidth;
                var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
                var semantic = new Tensor<T>(new[] { width });
                for (int i = 0; i < width; i++) semantic[i] = ops.FromDouble(Math.Sin(i * 0.37));
                spirit.TrainSonarExpressive(SpiritLmFixture.Tones<T>(0.25), semantic, true);
                return true;
            }
            if (option == "SpeechPropLearningRate")
            {
                spirit.TrainSpeechPropStyle(SpiritLmFixture.Tones<T>(0.25), 1);
                return true;
            }
        }
        if (network is AiDotNet.TextToSpeech.FlowDiffusion.E3TTS<T> e3 && option == "PaddingLossWeight"
            && e3.GetOptions() is AiDotNet.TextToSpeech.FlowDiffusion.E3TTSOptions e3Options)
        {
            var recording = new Tensor<T>(new[] { Math.Max(1, e3Options.TrainingWaveformSamples / 2) });
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < recording.Length; i++) recording[i] = ops.FromDouble(0.4 * Math.Sin(i * 0.13));
            e3.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, Text = "test", Audio = recording });
            return true;
        }
        if ((network is AiDotNet.TextToSpeech.Vocoders.DiffusionVocoderBase<T> && option == "CropFrames"
                || network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option is "SegmentSize" or "SamplingSeed"
                || network is AiDotNet.TextToSpeech.Vocoders.BigVGAN<T> && option == "SegmentSize")
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
        if (options is AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions chatterbox && option == "Watermark")
            chatterbox.WatermarkModelPath = AiDotNet.Tests.Audio.PerthReleasedCheckpointFixture.CheckpointPath;
        if (options is AiDotNet.TextToSpeech.Vocoders.WaveGlowOptions waveGlow && option == "SegmentSamples")
            waveGlow.SegmentSamples = 4 * waveGlow.HopSize;
        if (options is AiDotNet.TextToSpeech.CodecBased.SPEARTTSOptions spear) SpearOptionFixture.Prepare(spear, option);
        if (options is AiDotNet.TextToSpeech.Latest.OuteTTSOptions oute) OuteOptionFixture.Prepare(oute, option);
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.F5TTSOptions f5)
        {
            f5.WarmupSteps = 0;
            if (option == "TotalSteps") { f5.WarmupSteps = 0; f5.TotalSteps = 6; }
            if (option == "MaxGradientNorm") f5.MaxGradientNorm = 1e-8;
            if (option == "AudioDropProbability")
            {
                f5.ConditionDropProbability = 0;
                f5.MaskFractionMin = f5.MaskFractionMax = 0.5;
                f5.AudioDropProbability = 0;
            }
        }
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.E2TTSOptions e2)
        {
            e2.WarmupSteps = 0;
            if (option == "TotalSteps") { e2.WarmupSteps = 0; e2.TotalSteps = 6; }
            if (option == "MaxGradientNorm") e2.MaxGradientNorm = 1e-8;
            if (option == "AudioDropProbability")
            {
                e2.ConditionDropProbability = 0;
                e2.MaskFractionMin = e2.MaskFractionMax = 0.5;
                e2.AudioDropProbability = 0;
            }
        }
        if (options is AiDotNet.TextToSpeech.Classic.GradTTSOptions grad && option == "NumDecoderLayers")
        {
            // Three released resolutions and eight mel bins allow a fourth genuine U-Net resolution.
            grad.MelChannels = 8;
            grad.DecoderDimMultipliers = new[] { 1, 2, 4 };
        }

        if (options is AiDotNet.TextToSpeech.CodecBased.OrpheusTTSOptions orpheus && option == "MaxGradNorm")
            // Bind the actual optimizer clip in the compact LM; the release threshold can exceed every test gradient.
            orpheus.MaxGradNorm = 1e-12;

        // The single-speaker, monolingual graph has no conditioning tables to resize.
        // Enable the native conditioning path in both builds before perturbing its width.
        if ((options.GetType() == typeof(AiDotNet.TextToSpeech.EndToEnd.VITSOptions)
             || options.GetType() == typeof(AiDotNet.TextToSpeech.EndToEnd.PiperOptions))
            && options is AiDotNet.TextToSpeech.EndToEnd.VitsModelOptions vits)
        {
            if (option == "SpeakerEmbeddingDim") vits.NumSpeakers = 2;
            if (option == "LanguageEmbeddingDim") vits.NumLanguages = 2;
        }
        if (options is AiDotNet.TextToSpeech.EndToEnd.VITS2Options vits2)
        {
            if (option is "SpeakerEmbeddingDim" or "SpeakerConditionedEncoderBlock") vits2.NumSpeakers = 2;
            // The paper conditions the third block: retain it when testing multi-speaker configurations.
            if (option is "SpeakerEmbeddingDim" or "SpeakerConditionedEncoderBlock" or "NumSpeakers")
                vits2.NumEncoderLayers = 3;
            if (option == "LanguageEmbeddingDim") vits2.NumLanguages = 2;
            // At two steps the paper's 0.01-scale noise can leave the discrete MAS path unchanged.
            // Exercise the same native schedule with noise large enough to alter alignment decisions.
            if (option == "AlignmentNoiseDecay") vits2.AlignmentNoiseScale = 1.0;
        }
        if (options is AiDotNet.TextToSpeech.EndToEnd.YourTTSOptions your)
        {
            if (option == "LanguageEmbeddingDim") your.NumLanguages = 2;
            if (option == "SpeakerConsistencyLossWeight") your.UseSpeakerConsistencyLoss = true;
        }


        if (options is AiDotNet.TextToSpeech.CodecBased.VoiceboxOptions durationVoiceboxOptions && option == "MaximumPhoneDuration")
            durationVoiceboxOptions.MaximumPhoneDuration = 2;

        if (options is AiDotNet.TextToSpeech.MultiModal.MinMoOptions minMoOptions)
            MinMoOptionFixture.Prepare(minMoOptions, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.CSMOptions csmOptions)
            CsmOptionFixture.Prepare(csmOptions, option);

        if (options is AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1BOptions metaVoiceOptions)
            MetaVoiceOptionFixture.Prepare(metaVoiceOptions, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.TortoiseTTSOptions tortoiseOptions)
            TortoiseOptionFixture.Prepare(tortoiseOptions, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.VoiceboxOptions voiceboxMask && option == "SpectrogramMaskMinimum")
            voiceboxMask.SpectrogramDropProbability = 0;

        if (options is AiDotNet.VisionLanguage.Grounding.GroundedSAM2Options groundedSam && option is "MaskThreshold" or "SamDecoderHeads")
        {
            groundedSam.Detector.ConfidenceThreshold = 0;
            groundedSam.Detector.MaxDetections = 1;
        }

        if (options is AiDotNet.TextToSpeech.Classic.DeepVoice3Options deepVoice && option == "GradientClipValue")
            deepVoice.GradientClipValue = 1e-5;

        if (options is AiDotNet.TextToSpeech.CodecBased.BarkOptions bark && BarkOptionFixture.Handles(option))
            BarkOptionFixture.PrepareOptions(bark, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.VevoOptions vevo && option == "MaxLength")
            VevoOptionFixture.PrepareOptions(vevo);

        if (options is AiDotNet.TextToSpeech.DescriptionBased.ParlerTTSOptions parler && option == "CodebookWeights")
            parler.CodebookWeights ??= Enumerable.Repeat(1.0, parler.NumCodebooks).ToArray();

        if (options is AiDotNet.Video.Options.Upscale4KAgentOptions agent && option == "RollbackThreshold")
            AiDotNet.Tests.Video.Upscale4KOptionFixture.PrepareOptions(agent);
        if (options is AiDotNet.TextToSpeech.CodecBased.UniAudioOptions uniAudio)
            UniAudioOptionFixture.PrepareOptions(uniAudio, option);
        if (options is AiDotNet.TextToSpeech.EndToEnd.KokoroOptions kokoro)
            KokoroOptionFixture.PrepareOptions(kokoro, option);
        if (options is AiDotNet.SpeechRecognition.CTCVariants.CTCSegmentationOptions segmentation && option == "ConfidenceWindow")
            segmentation.ConfidenceWindow = 2;
        // Infinity's only finite generic candidate is 1, above every LegS step 1/(t+1).
        // A finite cap below one exercises the upper-clamp contract.
        if (option == "TimescaleMax" && options.GetType().IsGenericType
            && options.GetType().GetGenericTypeDefinition() == typeof(AiDotNet.Models.Options.HippoOptions<>))
            options.GetType().GetProperty("TimescaleMax")!.SetValue(options, .1);
        AiDotNet.Tests.IntegrationTests.SyntheticData.SyntheticGanOptionFixture.PrepareOptions(options, option);
        if (options is AiDotNet.TextToSpeech.MultiModal.AudioPaLMOptions audioPalm)
            AudioPalmAsrOptionFixture.PrepareOptions(audioPalm, option);
        if (options is AiDotNet.TextToSpeech.VoiceCloning.XTTSv2Options xtts)
            XttsOptionFixture.PrepareOptions(xtts, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.SoundStormOptions soundStorm)
            SoundStormOptionFixture.PrepareOptions(soundStorm, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.PhemeOptions pheme)
            PhemeOptionFixture.PrepareOptions(pheme, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.MARS5TTSOptions mars5)
            Mars5OptionFixture.PrepareOptions(mars5, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.SeedTTSOptions seedTts)
            SeedTtsOptionFixture.PrepareOptions(seedTts, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech3Options naturalSpeech)
            NaturalSpeech3OptionFixture.PrepareOptions(naturalSpeech, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.VoiceCraftOptions voiceCraft)
            VoiceCraftOptionFixture.PrepareOptions(voiceCraft, option);
        if (options is AiDotNet.TextToSpeech.MultiModal.GLM4VoiceOptions glmVoice)
            GlmVoiceOptionFixture.PrepareOptions(glmVoice, option);
        if (options is AiDotNet.Audio.MusicGen.MusicGenOptions musicGen)
            AiDotNet.Tests.Audio.MusicGenOptionFixture.PrepareOptions(musicGen, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.VALLE2Options valle2)
            Valle2OptionFixture.PrepareOptions(valle2, option);

        if (options is AiDotNet.TextToSpeech.Latest.MegaTTS3Options mega)
            MegaTts3Fixture.PrepareOptions(mega, option);

        if (options is AiDotNet.TextToSpeech.StyleEmotion.StyleTTSZSOptions styleZs)
            StyleTtsZsFixture.PrepareOptions(styleZs, option);

        if (options is AiDotNet.TextToSpeech.MultiModal.WhisperSpeechOptions whisperSpeech)
            WhisperSpeechFixture.PrepareOptions(whisperSpeech, option);

        if (options is AiDotNet.TextToSpeech.MultiModal.SpeechT5Options speechT5)
            SpeechT5Fixture.PrepareOptions(speechT5, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.ZonosOptions zonos)
            ZonosFixture.PrepareOptions(zonos, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.GPTSoVITSOptions gptSoVits)
            GptSoVitsFixture.PrepareOptions(gptSoVits, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.VALLEXOptions vallEX)
            VallEXCloneFixture.PrepareOptions(vallEX, option);

        if (options is AiDotNet.TextToSpeech.CodecBased.FireRedTTSOptions fireRed)
            FireRedTtsFixture.PrepareOptions(fireRed, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.SparkTTSOptions spark)
            SparkNativeFixture.PrepareOptions(spark, option);
        if (options is AiDotNet.TextToSpeech.MultiModal.SpiritLMOptions spirit)
            SpiritLmFixture.PrepareOptions(spirit, option);
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions guidanceMask && option == "GuidanceRescale")
            guidanceMask.Guidance = 2.5;
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions acousticMask && option == "SemanticToAcousticHeads")
            acousticMask.CurrentStage = AiDotNet.TextToSpeech.FlowDiffusion.MaskGctStage.FirstAcousticLayer;
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions durationMask && IsMaskGctDurationOption(option))
        {
            durationMask.TargetDurationSeconds = null;
            durationMask.DurationMode = AiDotNet.TextToSpeech.FlowDiffusion.MaskGctDurationMode.Predictor;
            durationMask.DurationSteps = 2;
            durationMask.CurrentStage = AiDotNet.TextToSpeech.FlowDiffusion.MaskGctStage.Duration;
        }
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.MaskGCTOptions mask && IsMaskGctSamplingOption(option))
        {
            mask.FilterThreshold = 0; // A top-one filter removes every temperature/sampling distinction.
            mask.Guidance = 2.5;
        }
        if (options is AiDotNet.Audio.Effects.DACOptions dac && option == "QuantizerDropout")
            dac.NumQuantizers = 4; // With a single quantizer, every dropout draw still selects one.
        if (options is AiDotNet.TextToSpeech.CodecBased.FishSpeechFamilyOptions fish && IsFishScheduleOption(option))
        {
            fish.WarmupSteps = option == "WarmupSteps" ? 1 : 0;
            fish.TrainingSteps = 4;
            fish.CosineDecay = true;
            fish.FinalLearningRateRatio = 0.5;
        }
        if (options is AiDotNet.Audio.Generation.AudioLMOptions audioLm && option is "SemanticCropSeconds" or "CoarseCropSeconds" or "FineCropSeconds")
        {
            audioLm.SemanticCropSeconds = 0.08;
            audioLm.CoarseCropSeconds = 0.08;
            audioLm.FineCropSeconds = 0.08;
        }
        if (options is AiDotNet.SpeechRecognition.Foundation.BESTRQOptions bestRq && IsGooglePretrainingOption(option))
        {
            bestRq.MaskProbability = 0.5;
            bestRq.MaskLength = 1;
        }
        if (options is AiDotNet.SpeechRecognition.ProprietaryAPI.GoogleSpeechV2Options google && IsGooglePretrainingOption(option))
        {
            google.MaskProbability = 0.5;
            google.MaskLength = 1;
        }
        if (options is AiDotNet.TextToSpeech.FlowDiffusion.E3TTSOptions e3)
        {
            if (option == "DefaultSpeakerId") e3.SpeakerCount = 2;
            if (option == "TrainingBetaEnd") e3.NumDiffusionSteps = 4;
            if (option is "SilenceBlockSize" or "SilenceMagnitudeThreshold")
            {
                e3.SilenceBlockSize = 4;
                e3.SilenceMagnitudeThreshold = 0.05;
            }
        }
        if (options is AiDotNet.Models.Options.DGCNNOptions dgcnn && option is "DropoutRate" or "UseDropout")
        {
            dgcnn.UseDropout = true;
            dgcnn.DropoutRate = 0.5;
        }
        if (options is AiDotNet.VisionLanguage.Grounding.GLaMMOptions glamm && option == "TokenizerPieces")
        {
            glamm.TokenizerModel = null;
            glamm.TokenizerPath = null; // Vocabulary sizing applies before a tokenizer asset is supplied.
        }
        if (options is AiDotNet.TextToSpeech.Latest.IndexTTS2Options stagedIndex)
        {
            if (option is "EmotionGroupSizes" or "EmotionMode")
            {
                stagedIndex.EmotionGroupSizes = Enumerable.Repeat(2, stagedIndex.EmotionLabels.Length).ToArray();
                stagedIndex.EmotionMode = AiDotNet.TextToSpeech.Latest.IndexTts2EmotionMode.Paper;
            }
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
        if (options is AiDotNet.TextToSpeech.Vocoders.BigVGANOptions bigVgan && option == "SegmentSize")
            bigVgan.SegmentSize = checked(2 * bigVgan.HopSize);

        if (options is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGANOptions gan)
        {
            if (option == "AdversarialWeight") gan.PretrainingSteps = 0;
            if (option is "SegmentSize" or "SamplingSeed") gan.SegmentSize = checked(2 * gan.HopSize);
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGANOptions multiBand && option is "MelMean" or "MelScale")
        {
            multiBand.MelMean = new double[multiBand.MelChannels];
            multiBand.MelScale = Enumerable.Repeat(1.0, multiBand.MelChannels).ToArray();
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.VoiceboxOptions voicebox && option == "MaximumAudioFrames")
            voicebox.MaximumPhoneDuration = 1;
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoiceConformerOptions cosy && IsCosyFlowOption(option))
            cosy.TrainingCfgRate = 0.5;
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoice3Options dit && option == "DiTTrainingCfgRate")
            dit.DiTTrainingCfgRate = 0.5;
        if (options is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2Options styleDecoder)
        {
            if (StyleTts2Fixture.IsSlmAdversarialOption(option))
                StyleTts2Fixture.PrepareSlmAdversarialOptions(styleDecoder, option);
            // The aligner objectives begin only after the paper's TMA warm-up.
            // Both the control and variant must train the aligner before probing its loss weights.
            if (option is "MonotonicLossWeight" or "S2sLossWeight")
                styleDecoder.TmaStartStep = 0;
            if (option == "DecoderType")
            {
                styleDecoder.IstftHopSize = 1;
                styleDecoder.IstftFftSize = 4;
            }
            if (option == "IstftFftSize")
            {
                styleDecoder.DecoderType = AiDotNet.TextToSpeech.StyleEmotion.StyleTts2DecoderType.IstftNet;
                styleDecoder.HopSize = styleDecoder.UpsampleRates.Aggregate(1, (a, b) => a * b) * styleDecoder.IstftHopSize;
            }
        }
        if (options is AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions chatter && option == "VoiceEncoderPartialRate")
            chatter.VoiceEncoderPartialFrames = 160;
        if (options is AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions chunked && option == "StaticChunkSize")
            chunked.StaticChunkSize = 2;
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoiceAutoregressiveOptions sampling && option == "TopK")
        {
            sampling.TopK = 2;
            sampling.TopP = 1;
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.CosyVoice3Options sampling3 && option == "TopK")
        {
            sampling3.TopK = 2;
            sampling3.TopP = 1;
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

    internal static void PrepareConstruction<T>(INeuralNetworkModel<T> network, string option)
    {
        // An explicit architecture seed wins over ModelOptions.Seed. Release that fixture override
        // on the shell so the rebuilt control and candidate can exercise the option's own seed.
        if (option == nameof(AiDotNet.Models.Options.ModelOptions.Seed)
            && network is AiDotNet.NeuralNetworks.NeuralNetworkBase<T> neural)
            neural.Architecture.RandomSeed = null;
    }


    internal static void UseConfiguredOptimizer<T>(INeuralNetworkModel<T> network)
    {
        if (network is not AiDotNet.TextToSpeech.TtsModelBase<T>
            and not AiDotNet.Audio.Codecs.NeuralAudioCodecBase<T>
            and not AiDotNet.SpeechRecognition.NeMo.CanaryQwen<T>) return;
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
        if (network is AiDotNet.TextToSpeech.EndToEnd.VITS<T> or AiDotNet.TextToSpeech.EndToEnd.Piper<T>
            or AiDotNet.TextToSpeech.EndToEnd.VITS2<T>
            && ((AiDotNet.TextToSpeech.TtsModelBase<T>)network).GetOptions() is AiDotNet.TextToSpeech.EndToEnd.VitsModelOptions vits
            && (vits.NumSpeakers > 1 || vits.NumLanguages > 1))
            ((AiDotNet.TextToSpeech.TtsModelBase<T>)network).Voice = new AiDotNet.TextToSpeech.TtsVoice<T>
            {
                SpeakerId = 0, LanguageId = 0,
            };

        if (network is AiDotNet.TextToSpeech.CodecBased.GPTSoVITS<T> gptSoVits)
            GptSoVitsFixture.Prepare(gptSoVits);

        switch (network)
        {
            case AiDotNet.Audio.Generation.SoundStream<T> model:
                // Configuration clones preserve the source's public runtime flag, which can
                // override the options candidate applied by the probe's reconstructed constructor.
                model.Denoise = ((AiDotNet.Audio.Generation.SoundStreamOptions)model.GetOptions()).Denoise;
                break;
            case AiDotNet.TextToSpeech.CodecBased.VoiceCraft<T> model:
                VoiceCraftNativeFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.MultiModal.GLM4Voice<T> model:
                GLM4VoiceFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.Latest.MegaTTS3<T> model:
                MegaTts3Fixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.CodecBased.SparkTTS<T> model:
                SparkNativeFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.MultiModal.SpiritLM<T> model:
                SpiritLmFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.Latest.KaniTTS<T> model:
                KaniTTSFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.Latest.KaniTTS2<T> model:
                KaniTTSFixture.Prepare(model);
                break;
            case AiDotNet.TextToSpeech.EndToEnd.Kokoro<T> model:
                KokoroNativeFixture.Prepare(model);
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
