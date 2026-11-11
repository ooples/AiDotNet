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
            or AiDotNet.TextToSpeech.MultiModal.Moshi<T>
            or AiDotNet.TextToSpeech.Latest.MegaTTS3<T>
            or AiDotNet.TextToSpeech.CodecBased.SparkTTS<T>;

    internal static long ParameterCountBeforeFitting<T>(INeuralNetworkModel<T> network)
        => network is AiDotNet.TextToSpeech.EndToEnd.Kokoro<T> kokoro
            ? kokoro.ParameterCount
            : network is AiDotNet.TextToSpeech.VoiceCloning.MetaVoice1B<T> metaVoice ? MetaVoiceOptionFixture.ParameterCount(metaVoice)
            : network.GetParameters().Length;

    internal static string ControlGroup<T>(INeuralNetworkModel<T> network, string option)
        => AiDotNet.Tests.VisionLanguage.NormalizationOptionFixture.Handles(network, option) ? "image-normalization"
            : network is AiDotNet.TextToSpeech.MultiModal.Moshi<T> ? "moshi-" + option
            : network is AiDotNet.TextToSpeech.Classic.PortaSpeech<T> && option is "PostNetLearningRate" or "PostNetTemperature" ? "portaspeech-postnet"
            : network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> && option == "Watermark" ? "chatterbox-published-perth"
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech<T> && option == "AddBlank" ? "natural-phoneme-frontend"
            : network is AiDotNet.TextToSpeech.Vocoders.WaveGlow<T> && option == "SegmentSamples" ? "waveglow-recording-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.UnivNet<T> && option == "SegmentSize" ? "univnet-recording-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.MelGAN<T> && option is "SegmentSize" or "SamplingSeed" ? "melgan-recording-crop"
            : network is AiDotNet.TextToSpeech.Vocoders.HiFiGAN<T> && IsHiFiGanTrainingOption(option) ? "hifigan-recording-" + option
            : network is AiDotNet.TextToSpeech.Vocoders.WaveNet<T> && IsWaveNetTrainingOption(option) ? "wavenet-recording-" + option
            : network is AiDotNet.TextToSpeech.Vocoders.Vocos<T> && IsVocosRecordingOption(option) ? "vocos-recording-" + option
            : network is AiDotNet.TextToSpeech.Vocoders.WaveRNN<T> && IsWaveRnnTrainingOption(option) ? "wavernn-training-" + option
            : network is AiDotNet.TextToSpeech.Vocoders.DiffWave<T> && option is "NumClasses" or "UtteranceSamples" ? "diffwave-task-" + option
            : network is AiDotNet.TextToSpeech.Vocoders.ParallelWaveGAN<T> && option is "AdversarialWeight" or "DiscriminatorGradientClipNorm" or "DiscriminatorLearningRate" ? "parallel-wavegan-adversarial-" + option
            : network is AiDotNet.TextToSpeech.Vocoders.ParallelWaveGAN<T> && option is "MelMean" or "MelScale" ? "parallel-wavegan-normalization"
            : network is AiDotNet.TextToSpeech.Vocoders.ParallelWaveGAN<T> && option is "SegmentSize" or "Epsilon" ? "parallel-wavegan-" + option
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

            : network is AiDotNet.TextToSpeech.CodecBased.Vevo<T> ? "vevo-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.Llasa<T> && LlasaOptionFixture.Handles(option) ? "llasa-" + option
            : network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech2<T> && NaturalSpeech2OptionFixture.Handles(option) ? "naturalspeech2-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.StepAudio<T> && StepAudioOptionFixture.IsChatSamplingOption(option) ? "stepaudio-chat"
            : network is AiDotNet.TextToSpeech.MultiModal.StepAudio<T> && StepAudioOptionFixture.IsPpoOption(option) ? "stepaudio-ppo-" + option
            : network is AiDotNet.TextToSpeech.MultiModal.StepAudio<T> && option == "LinguisticDither" ? "stepaudio-linguistic"
            : network is AiDotNet.TextToSpeech.CodecBased.VALLE<T> && option is "PromptSeconds" or "StopOnMostLikelyEndToken" ? "valle-" + option
            : network is AiDotNet.TextToSpeech.Classic.NonAttentiveTacotron<T>
                && (NonAttentiveTacotronOptionFixture.UsesTraining(option) || option is "MaxMelLength" or "MaxTextLength") ? "nat-" + option

            : network is AiDotNet.Finance.Forecasting.StateSpace.Hippo<T> && option == "TimescaleMax" ? "hippo-timescale-clamp"
            : network is AiDotNet.SpeechRecognition.CTCVariants.CTCSegmentation<T> && option == "ConfidenceWindow" ? "ctc-segmentation-confidence"
            : network is AiDotNet.TextToSpeech.Latest.MegaTTS3<T> ? "mega-" + option
            : network is AiDotNet.TextToSpeech.Latest.MegaTTS<T> or AiDotNet.TextToSpeech.Latest.MegaTTS2<T> ? "mega-" + option
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

    /// <summary>Several sentences, with Chinese numbers for the normalizer.</summary>
    private const string FishLongText = "Today is 2024年1月5日, it is 3.5 degrees. We walked to the station. The train was late again. " +
        "Then it rained for 12 hours. Everyone went home early. 第3章 starts tomorrow.";

    /// <summary>The models that run openai-whisper's native network and its transcribe() decoding loop.</summary>
    private static readonly HashSet<string> OpenAiWhisperNetworks = new(StringComparer.Ordinal)
    {
        "GladiaASR`1", "GroqWhisper`1", "NoiseRobustASR`1", "WhisperCPP`1", "DistilWhisper`1", "FasterWhisper`1",
        "KotobaWhisper`1", "WhisperLargeV3`1", "WhisperLargeV3Turbo`1", "WhisperLive`1", "WhisperTimestamped`1", "WhisperX`1",
    };

    /// <summary>A deterministic 10-second, 16 kHz waveform of shape [1, 160000].</summary>
    private static Tensor<T> LongWaveform<T>()
    {
        var waveform = Waveform<T>(160000);
        return new Tensor<T>(new[] { 1, 160000 }, waveform.ToVector());
    }

    /// <summary>A deterministic waveform of shape [samples].</summary>
    // offset + amplitude * sin(frequency * i) over i = 0 .. length - 1, built with engine ops.
    // [1, channels, frames] mel frames 0.2 sin(0.3 c + 0.2 f); sin(a + b) = sin a cos b + cos a sin b.
    private static Tensor<T> MelFrames<T>(int channels, int frames)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var grid = engine.TensorAdd(engine.TensorOuter(SineRamp<T>(channels, 0.3, 0.2), CosineRamp<T>(frames, 0.2, 1.0)),
            engine.TensorOuter(CosineRamp<T>(channels, 0.3, 0.2), SineRamp<T>(frames, 0.2, 1.0)));
        return Shaped(grid, new[] { 1, channels, frames });
    }

    private static Tensor<T> SineRamp<T>(int length, double frequency, double amplitude, double offset = 0.0, double phase = 0.0)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var angle = engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, length, 1), ops.FromDouble(frequency)), ops.FromDouble(phase));
        return engine.TensorAddScalar(engine.TensorMultiplyScalar(engine.TensorSin(angle), ops.FromDouble(amplitude)), ops.FromDouble(offset));
    }

    // amplitude * cos(frequency * i) over i = 0 .. length - 1.
    private static Tensor<T> CosineRamp<T>(int length, double frequency, double amplitude)
        => SineRamp<T>(length, frequency, amplitude, 0.0, Math.PI / 2);

    // A tensor of the given shape holding the ramp's values in row-major order.
    private static Tensor<T> Shaped<T>(Tensor<T> ramp, int[] shape) => AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Reshape(ramp, shape);

    // (start + i) mod modulus over i = 0 .. length - 1 (no modulus when it is zero).
    private static Tensor<T> IndexRamp<T>(int length, int start = 0, int modulus = 0)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var ramp = engine.TensorArange<T>(start, start + length, 1);
        return modulus > 0 ? engine.TensorRemainder(ramp, engine.TensorLinspace(ops.FromDouble(modulus), ops.FromDouble(modulus), length)) : ramp;
    }

    // A training target near the model's own prediction: 0.9 * prediction + amplitude * sin(i).
    private static Tensor<T> NearbyTarget<T>(Tensor<T> prediction, double amplitude)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var ripple = engine.Reshape(SineRamp<T>(prediction.Length, 1.0, amplitude), prediction.Shape.ToArray());
        return engine.TensorAdd(engine.TensorMultiplyScalar(prediction, ops.FromDouble(0.9)), ripple);
    }

    // frames x 3 x side x side pixels in [0, 255] of 127.5 + 100 sin(0.4 (x + f) + c) cos(0.3 y): one pattern shifted a
    // pixel a frame, so a flow estimator sees motion. sin(a + b) = sin a cos b + cos a sin b keeps it to outer products.
    private static Tensor<T> ShiftedClip<T>(int frames, int side)
    {
        const int channels = 3;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var columns = engine.TensorMultiplyScalar(engine.TensorArange<T>(0, side, 1), ops.FromDouble(0.4));
        var rows = engine.TensorCos(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, side, 1), ops.FromDouble(0.3)));
        // The phase offset 0.4 f + c of each (frame, channel), flattened to frames * channels.
        var frameOffsets = engine.TensorOuter(engine.TensorMultiplyScalar(engine.TensorArange<T>(0, frames, 1), ops.FromDouble(0.4)),
            engine.TensorLinspace(ops.One, ops.One, channels));
        var channelOffsets = engine.TensorOuter(engine.TensorLinspace(ops.One, ops.One, frames), engine.TensorArange<T>(0, channels, 1));
        var offsets = engine.Reshape(engine.TensorAdd(frameOffsets, channelOffsets), new[] { frames * channels });
        var sines = engine.TensorAdd(engine.TensorOuter(engine.TensorCos(offsets), engine.TensorSin(columns)),
            engine.TensorOuter(engine.TensorSin(offsets), engine.TensorCos(columns)));
        // [k, x] times rows[y] gives [k, x, y]; permute to [k, y, x].
        var pattern = engine.TensorPermute(engine.Reshape(engine.TensorOuter(engine.Reshape(sines, new[] { frames * channels * side }), rows),
            new[] { frames * channels, side, side }), new[] { 0, 2, 1 });
        return engine.Reshape(engine.TensorAddScalar(engine.TensorMultiplyScalar(pattern, ops.FromDouble(100)), ops.FromDouble(127.5)),
            new[] { frames, channels, side, side });
    }

    private static Tensor<T> Waveform<T>(int samples)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var waveform = new Tensor<T>(new[] { samples });
        waveform = Shaped(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(waveform.Length, 0.13, 0.4), CosineRamp<T>(waveform.Length, 0.37, 0.2)), waveform.Shape.ToArray());
        return waveform;
    }

    private static long ObserveTask<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option,
        Func<INeuralNetworkModel<T>, Tensor<T>, long> observe)
    {
        if (network is AiDotNet.Audio.Foundations.Wav2Vec2<T> wav2Vec2 && option == "MaskProbability")
        {
            // transformers' _compute_mask_indices draws max(floor(p*L/M + U), mask_time_min_masks = 2) spans, so on
            // the 2048-sample fixture (6 frames) every probability masks the same frames. A 10-second waveform has
            // ~500 frames, where the default and the doubled probability draw different span counts. The fixed-seed
            // pre-training objective is the path the probability feeds.
            var objective = (AiDotNet.Interfaces.ITrainingObjectiveProvider<T>)wav2Vec2;
            var waveform = LongWaveform<T>();
            return OptionSensitivityHarness.Fingerprint(new[]
            {
                AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble(
                    objective.EvaluateTrainingObjective(waveform, waveform)),
            });
        }
        if (network is AiDotNet.Audio.Foundations.WavLM<T> wavLm && option == "MaskTimeProbability")
        {
            // Same min-masks clamp as Wav2Vec2 above; the time mask applies on the training forward.
            using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
            return OptionSensitivityHarness.Fingerprint(wavLm.ForwardForTraining(LongWaveform<T>())
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        }
        if (network is AiDotNet.Interfaces.ISpeechRecognizer<T> whisper && option is "Temperatures" or "WordTimestamps"
            && OpenAiWhisperNetworks.Contains(network.GetType().Name))
        {
            // openai-whisper's decoding options: transcribe() retries a failed window at the next temperature and
            // word_timestamps splits segments by word. Both act in Transcribe; Predict returns the decoder's logits.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var result = whisper.Transcribe(Waveform<T>(16000));
            var values = result.Text.Select(character => (double)character).Append(result.Text.Length).ToList();
            foreach (var segment in result.Segments)
            {
                values.Add(segment.StartTime);
                values.Add(segment.EndTime);
                values.Add(segment.Text.Length);
                values.Add(ops.ToDouble(segment.Confidence));
            }
            return OptionSensitivityHarness.Fingerprint(values);
        }
        if (option is "AttentionChunkSeconds" or "CodeDim" or "MaskLength" or "MaskNoiseStd" or "MaskProbability" or "StackedFrames"
            && network switch
            {
                AiDotNet.SpeechRecognition.Multilingual.USM<T> usm => (Func<Tensor<T>, Random, T>?)((a, r) => usm.PretrainingLoss(a, r)),
                AiDotNet.SpeechRecognition.ProprietaryAPI.GoogleSpeechV2<T> googleV2 => (a, r) => googleV2.PretrainingLoss(a, r),
                AiDotNet.SpeechRecognition.Multilingual.Chirp<T> chirp => (a, r) => chirp.PretrainingLoss(a, r),
                AiDotNet.SpeechRecognition.Multilingual.Chirp2<T> chirp2 => (a, r) => chirp2.PretrainingLoss(a, r),
                AiDotNet.SpeechRecognition.Multilingual.Chirp3<T> chirp3 => (a, r) => chirp3.PretrainingLoss(a, r),
                _ => null,
            } is { } bestRqLoss)
            // BEST-RQ pre-training options (Zhang et al. 2023, section 2.2): they act in the model's pre-training loss,
            // which the probe's supervised CTC step never runs. A 10-second waveform spans more than one 8-second
            // attention chunk and holds masked spans at the 0.01 probability; the mask and noise draws are seeded.
            return OptionSensitivityHarness.Fingerprint(new[]
            {
                AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble(
                    bestRqLoss(LongWaveform<T>(), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7))),
            });
        if (network is AiDotNet.Audio.Generation.EnCodec<T> encodec && option is "ChunkSeconds" or "ChunkOverlap")
        {
            // Chunked coding (section 3.1 "non-streamable") applies in EncodeFrames, not in the probe's Predict.
            var frames = encodec.EncodeFrames(LongWaveform<T>());
            var values = new List<double>();
            foreach (var frame in frames)
            {
                values.Add(frame.Scale);
                foreach (int code in frame.Codes) values.Add(code);
            }
            return OptionSensitivityHarness.Fingerprint(values);
        }
        if (network is AiDotNet.Audio.Classification.AudioEventDetector<T> events && option == "PatchFrames")
            // A recording becomes one PatchFrames-frame log-mel patch (Hershey et al.'s 960 ms window); the fixture
            // feeds ready-made patches. The ResNet's global average pooling takes any patch length.
            return OptionSensitivityHarness.Fingerprint(events.GetEventProbabilities(LongWaveform<T>())
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS<T> styleMel && option == "WindowLength")
            // The analysis window acts in the model's mel front end (torchaudio MelSpectrogram), which the probe's
            // ready-made mel targets bypass.
            return OptionSensitivityHarness.Fingerprint(styleMel.ComputeMel(Waveform<T>(16000))
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.Audio.Classification.GenreClassifier<T> genre && option is "BeatWindow" or "BeatHop" or "TextureWindowFrames")
            // Tzanetakis and Cook's beat histogram reads 65,536-sample windows every 32,768 samples, and the texture
            // window spans 43 analysis frames; the fixture's clip is shorter than one beat window. Twelve seconds at
            // 22.05 kHz hold several of each. The options shape the 30 surface features; an unfitted classifier's
            // softmax over raw features (centroids in Hz) saturates, so the features themselves are observed.
            // A steady tone has no periodicity between 40 and 200 bpm, so its beat histogram is empty at every window;
            // the tone pulses at 120 bpm (2 Hz at 22.05 kHz), 0.5 (1 + cos) of an envelope on a 440 Hz carrier.
        {
            var genreEngine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            var genreOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            const int genreSamples = 262144;
            var pulse = genreEngine.TensorAddScalar(CosineRamp<T>(genreSamples, 2 * Math.PI * 2 / 22050, 0.5), genreOps.FromDouble(0.5));
            var pulsed = genreEngine.TensorMultiply(SineRamp<T>(genreSamples, 2 * Math.PI * 440 / 22050, 0.5), pulse);
            return OptionSensitivityHarness.Fingerprint(((AiDotNet.Interfaces.IGenreClassifier<T>)genre).ExtractFeatures(pulsed)
                .Select(genreOps.ToDouble));
        }
        if (network is AiDotNet.Audio.Speaker.SpeakerEmbeddingExtractor<T> or AiDotNet.Audio.Speaker.SpeakerVerifier<T>
            && option is "SampleRate" or "FftSize" or "HopLength" or "NormalizationWindowFrames")
            // The MFCC front end (Kaldi's 25 ms / 10 ms frames, 3-second sliding mean normalization) runs on a waveform;
            // the fixture's ready-made feature frames bypass it. Four seconds span more than the normalization window.
            return OptionSensitivityHarness.Fingerprint(network.Predict(Waveform<T>(64000))
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.SpeechRecognition.LLMIntegrated.SALM<T> && option == "NumAdapterHeads")
            // The modality adapter attends after the encoder's 8x and its own 4x subsampling: the fixture's 0.2 s are
            // one adapter frame, where every head count attends to itself. One second gives several frames.
            return observe(network, Waveform<T>(16000));
        if (network is AiDotNet.Video.Enhancement.TTVSR<T> && option == "KeyframeStride")
        {
            // Keyframes fall every KeyframeStride frames from the first; the fixture's two frames hold only frame 0
            // at every stride. Seven 16 x 16 frames place them at 0, 3, 6 against 0, 2, 4, 6 or 0, 4.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var clip = new Tensor<T>(new[] { 7, 3, 16, 16 });
            clip = Shaped(SineRamp<T>(clip.Length, 0.07, 100.0, 127.5), clip.Shape.ToArray());
            return observe(network, clip);
        }
        if (network is AiDotNet.Video.Enhancement.MGLDVSR<T> && option is "GuidanceScale" or "FlowIterations")
        {
            // Motion guidance steps the latents along the warping-error gradient of RAFT flows computed on the LQ clip at
            // a quarter of its size. The fixture's 8 x 8 frames give 2 x 2 flows, at which every pixel is occluded and the
            // gradient vanishes. Two 32 x 32 frames of a pattern shifted one pixel give RAFT a visible motion.
            if (option == "GuidanceScale")
            {
                // The untrained RAFT's random flows push every pixel's forward-backward check past the occlusion bound,
                // so the visibility mask sums to zero and the guidance step is empty at every scale. A zeroed flow-update
                // head makes RAFT predict zero flow, which leaves every pixel visible and the warping error nonzero.
                var bindingFlags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
                var raft = typeof(AiDotNet.Video.Enhancement.MGLDVSR<T>).GetField("_raft", bindingFlags)?.GetValue(network);
                if (raft?.GetType().GetField("_flow2", bindingFlags)?.GetValue(raft) is AiDotNet.Diffusion.VAE.LazyWeightModule<T> flowHead)
                    flowHead.SetParameters(new AiDotNet.Tensors.LinearAlgebra.Vector<T>(checked((int)flowHead.ParameterCount)));
                else
                    throw new InvalidOperationException("MGLDVSR's RAFT flow-update head was not found; the GuidanceScale probe cannot zero it.");
            }
            return observe(network, ShiftedClip<T>(2, 32));
        }
        if (network is AiDotNet.Video.Enhancement.StreamDiffVSR<T> streamDiff && option == "AttentionHeads")
        {
            // The heads act only in cross-attention to the frozen empty-prompt embedding. The fixture's embedding is
            // all zeros, so every key is the same and each head count attends uniformly; OpenCLIP's embedding of the
            // empty prompt differs per token. Give the probe a token-varying embedding.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var embedding = streamDiff.PromptEmbeddingLayer().GetParameters();
            var varying = new AiDotNet.Tensors.LinearAlgebra.Vector<T>(SineRamp<T>(embedding.Length, 0.37, 1.0).ToArray());
            streamDiff.PromptEmbeddingLayer().SetParameters(varying);
            return observe(network, input);
        }
        if (network is AiDotNet.Video.Enhancement.SeedVR<T> seedVr && option is "WindowRows" or "WindowColumns")
        {
            // SeedVR sizes its windows on the grid resized to 720p's 45 x 80 tokens (window.py 720pwin_by_size_bysize),
            // so the fixture's 2 x 2-token grid is one window at every count. A 30 x 30-token grid (60 x 60 latents)
            // resizes to 60 x 60, which three windows per axis cut into 20-token windows.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var latents = new Tensor<T>(new[] { 1, seedVr.LatentChannelCount, 1, 60, 60 });
            latents = Shaped(SineRamp<T>(latents.Length, 0.07, 1.0), latents.Shape.ToArray());
            return OptionSensitivityHarness.Fingerprint(seedVr.LatentVelocity(latents, 500).Select(ops.ToDouble));
        }
        if (option == "HubertHeads" && network switch
            {
                AiDotNet.TextToSpeech.MultiModal.SpeechGPT<T> speechGpt => speechGpt.Core.Units,
                AiDotNet.TextToSpeech.MultiModal.LlamaOmni<T> llamaOmni => llamaOmni.UnitExtractor,
                AiDotNet.SpeechRecognition.Specialized.VoxtLM<T> voxtLm => voxtLm.Network?.Units,
                _ => null,
            } is { } unitExtractor)
            // The frozen HuBERT tokenizer's heads act on the features its k-means quantizes; at fixture scale every head
            // count snaps a second of audio to the same few centroids, so observe the features themselves.
            return OptionSensitivityHarness.Fingerprint(unitExtractor.LayerFeatures(Waveform<T>(16000))
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.ComputerVision.Segmentation.Medical.MedicalSAM2<T> medicalSam
            && option is "MemoryBankCapacity" or "MemoryBankRule" or "MemoryResampling" or "ConfidenceThreshold")
        {
            // The self-sorting memory bank (Zhu et al. 2024, section 3.2) fills, filters and resamples while an image set
            // is segmented from one prompted template; Predict segments a single image with an empty bank. Eight
            // distinct 64-pixel images overfill the fixture's four-entry bank.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var images = new List<Tensor<T>>();
            for (int k = 0; k < 8; k++)
            {
                var image = Shaped(SineRamp<T>(3 * 64 * 64, 0.011 + 0.003 * k, 0.4, 0.5, k), new[] { 3, 64, 64 });
                images.Add(image);
            }
            var masks = medicalSam.SegmentImageSet(images, 0, new float[,] { { 32f, 32f } }, new[] { 1 });
            return OptionSensitivityHarness.Fingerprint(masks.SelectMany(mask => mask.Select(ops.ToDouble)));
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> fishText && option is "NormalizeChineseText" or "ChunkLength")
        {
            // Both act on the text Synthesize speaks (clean_text's number normalizer, split_text); Predict takes ids.
            var segments = fishText.SplitText(FishLongText);
            return OptionSensitivityHarness.Fingerprint(segments.SelectMany(segment => segment.Select(character => (double)character))
                .Concat(segments.Select(segment => (double)segment.Length)));
        }
        if (network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> fishHistory && option == "ContextLength")
            // The context budget trims the earlier segments' history in long-form synthesis; PrepareOptions shrinks it
            // to a few tokens and the chunk length to a few words so the text spans several segments.
            return OptionSensitivityHarness.Fingerprint(fishHistory.Synthesize(FishLongText)
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.CodecBased.IndexTTS<T> indexSilence
            && option is "SilentCode" or "MaxSilentCodes" or "KeptSilentCodes")
        {
            // remove_long_silence acts on generated codes; the fixture's eight generated codes never hold 30 silent ones.
            // Runs of 31 52s and 35 each of 53 and 51 sit on either side of every threshold the perturbations reach.
            var codes = new List<int> { 7 };
            codes.AddRange(Enumerable.Repeat(52, 31));
            codes.Add(8);
            codes.AddRange(Enumerable.Repeat(53, 35));
            codes.Add(9);
            codes.AddRange(Enumerable.Repeat(51, 35));
            codes.Add(10);
            return OptionSensitivityHarness.Fingerprint(indexSilence.RemoveLongSilence(codes).Select(code => (double)code));
        }
        if (network is AiDotNet.TextToSpeech.Latest.IndexTTS2<T> index2 && option is "MaxPromptSeconds" or "SegmentSilenceMilliseconds" or "EmotionMaxTokens")
        {
            // Each acts outside Predict's single segment: the prompt reader's cut (here a 20-second recording at 100 Hz),
            // the silence between long-form segments, and the emotion model's answer budget.
            if (option == "MaxPromptSeconds")
                return OptionSensitivityHarness.Fingerprint(index2.CutPrompt(
                    Enumerable.Range(0, 2000).Select(i => Math.Sin(i * .1)).ToArray(), 100));
            if (option == "SegmentSilenceMilliseconds")
                return OptionSensitivityHarness.Fingerprint(index2.JoinSegments(new List<double[]>
                {
                    Enumerable.Range(0, 64).Select(i => Math.Sin(i * .2)).ToArray(),
                    Enumerable.Range(0, 64).Select(i => Math.Cos(i * .3)).ToArray(),
                }));
            return OptionSensitivityHarness.Fingerprint(index2.EmotionAnswer("I am so happy today!").Select(character => (double)character));
        }
        if (network is AiDotNet.TextToSpeech.MultiModal.LlamaOmni<T> omni && option is "Temperature" or "TopP" or "MaxNewTokens" or "EncoderHeads")
        {
            // Decoding acts in Chat's text generation (Algorithm 1) and the Whisper encoder's heads in the speech
            // embeddings Chat feeds the LLM; Predict reads text ids aloud.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            if (option == "EncoderHeads")
                return OptionSensitivityHarness.Fingerprint(omni.SpeechEmbeddings(Waveform<T>(16000)).Select(ops.ToDouble));
            var response = omni.Chat(Waveform<T>(16000));
            return OptionSensitivityHarness.Fingerprint(response.Text.Select(character => (double)character)
                .Concat(response.Units.Select(unit => (double)unit)).Concat(response.Audio.Select(ops.ToDouble)));
        }
        if (network is AiDotNet.SpeechRecognition.AlibabaASR.Qwen3ASR<T> or AiDotNet.SpeechRecognition.AlibabaASR.Qwen3ASRSmall<T>
            && option == "WindowChunks")
            // The encoder attends within windows of WindowChunks one-second chunks; three seconds of audio span several
            // windows once the window is shorter than the clip.
            return observe(network, Waveform<T>(48000));
        if (network is AiDotNet.SpeechRecognition.AlibabaASR.SeACo<T> seacoFilter && option == "AttentionFilterTopK")
        {
            // The filter keeps the top-k hotwords by decoder attention only when the list is longer than k (section 2.3);
            // 120 hotwords exceed every k the probe tries (up to 100), and the bias decoder's probabilities show the kept set.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            // Without a vocabulary each character's code point is its token; the 16-token fixture vocabulary has eight
            // non-whitespace ids above the end token, so three-character words give 120 distinct hotwords.
            var ids = new[] { 3, 4, 5, 6, 7, 8, 14, 15 };
            var hotwords = Enumerable.Range(0, 120)
                .Select(i => new string(new[] { (char)ids[i % 8], (char)ids[i / 8 % 8], (char)ids[i / 64 % 8] })).ToList();
            var probabilities = seacoFilter.BiasProbabilities(Waveform<T>(16000), hotwords);
            return OptionSensitivityHarness.Fingerprint(probabilities is null ? Array.Empty<double>() : probabilities.Select(ops.ToDouble));
        }
        if (network is AiDotNet.TextToSpeech.EndToEnd.MeloTTS<T> melo && option == "BertHeads")
            // The BERT features condition the text encoder only when text is synthesized; Predict takes phone ids.
            return OptionSensitivityHarness.Fingerprint(melo.Synthesize("Hello there, how are you today?")
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.Audio.Classification.SceneClassifier<T> scene && option == "ContextFrames")
            // The fixture feeds ready-made context vectors; the option sizes the vectors a recording becomes.
            return OptionSensitivityHarness.Fingerprint(scene.ContextFeatures(LongWaveform<T>())
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.MultiModal.Moshi<T> moshi)
            return MoshiOptionFixture.Observe(moshi, option);
        if (network is AiDotNet.TextToSpeech.Classic.PortaSpeech<T> porta && option is "PostNetLearningRate" or "PostNetTemperature")
            // The author trains and samples the flow post-net only in the second phase.
            porta.CurrentPhase = AiDotNet.TextToSpeech.Classic.PortaSpeechTrainingPhase.PostNet;
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
                mel = Shaped(SineRamp<T>(mel.Length, 0.13, 0.2, -5.0), mel.Shape.ToArray());
                var tokens = IndexRamp<T>(4, start: 1);
                var objective = voicebox.EvaluateTrainingObjective(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
                {
                    Tokens = tokens, Mel = mel, Durations = new[] { 4, 4, 4, 4 },
                });
                return OptionSensitivityHarness.Fingerprint(new[] { numeric.ToDouble(objective) });
            }
        }

        if (network is AiDotNet.ComputerVision.Segmentation.Video.EfficientTAM<T> tam && option == "MemoryCrossAttention")
        {
            // The efficient memory cross-attention (Xiong et al. 2024, section 3.3) pools the memory bank of a tracked
            // video; Predict segments single images, which attend to no memory. Track one object from a square mask on
            // frame 0 through three more frames of a moving pattern, so later frames attend to two memories.
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            var tamOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            const int side = 64;
            var clip = engine.TensorMultiplyScalar(ShiftedClip<T>(4, side), tamOps.FromDouble(1.0 / 255));
            Tensor<T> Frame(int f) => engine.Reshape(engine.TensorSlice(clip, new[] { f, 0, 0, 0 }, new[] { 1, 3, side, side }), new[] { 3, side, side });
            var band = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { side / 4 }), engine.TensorLinspace(tamOps.One, tamOps.One, side / 2),
                new Tensor<T>(new[] { side / 4 }) }, 0);
            var square = engine.Reshape(engine.TensorOuter(band, band), new[] { 1, side, side });
            tam.InitializeTracking(Frame(0), square, new[] { 1 });
            var tracked = new List<double>();
            for (int f = 1; f < 4; f++) tracked.AddRange(tam.PropagateToFrame(Frame(f)).Masks.Select(tamOps.ToDouble));
            return OptionSensitivityHarness.Fingerprint(tracked);
        }

        if (network is AiDotNet.VisionLanguage.Grounding.GroundedSAM2<T> groundedSam && option is "MaskThreshold" or "SamDecoderHeads")
            return AiDotNet.Tests.VisionLanguage.GroundedSam2OptionFixture.Observe(groundedSam, input);

        if (network is AiDotNet.TextToSpeech.CodecBased.CosyVoice<T> cosyTokenizer && option == "SpeechTokenizerHeads")
            return CosyVoiceTokenizerOptionFixture.Observe(cosyTokenizer);

        if (network is AiDotNet.TextToSpeech.FlowDiffusion.DiTToTTS<T> ditto && option is "TextPadId" or "NumHeads")
            return option == "NumHeads" ? DiTToOptionFixture.ObserveHeads(ditto) : DiTToOptionFixture.ObserveTextPadding(ditto);

        if (network is AiDotNet.TextToSpeech.CodecBased.Bark<T> bark && BarkOptionFixture.Handles(option))
            return BarkOptionFixture.Observe(bark, option);

        if (network is AiDotNet.TextToSpeech.CodecBased.Vevo<T> vevo)
            return VevoOptionFixture.Observe(vevo, option);
        if (network is AiDotNet.TextToSpeech.CodecBased.Llasa<T> llasa && option == "IncludePromptAudio")
            return LlasaOptionFixture.ObservePrompt(llasa);
        if (network is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech2<T> naturalSpeech2 && NaturalSpeech2OptionFixture.Handles(option))
            return NaturalSpeech2OptionFixture.Observe(naturalSpeech2, option);
        if (network is AiDotNet.TextToSpeech.MultiModal.StepAudio<T> stepAudio
            && (StepAudioOptionFixture.IsChatSamplingOption(option) || StepAudioOptionFixture.IsPpoOption(option)))
            return StepAudioOptionFixture.ObserveChat(stepAudio);
        if (network is AiDotNet.TextToSpeech.MultiModal.StepAudio<T> tokenizer && option == "LinguisticDither")
            return StepAudioOptionFixture.ObserveLinguisticDither(tokenizer);
        if (network is AiDotNet.TextToSpeech.CodecBased.VALLE<T> vallE && option == "PromptSeconds")
            return VallEOptionFixture.ObserveTrainingPrompt(vallE);
        if (network is AiDotNet.TextToSpeech.CodecBased.VALLE<T> endCondition && option == "StopOnMostLikelyEndToken")
            return VallEOptionFixture.ObserveEndCondition(endCondition);
        if (network is AiDotNet.TextToSpeech.Classic.NonAttentiveTacotron<T> nonAttentive
            && NonAttentiveTacotronOptionFixture.UsesTraining(option))
            return NonAttentiveTacotronOptionFixture.Observe(nonAttentive);
        if (network is AiDotNet.TextToSpeech.TtsModelBase<T> megaSpeech
            && megaSpeech is AiDotNet.TextToSpeech.Latest.MegaTTS<T> or AiDotNet.TextToSpeech.Latest.MegaTTS2<T>)
        {
            if (option == "Language") return MegaTtsOptionFixture.ObserveTokens(megaSpeech);
            if (MegaTtsOptionFixture.HandlesVocoder(option)) return MegaTtsOptionFixture.ObserveWaveform(megaSpeech);
            if (option == "ProsodyInterpolation" && megaSpeech is AiDotNet.TextToSpeech.Latest.MegaTTS2<T> styleModel)
                return MegaTtsOptionFixture.ObserveProsodyInterpolation(styleModel);
            if (MegaTtsOptionFixture.HandlesAcoustic(option) || MegaTtsOptionFixture.HandlesLanguageModels(option))
                return MegaTtsOptionFixture.ObserveObjective(megaSpeech, MegaTtsOptionFixture.HandlesLanguageModels(option));
        }

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
            var slots = Shaped(IndexRamp<T>(7, start: 1), new[] { 1, 7 });
            var codes = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorConcatenate(new[] { slots, slots, AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAddScalar(slots, ops.One) }, 0);
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
        if (network is AiDotNet.VisionLanguage.Grounding.GroundingDINO<T> groundingDino && option == "TextThreshold")
        {
            // TextThreshold picks the caption words that label a detection, inside Ground; Predict returns the image
            // tokens and never thresholds. PrepareOptions drops the box threshold so every query is a detection.
            var detections = groundingDino.Ground(input, "cat . remote control .");
            return OptionSensitivityHarness.Combine(fingerprint,
                OptionSensitivityHarness.Fingerprint(detections.SelectMany(d => d.Phrase.Select(c => (double)c).Append(-1.0))));
        }
        if (network is AiDotNet.SpeechRecognition.ConformerFamily.ConvTransformer<T> or AiDotNet.SpeechRecognition.ConformerFamily.EBranchformer<T>
                or AiDotNet.SpeechRecognition.ConformerFamily.Branchformer<T>
            && network is AiDotNet.Interfaces.ISpeechRecognizer<T> convTransformer && option == "MaxTextLength")
        {
            // MaxTextLength bounds the greedy attention decode inside Transcribe; Predict returns encoder logits and
            // never decodes. An untrained decoder can emit the end token first on a given clip, so the probe decodes four
            // clips of different tones; any that runs past one token shows the bound.
            var transcripts = new[] { 0.13, 0.29, 0.53, 0.87 }.Select(frequency =>
                convTransformer.Transcribe(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(
                    SineRamp<T>(16000, frequency, 0.4), CosineRamp<T>(16000, 0.37, 0.2)), includeTimestamps: false).Text).ToList();
            return OptionSensitivityHarness.Combine(fingerprint, OptionSensitivityHarness.Fingerprint(
                transcripts.SelectMany(text => text.Select(character => (double)character).Append(-text.Length - 1.0))));
        }
        if (network is AiDotNet.SpeechRecognition.CTCVariants.CIFDecoder<T> cif && option == "TailThreshold")
        {
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            foreach (int samples in new[] { 1600, 3200, 6400, 16000 })
            {
                var waveform = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(samples, 0.13, 0.4), CosineRamp<T>(samples, 0.37, 0.2));
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
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            var semantic = IndexRamp<T>(8, modulus: settings.SemanticCodes);
            // codes[frame, q] = (frame + q) mod CodebookSize: the outer sum of the two index ramps.
            int books = settings.NumCodebooks;
            var frameIndex = engine.TensorOuter(IndexRamp<T>(16), engine.TensorLinspace(ops.One, ops.One, books));
            var bookIndex = engine.TensorOuter(engine.TensorLinspace(ops.One, ops.One, 16), IndexRamp<T>(books));
            var codebookSize = ops.FromDouble(settings.CodebookSize);
            var codes = engine.TensorRemainder(engine.TensorAdd(frameIndex, bookIndex),
                Shaped(engine.TensorLinspace(codebookSize, codebookSize, 16 * books), new[] { 16, books }));
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
        if (network is AiDotNet.TextToSpeech.Classic.Tacotron<T> griffin
            && option is "PreEmphasis" or "GriffinLimIterations" or "GriffinLimPower")
            // These act only where the linear spectrogram becomes audio: Griffin-Lim, then de-emphasis.
            return OptionSensitivityHarness.Fingerprint(griffin.SynthesizeWaveform("ab")
                .Select(AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble));
        if (network is AiDotNet.TextToSpeech.DescriptionBased.PromptTTS<T> prompted
            && option is "StyleLearningRate" or "StyleTrainingSteps" or "SamplingSeed" or "PrefixHiddenSize")
        {
            // These drive the style encoder's own stage (TrainStyle), which the generic step does not run.
            var settings = (AiDotNet.TextToSpeech.DescriptionBased.PromptTTSOptions)prompted.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            // 2 + (7 i) mod (StyleVocabularySize - 2), past the two special ids.
            var styleRange = ops.FromDouble(Math.Max(1, settings.StyleVocabularySize - 2));
            var promptIds = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAddScalar(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorRemainder(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorMultiplyScalar(IndexRamp<T>(5), ops.FromDouble(7)),
                AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorLinspace(styleRange, styleRange, 5)), ops.FromDouble(2));
            var labels = settings.StyleFactorClasses.Select((classes, i) => i % classes).ToArray();
            var losses = new List<double>();
            for (int step = 0; step < 3; step++) losses.Add(ops.ToDouble(prompted.TrainStyle(promptIds, labels)));
            losses.Add(ops.ToDouble(prompted.EvaluateStyleLoss(promptIds, labels)));
            return OptionSensitivityHarness.Fingerprint(losses);
        }
        if (network is AiDotNet.Audio.Foundations.W2vBert<T> w2v && option is "SemanticLayer" or "SemanticClusters")
        {
            // The layer and cluster count shape AudioLM's semantic tokenizer, fitted here on three synthetic voices.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            Tensor<T> Voice(double hz)
            {
                return AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(8000, 2 * Math.PI * hz / 16000, 0.5), SineRamp<T>(8000, 2 * Math.PI * 3.1 * hz / 16000, 0.2));
            }
            var recordings = new[] { Voice(120), Voice(220), Voice(320) };
            w2v.FitSemanticTokenizer(recordings);
            return OptionSensitivityHarness.Fingerprint(recordings.SelectMany(r => w2v.SemanticTokens(r)).Select(t => (double)t));
        }
        if (network is AiDotNet.Audio.Foundations.HuBERT<T> hubert && option == "MaskTimeProbability")
        {
            // At the fixture's length a span covers every frame whatever the rate; draw the mask over 500 frames.
            var model = hubert.Network ?? throw new InvalidOperationException("native model");
            var drawn = model.DrawMask(1, 500, model.Configuration.MaskTimeProbability, new Random(1337));
            return OptionSensitivityHarness.Fingerprint(Enumerable.Range(0, 500).Select(t => drawn[0, t] ? 1.0 : 0.0));
        }
        if (network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> topK && option == "GenerationTopK")
        {
            // Top-k filters the T3 speech-token draw; the fingerprint is the seeded generation itself.
            var settings = (AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions)topK.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var speaker = new Tensor<T>(new[] { settings.T3SpeakerEmbeddingDim });
            speaker = Shaped(CosineRamp<T>(speaker.Length, 0.5, 1.0), speaker.Shape.ToArray());
            var condition = new AiDotNet.TextToSpeech.VoiceCloning.ChatterboxT3Condition<T>(speaker, new[] { 2, 4, 6 }, 0.5);
            var tokens = new List<double>();
            for (int seed = 0; seed < 4; seed++)
                tokens.AddRange(topK.T3Model!.Generate(new[] { 1, 3, 5, 7 }, condition, new Random(1337 + seed)).Select(t => (double)t));
            return OptionSensitivityHarness.Fingerprint(tokens);
        }
        if (network is AiDotNet.TextToSpeech.VoiceCloning.Chatterbox<T> distilled && option == "DistillationTeacherSteps")
        {
            // Prepare makes the fixture Turbo; the teacher's Euler steps set the student's regression target.
            var flow = distilled.FlowModel!;
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var speech = Enumerable.Range(0, 6).Select(i => (i * 37) % 50).ToArray();
            var mel = new Tensor<T>(new[] { 1, flow.Configuration.OutputSize, flow.MelFrames(speech.Length) });
            mel = Shaped(SineRamp<T>(mel.Length, 0.37, 1.0), mel.Shape.ToArray());
            var xvector = new Tensor<T>(new[] { flow.Configuration.SpeakerDim });
            xvector = Shaped(CosineRamp<T>(xvector.Length, 0.5, 1.0), xvector.Shape.ToArray());
            var settings = (AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions)distilled.GetOptions();
            return OptionSensitivityHarness.Fingerprint(flow.DistillationLoss(distilled.TeacherFlow, speech, mel, xvector, false,
                new Random(1337), settings.DistillationTeacherSteps).Select(ops.ToDouble));
        }
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
            mel = Shaped(SineRamp<T>(mel.Length, 0.17, 0.2), mel.Shape.ToArray());
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
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var loud = ops.FromDouble(0.075);
            var waveform = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorConcatenate(new[] { AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorLinspace(loud, loud, 9), new Tensor<T>(new[] { 7 }) }, 0);
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
        if (network is AiDotNet.TextToSpeech.MultiModal.Moshi<T> moshi)
        {
            // Predict consumes SentencePiece IDs. Numeric ramp inputs can decode to repeated
            // unknown pieces and spend the entire streaming context on an invalid utterance.
            var ids = moshi.Core.Tokenizer.Encode("the cat sat").ToArray();
            var moshiOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(id => moshiOps.FromDouble(id)).ToArray()));
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.Vocos<T> vocos && !IsVocoderFrontendOption(option))
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.VocosOptions)vocos.GetOptions();
            return MelFrames<T>(settings.MelChannels, 3);
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.WaveNet<T> waveNet && !IsVocoderFrontendOption(option))
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.WaveNetOptions)waveNet.GetOptions();
            return MelFrames<T>(settings.MelChannels, 1);
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.WaveRNN<T> waveRnn && !IsVocoderFrontendOption(option))
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.WaveRNNOptions)waveRnn.GetOptions();
            return MelFrames<T>(settings.MelChannels, 2);
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.DiffWave<T> && option is "NumClasses" or "UtteranceSamples")
            return new Tensor<T>(new[] { 1 }); // Label zero, or the ignored unconditional input.
        if (network is AiDotNet.Audio.Speaker.CAMPlusPlus<T> cam && option is "FrameShiftMilliseconds" or "FrameLengthMilliseconds")
        {
            var settings = (AiDotNet.Audio.Speaker.CAMPlusPlusOptions)cam.GetOptions();
            var camWaveform = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(settings.SampleRate, 0.13, 0.4), CosineRamp<T>(settings.SampleRate, 0.37, 0.2));
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
        // Recognizers live under AiDotNet.SpeechRecognition and, for the classic ones (Conformer), AiDotNet.Audio.SpeechRecognition.
        if (option != "MaxAudioLengthSeconds" || input.Length == 0
            || network.GetType().Namespace is not { } recognizerNamespace
            || !(recognizerNamespace.StartsWith("AiDotNet.SpeechRecognition", StringComparison.Ordinal)
                || recognizerNamespace.StartsWith("AiDotNet.Audio.SpeechRecognition", StringComparison.Ordinal)))
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
        // One second of the input repeated, then the same second negated.
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var flat = engine.Reshape(input, new[] { input.Length });
        var tiled = engine.TensorTile(flat, new[] { (sampleRate + input.Length - 1) / input.Length });
        var second = engine.TensorSlice(tiled, new[] { 0 }, new[] { sampleRate });
        var waveform = engine.TensorConcatenate(new[] { second, engine.TensorNegate(second) }, 0);
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
        "MelMinFrequency" or "MelMaxFrequency" or "WindowSize" or "MelMean" or "MelScale" or "EnergyMin" or "EnergyMax";

    private static Tensor<T> VocoderRecording<T>(AiDotNet.TextToSpeech.Vocoders.VocoderOptions options)
    {
        int length = Math.Max(2 * options.FftSize, 8 * options.HopSize);
        return AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(length, 0.13, 0.4), CosineRamp<T>(length, 0.37, 0.2));
    }

    internal static int TrainingSteps<T>(INeuralNetworkModel<T> network, string option, int steps)
        => AiDotNet.Tests.VisionLanguage.NormalizationOptionFixture.Handles(network, option) ? 0
        : network is AiDotNet.TextToSpeech.MultiModal.Moshi<T> && MoshiOptionFixture.IsSampling(option) ? 0
        : network is AiDotNet.TextToSpeech.CodecBased.Vevo<T> && VevoOptionFixture.IsSampling(option) ? 0
        : network is AiDotNet.TextToSpeech.CodecBased.Llasa<T> && option == "IncludePromptAudio" ? 0
        : network is AiDotNet.TextToSpeech.CodecBased.Llasa<T> && LlasaOptionFixture.IsScheduler(option) ? Math.Max(4, steps)
        : network is AiDotNet.TextToSpeech.Vocoders.Vocos<T> && option == "CosineSteps" ? Math.Max(4, steps)
        : network is AiDotNet.TextToSpeech.Vocoders.WaveNet<T> && option == "LearningRateHalvingSteps" ? Math.Max(4, steps)
        : network is AiDotNet.TextToSpeech.Vocoders.HiFiGAN<T> && option is "LearningRateDecay" or "UpdatesPerEpoch" ? Math.Max(4, steps)
        : network is AiDotNet.TextToSpeech.Vocoders.ParallelWaveGAN<T> && option == "Epsilon" ? Math.Max(6, steps)
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
        if (network is AiDotNet.TextToSpeech.StyleEmotion.StyleTTS<T> styleTts && option is "MaxStretch" or "DropoutRate")
            // Section II-B's second stage trains the prosody predictor, with the duration-stretch augmentation and the
            // predictor's dropout; the first (reconstruction) stage uses neither.
            styleTts.CurrentStage = AiDotNet.TextToSpeech.StyleEmotion.StyleTtsTrainingStage.Prosody;
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
        var recording = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(samples, 2 * Math.PI * 180 / options.SampleRate, 0.4),
            SineRamp<T>(samples, 2 * Math.PI * 470 / options.SampleRate, 0.2));
        natural.InitializeMemoryBank(new[] { recording });
    }

    internal static bool TryTrainOption<T>(INeuralNetworkModel<T> network, Tensor<T> input, string option)
    {
        if (network is AiDotNet.Audio.LanguageIdentification.Wav2Vec2LanguageIdentifier<T> lid && option == "FeatureProjectionDropout")
        {
            // The projection dropout acts before SpecAugment's time masks (10-frame spans, at least two). The fixture's
            // 2,048 samples give about six frames, every one masked, so nothing before the mask reaches the loss. Two
            // seconds give about a hundred frames, most of them unmasked.
            var lidOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var english = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorConcatenate(new[] { new Tensor<T>(new[] { 1 }, new AiDotNet.Tensors.LinearAlgebra.Vector<T>(new[] { lidOps.One })), new Tensor<T>(new[] { 3 }) }, 0);
            lid.Train(Waveform<T>(32000), english);
            return true;
        }
        if (network is AiDotNet.Finance.Forecasting.Foundation.Chronos<T> chronos && option is "BatchSize" or "MinPast" or "MissingValueProbability")
        {
            // The batch, the earliest split point and the missing-value augmentation shape the windows the paper's
            // recipe samples from a long series; a supervised step on a ready-made window has none of them.
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            var series = new Tensor<T>(new[] { 240 });
            series = Shaped(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(series.Length, 0.21, 1.0, 10.0), CosineRamp<T>(series.Length, 0.05, 0.3)), series.Shape.ToArray());
            chronos.TrainOnSeries(series);
            return true;
        }
        if (network is AiDotNet.SpeechRecognition.Robust.RobustConformer<T> robust && option is "DiversityWeight" or "ReconstructionWeight")
        {
            // λ1 and λ2 weight the continual pre-training loss (Equation 4), which supervised fine-tuning does not use.
            var clean = Waveform<T>(32000);
            var noisy = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(clean,
                AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Reshape(SineRamp<T>(clean.Length, 1.7, 0.05), clean.Shape.ToArray()));
            robust.ContinualPretrain(noisy, clean);
            robust.ContinualPretrain(noisy, clean);
            return true;
        }
        if (network is AiDotNet.SpeechRecognition.Robust.AVHuBERT<T> avHubert
            && avHubert.GetOptions() is AiDotNet.SpeechRecognition.Robust.AVHuBERTOptions avOptions
            && option is "LipSize" or "LogitTemperature" or "AudioMaskLength" or "AudioMaskProbability" or "VideoMaskLength"
                or "VideoMaskProbability" or "ModalityDropout" or "AudioDropout" or "FeaturePenaltyWeight")
        {
            // The masks, modality dropout, cluster logits and feature penalty act in masked multimodal cluster
            // prediction (section 3), on half a second of audio and its 12 mouth crops of LipSize at 25 Hz.
            var audio = Waveform<T>(8000);
            int side = avOptions.LipSize;
            var lips = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Reshape(SineRamp<T>(12 * side * side, 0.37, 0.4, 0.5), new[] { 12, side, side });
            var targets = Enumerable.Range(0, 12).Select(t => (t / 2) % avOptions.NumClusters).ToArray();
            avHubert.Pretrain(audio, lips, targets);
            avHubert.Pretrain(audio, lips, targets);
            return true;
        }
        if (network is AiDotNet.Video.FrameInterpolation.VFIformer<T> vfiformer && option == "FlowWeight" && input.Rank == 3 && input.Shape[0] == 6)
        {
            // The flow distillation term needs the midpoint's flows to both frames, four channels after the frame pair.
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            int h = input.Shape[1], w = input.Shape[2];
            var flows = engine.Reshape(SineRamp<T>(4 * h * w, 0.11, 1.5), new[] { 4, h, w });
            var withFlows = engine.TensorConcatenate(new[] { input, flows }, 0);
            var target = NearbyTarget(vfiformer.Predict(input), 5.0);
            vfiformer.Train(withFlows, target);
            vfiformer.Train(withFlows, target);
            return true;
        }
        if (network is AiDotNet.Video.Enhancement.PSRT<T> psrtFreeze && option == "FlowFreezeIterations")
        {
            // SPyNet joins training after fix_flow iterations; its pyramid needs frames wider than the fixture's 8 x 8
            // for a flow (and so a flow gradient) to exist. Three 32 x 32 frames of a pattern shifted one pixel a frame.
            var clip = ShiftedClip<T>(3, 32);
            var target = NearbyTarget(psrtFreeze.Predict(clip), 5.0);
            psrtFreeze.Train(clip, target);
            psrtFreeze.Train(clip, target);
            return true;
        }
        if (network is AiDotNet.Video.FrameInterpolation.XVFI<T> xvfi && option is "LearningRateDecay" or "SmoothnessWeight")
        {
            // MultiStepLR steps per epoch, so the decay applies from the step after the first epoch ends. The refined
            // flow starts at zero, where the smoothness term has no gradient, so the weight shows once a step has moved it.
            var target = NearbyTarget(xvfi.Predict(input), 0.05);
            for (int epoch = 0; epoch < 3; epoch++)
            {
                xvfi.Train(input, target);
                xvfi.EndEpoch();
            }
            return true;
        }
        if (network is AiDotNet.Video.FrameInterpolation.FLAVR<T> flavr && option is "PlateauFactor" or "PlateauPatience")
        {
            // ReduceLROnPlateau steps once per epoch on the validation loss. Eight epochs of a plateaued loss cross every
            // patience the probe tries, with a training step after the drop.
            var target = NearbyTarget(flavr.Predict(input), 0.05);
            for (int epoch = 0; epoch < 8; epoch++)
            {
                flavr.Train(input, target);
                flavr.EndEpoch(1.0);
            }
            return true;
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.WaveRNN<T> waveRnn && IsWaveRnnTrainingOption(option))
        {
            var audio = new Tensor<T>(new[] { 4096 });
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            audio = Shaped(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(audio.Length, 0.17, 0.3), CosineRamp<T>(audio.Length, 0.31, 0.1)), audio.Shape.ToArray());
            waveRnn.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = new Tensor<T>(new[] { 1 }), Audio = audio });
            return true;
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.DiffWave<T> diffWave && option is "NumClasses" or "UtteranceSamples")
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.DiffWaveOptions)diffWave.GetOptions();
            var audio = new Tensor<T>(new[] { 128 });
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            audio = Shaped(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(audio.Length, 0.17, 0.3), CosineRamp<T>(audio.Length, 0.31, 0.1)), audio.Shape.ToArray());
            diffWave.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            {
                Tokens = new Tensor<T>(new[] { 1 }), Audio = audio,
                ClassLabel = settings.Conditioner == AiDotNet.TextToSpeech.Vocoders.DiffWaveConditioner.ClassLabel ? 0 : null,
            });
            return true;
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.ParallelWaveGAN<T> parallel && option == "SegmentSize")
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.ParallelWaveGANOptions)parallel.GetOptions();
            parallel.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            { Tokens = new Tensor<T>(new[] { 1 }), Audio = VocoderRecording<T>(settings) });
            return true;
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.MelGAN<T> melGan && option is "SegmentSize" or "SamplingSeed")
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.MelGANOptions)melGan.GetOptions();
            melGan.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            { Tokens = new Tensor<T>(new[] { 1 }), Audio = VocoderRecording<T>(settings) });
            return true;
        }
        if (network is AiDotNet.TextToSpeech.Vocoders.UnivNet<T> univNet && option == "SegmentSize")
        {
            var settings = (AiDotNet.TextToSpeech.Vocoders.UnivNetOptions)univNet.GetOptions();
            univNet.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T>
            { Tokens = new Tensor<T>(new[] { 1 }), Audio = VocoderRecording<T>(settings) });
            return true;
        }
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
            mel = Shaped(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(mel.Length, 0.17, 0.4), CosineRamp<T>(mel.Length, 0.31, 0.2)), mel.Shape.ToArray());
            flow.Train(input, mel);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.EndToEnd.VITS2<T> alignment && option == "AlignmentNoiseDecay")
        {
            var settings = (AiDotNet.TextToSpeech.EndToEnd.VITS2Options)alignment.GetOptions();
            var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
            // The generic minimum-length recording can give each token exactly one frame:
            // MAS then has only one valid path, regardless of the actual noise schedule.
            var tokens = IndexRamp<T>(4, start: 1, modulus: settings.VocabSize);
            int hop = settings.UpsampleRates.Aggregate(1, (a, b) => a * b);
            var audio = new Tensor<T>(new[] { hop * 4 * tokens.Length });
            audio = Shaped(AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorAdd(SineRamp<T>(audio.Length, 0.17, 0.3), SineRamp<T>(audio.Length, 0.31, 0.1)), audio.Shape.ToArray());
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
        if (network is AiDotNet.TextToSpeech.CodecBased.FishSpeechBase<T> fishPreference && option is "DpoBeta" or "ReferenceFreeDpoBeta")
        {
            // Preference tuning (the standard DPO and the release's reference-free objective) on a pair of recordings of
            // the same text; the probe's supervised step never reads either beta. The fixture's 1,024-sample recordings
            // (16 code frames) keep the packed pair inside its 160-token context, which 1.5's system turn shares.
            var rejectedAudio = SineRamp<T>(1024, 0.05, 0.3);
            var chosen = new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, Audio = Waveform<T>(1024) };
            var rejected = new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, Audio = rejectedAudio };
            if (option == "DpoBeta") fishPreference.TrainPreference(chosen, rejected);
            else fishPreference.TrainPreferenceReferenceFree(chosen, rejected);
            return true;
        }
        if (network is AiDotNet.TextToSpeech.MultiModal.SpeechGPT<T> speechGptVocoder
            && option is "VocoderLearningRate" or "VocoderLearningRateDecay" or "VocoderSegmentSize" or "VocoderUpdatesPerEpoch")
        {
            // The unit vocoder trains apart from the language model (section 4.1; speech-resynthesis' train.py).
            speechGptVocoder.TrainVocoder(Waveform<T>(16000));
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
                var semantic = SineRamp<T>(width, 0.37, 1.0);
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
            recording = Shaped(SineRamp<T>(recording.Length, 0.13, 0.4), recording.Shape.ToArray());
            e3.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<T> { Tokens = input, Text = "test", Audio = recording });
            return true;
        }
        if ((network is AiDotNet.TextToSpeech.Vocoders.DiffusionVocoderBase<T> && option == "CropFrames"
                || network is AiDotNet.TextToSpeech.Vocoders.Vocos<T> && IsVocosRecordingOption(option)
                || network is AiDotNet.TextToSpeech.Vocoders.WaveNet<T> && IsWaveNetTrainingOption(option)
                || network is AiDotNet.TextToSpeech.Vocoders.HiFiGAN<T> && IsHiFiGanTrainingOption(option)
                || network is AiDotNet.TextToSpeech.Vocoders.MultiBandMelGAN<T> && option is "SegmentSize" or "SamplingSeed"
                || network is AiDotNet.TextToSpeech.Vocoders.BigVGAN<T> && option == "SegmentSize"
                || network is AiDotNet.TextToSpeech.Vocoders.ISTFTNet<T> && option == "SegmentSize")
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

    private static bool IsVocosRecordingOption(string option) => option is "SegmentSize" or "SamplingSeed"
        or "LearningRate" or "WeightDecay" or "Beta1" or "Beta2" or "CosineSteps";

    private static bool IsWaveNetTrainingOption(string option) => option is "SegmentSamples" or "LearningRate"
        or "WeightDecay" or "LearningRateHalvingSteps";

    private static bool IsHiFiGanTrainingOption(string option) => option is "SegmentSize" or "LearningRate"
        or "WeightDecay" or "LearningRateDecay" or "UpdatesPerEpoch";

    private static bool IsWaveRnnSparseOption(string option) => option is "SparsityTarget"
        or "PruningSteps" or "PruningStartStep" or "PruningInterval" or "PruneBlockRows" or "PruneBlockColumns";

    private static bool IsWaveRnnTrainingOption(string option) => IsWaveRnnSparseOption(option)
        || option is "SequenceSamples" or "LearningRate" or "WeightDecay" or "MaxGradientNorm";

    /// <summary>After <paramref name="option"/> takes a perturbed value: options the model requires to equal it follow, so
    /// the perturbation tests the pair rather than the constructor's rejection of a mismatch.</summary>
    internal static void Reconcile(AiDotNet.Models.Options.ModelOptions options, string option)
    {
        if (options is AiDotNet.TextToSpeech.Latest.MegaTTSOptions or AiDotNet.TextToSpeech.Latest.MegaTTS2Options)
            MegaTtsOptionFixture.Reconcile(options, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech2Options naturalSpeech2)
            NaturalSpeech2OptionFixture.Reconcile(naturalSpeech2, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.VevoOptions vevoOptions)
            VevoOptionFixture.Reconcile(vevoOptions, option);
        // Parler-TTS's frames, codes and codebooks are its codec's: the codec follows (a hop of h is one stride of h).
        if (options is AiDotNet.TextToSpeech.DescriptionBased.ParlerTTSOptions parler
            && option is "SampleRate" or "HopSize" or "CodebookSize" or "NumCodebooks")
        {
            var codec = (AiDotNet.Audio.Effects.DACOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(parler.Codec);
            codec.SampleRate = parler.SampleRate;
            codec.CodebookSize = parler.CodebookSize;
            codec.NumQuantizers = parler.NumCodebooks;
            codec.TargetBandwidthKbps = 1e6;                // every quantizer, so the count is NumQuantizers
            if (option == "HopSize" && parler.HopSize > 0) { codec.EncoderRates = [parler.HopSize]; codec.DecoderRates = [parler.HopSize]; }
            parler.Codec = codec;
        }
        // medGAN's generator adds each layer's output to its input, so the prior, every generator width and the
        // autoencoder's code share one width (Choi et al. 2017, section 3.2).
        if (IsGenericOptions(options, typeof(AiDotNet.Models.Options.MedGANOptions<>)))
        {
            var type = options.GetType();
            int embedding = (int)(type.GetProperty("EmbeddingDimension")?.GetValue(options) ?? 0);
            int? noise = (int?)type.GetProperty("NoiseDimension")?.GetValue(options);
            int width = option == "NoiseDimension" && noise is int n ? n : embedding;
            if (option is "EmbeddingDimension" or "NoiseDimension")
            {
                SetOption(options, "EmbeddingDimension", width);
                if (noise is not null) SetOption(options, "NoiseDimension", (int?)width);
                var generator = (int[]?)type.GetProperty("GeneratorDimensions")?.GetValue(options) ?? Array.Empty<int>();
                SetOption(options, "GeneratorDimensions", generator.Select(_ => width).ToArray());
            }
        }
        // PromptTTS's style and content representations share one width (§4.1).
        if (options is AiDotNet.TextToSpeech.DescriptionBased.PromptTTSOptions prompt)
        {
            if (option == "HiddenDim") prompt.EncoderDim = prompt.HiddenDim;
            else if (option == "EncoderDim") prompt.HiddenDim = prompt.EncoderDim;
        }
    }

    private static bool IsGenericOptions(AiDotNet.Models.Options.ModelOptions options, Type definition)
        => options.GetType().IsGenericType && options.GetType().GetGenericTypeDefinition() == definition;

    private static void SetOption(AiDotNet.Models.Options.ModelOptions options, string name, object? value)
        => options.GetType().GetProperty(name)?.SetValue(options, value);

    internal static void PrepareOptions(AiDotNet.Models.Options.ModelOptions options, string option)
    {
        // Step schedules whose first switch is hundreds of thousands of batches out: bring the switch inside the
        // probe's training steps so the scheduled rate is the one a step actually uses.
        if (options is AiDotNet.VisionLanguage.Grounding.GroundingDINOOptions grounding && option == "TextThreshold")
            grounding.ConfidenceThreshold = 0.0;
        if (options is AiDotNet.Video.Options.VFITOptions vfit && option == "LearningRateSteps")
            vfit.LearningRateMilestones = new[] { 1, 2, 3, 4, 5 };
        if (options is AiDotNet.Video.Options.XVFIOptions xvfi && option == "LearningRateDecay")
            xvfi.LearningRateMilestones = new[] { 1 };
        if (options is AiDotNet.Video.Options.InterpAnyClearerOptions clearer && option == "MinLearningRate")
        {
            // The cosine floor acts only after the 2,000-step warmup; a four-step cosine starts decaying at once.
            clearer.WarmupSteps = 0;
            clearer.TotalSteps = 4;
        }
        if (options is AiDotNet.TextToSpeech.Latest.MegaTTSOptions or AiDotNet.TextToSpeech.Latest.MegaTTS2Options)
            MegaTtsOptionFixture.Prepare(options, option);
        if (options is AiDotNet.TextToSpeech.Classic.NonAttentiveTacotronOptions nonAttentive)
            NonAttentiveTacotronOptionFixture.Prepare(nonAttentive, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.VALLEOptions vallE && option == "PromptSeconds")
            vallE.PromptSeconds = 2.0 * vallE.HopSize / vallE.SampleRate;
        if (options is AiDotNet.TextToSpeech.MultiModal.StepAudioOptions stepAudio && StepAudioOptionFixture.IsPpoOption(option))
            StepAudioOptionFixture.PreparePpoOptions(stepAudio, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.NaturalSpeech2Options naturalSpeech2)
            NaturalSpeech2OptionFixture.Prepare(naturalSpeech2, option);
        if (options is AiDotNet.TextToSpeech.CodecBased.LlasaOptions llasa)
            LlasaOptionFixture.Prepare(llasa, option);
        if (options is AiDotNet.TextToSpeech.MultiModal.MoshiOptions moshiOptions)
            MoshiOptionFixture.Prepare(moshiOptions, option);
        if (options is AiDotNet.TextToSpeech.MultiModal.MoshiOptions moshi && option == "TtsPaddingTarget")
            // App. C applies the bonus only below the target fraction. A randomly initialized
            // tiny LM rarely samples padding with bonus 1, keeping all positive targets on
            // the same side of that branch. Use the real rate controller with a positive
            // bonus that lets the observed fraction cross the candidate targets.
            moshi.TtsPaddingBonus = 8;
        if (options is AiDotNet.TextToSpeech.Vocoders.VocosOptions vocos)
        {
            vocos.MelChannels = 8;
            // Both the control and its half-length candidate exceed torchaudio's reflection pad.
            if (IsVocosRecordingOption(option))
                vocos.SegmentSize = checked(2 * vocos.HopSize * Math.Max(2, 1 + vocos.FftSize / (2 * vocos.HopSize)));
            if (option == "CosineSteps") vocos.CosineSteps = 4;
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.WaveNetOptions waveNet)
        {
            waveNet.MelChannels = 8;
            if (IsWaveNetTrainingOption(option)) waveNet.SegmentSamples = checked(2 * waveNet.HopSize);
            if (option == "LearningRateHalvingSteps") waveNet.LearningRateHalvingSteps = 2;
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.HiFiGANOptions hiFiGan && IsHiFiGanTrainingOption(option))
        {
            // Keep the half-size perturbation frame-aligned and long enough for the released reflect padding.
            int hiFiCropHop = hiFiGan.HopSize;
            hiFiGan.SegmentSize = checked(2 * hiFiCropHop * Math.Max(2, (int)Math.Ceiling(hiFiGan.FftSize / (2.0 * hiFiCropHop))));
            if (option is "LearningRateDecay" or "UpdatesPerEpoch") hiFiGan.UpdatesPerEpoch = 2;
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.WaveRNNOptions waveRnn)
        {
            waveRnn.MelChannels = 8;
            waveRnn.SequenceSamples = 32;
            if (option == "MaxGradientNorm") waveRnn.MaxGradientNorm = 1e-5;
            if (IsWaveRnnSparseOption(option))
            {
                waveRnn.RnnDim = 16;
                waveRnn.SparsityTarget = .75;
                waveRnn.PruningStartStep = 0;
                waveRnn.PruningSteps = 4;
                waveRnn.PruningInterval = 1;
                waveRnn.PruneBlockRows = waveRnn.PruneBlockColumns = 1;
            }
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.DiffWaveOptions diffWave
            && options is not AiDotNet.TextToSpeech.Vocoders.PriorGradOptions && option is "NumClasses" or "UtteranceSamples")
        {
            diffWave.Conditioner = option == "NumClasses"
                ? AiDotNet.TextToSpeech.Vocoders.DiffWaveConditioner.ClassLabel
                : AiDotNet.TextToSpeech.Vocoders.DiffWaveConditioner.Unconditional;
            diffWave.NumClasses = option == "NumClasses" ? 2 : 0;
            diffWave.UtteranceSamples = 128;
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.ParallelWaveGANOptions parallel)
        {
            if (option is "AdversarialWeight" or "DiscriminatorGradientClipNorm" or "DiscriminatorLearningRate")
                parallel.DiscriminatorStartStep = 0;
            if (option == "DiscriminatorGradientClipNorm") parallel.DiscriminatorGradientClipNorm = 1e-6;
            if (option == "SegmentSize") parallel.SegmentSize = checked(4 * parallel.HopSize);
            if (option is "MelMean" or "MelScale")
            {
                parallel.MelMean = new double[parallel.MelChannels];
                parallel.MelScale = Enumerable.Repeat(1.0, parallel.MelChannels).ToArray();
            }
        }
        if (options is AiDotNet.TextToSpeech.Vocoders.MelGANOptions melGan && option is "SegmentSize" or "SamplingSeed")
            melGan.SegmentSize = checked(4 * melGan.UpsampleRates.Aggregate(1, (a, b) => a * b));
        if (options is AiDotNet.TextToSpeech.Vocoders.UnivNetOptions univNet && option == "SegmentSize")
            univNet.SegmentSize = checked(4 * univNet.HopSize);
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
        if (options is AiDotNet.Audio.Generation.EnCodecOptions encodecOptions)
        {
            // Residual unit j dilates by DilationBase^j; with the paper's single unit per block (j = 0) every base
            // gives dilation 1. Two units reach the base.
            if (option == "DilationBase") encodecOptions.ResidualLayers = 2;
            // The paper's 1-second chunks, so the overlap has chunks to overlap.
            if (option == "ChunkOverlap") encodecOptions.ChunkSeconds = 1.0;
        }
        if (option == "SACAlpha" && options.GetType().GetProperty("AutoTuneAlpha") is { CanWrite: true } autoTune
            && autoTune.PropertyType == typeof(bool))
            // SACAlpha is the fixed temperature (Haarnoja et al. 2018a); with the learned temperature it is unused.
            autoTune.SetValue(options, false);
        if (options is AiDotNet.ComputerVision.Segmentation.Semantic.SegNeXtOptions segNeXt && option == "WarmupStartFactor")
            // The fixture drops the 1500-iteration warmup the factor starts; restore a two-step ramp.
            segNeXt.WarmupIterations = 2;
        if (options is AiDotNet.Video.Options.ABMEOptions abme)
        {
            // Each motion stage decays its rate at its own milestones (100k+ iterations); train the stage the
            // option schedules, with a milestone the two probe steps reach.
            if (option == "SymmetricMilestones")
            {
                abme.Stage = AiDotNet.Enums.AbmeTrainingStage.SymmetricMotion;
                abme.SymmetricMilestones = new[] { 1 };
            }
            if (option == "AsymmetricMilestones")
            {
                abme.Stage = AiDotNet.Enums.AbmeTrainingStage.AsymmetricMotion;
                abme.AsymmetricMilestones = new[] { 1 };
            }
        }
        if (options is AiDotNet.SpeechRecognition.AlibabaASR.FunASRNanoOptions funAsrNano && option == "InverseTextNormalization")
            // The instruction is ChatML text in Qwen3's vocabulary; the 16-token test vocabulary has no prompt text, so
            // this probe uses the release's vocabulary (at the test width) where the instruction is tokenized.
            funAsrNano.VocabSize = 151936;
        if (options is AiDotNet.Video.Options.FlashVSROptions flash)
        {
            // FlashVSR trains in stages (decoder, dense teacher, sparse-causal student, distillation) of
            // 20000/20000/10000 iterations. Start each option's probe at the stage where it first acts.
            if (option is "TeacherIterations" or "NumTrainTimesteps" or "SigmaShift") flash.DecoderIterations = 0;
            if (option is "SparseIterations" or "SparseRatio" or "KvCacheChunks" or "LocalRange")
            {
                flash.DecoderIterations = 0;
                flash.TeacherIterations = 0;
            }
            // The LR projection's depth is bounded by the DiT's; the one-block test DiT admits only one projection layer.
            if (option == "LqProjectionLayers") flash.NumLayers = Math.Max(flash.NumLayers, 3);
        }
        if (options is AiDotNet.Video.Options.DOVEOptions dove && option is "FrameDifferenceWeight" or "ImageRatio" or "RefinementLearningRate")
        {
            // These act in the second (pixel-space refinement) stage, which follows 10000 adaptation iterations.
            dove.AdaptationIterations = 0;
            // The frame-difference loss needs a clip; the image branch trains on one frame.
            if (option == "FrameDifferenceWeight") dove.ImageRatio = 0;
        }
        if (options is AiDotNet.SpeechRecognition.AlibabaASR.SeACoOptions seaco && option is "BiasLabelSmoothing" or "HotwordMaskTokenId")
            // The bias decoder's loss (its smoothing and its no-bias class) trains in the separate bias stage that follows
            // backbone training with the Paraformer frozen.
            seaco.TrainingStage = AiDotNet.Enums.SeACoTrainingStage.Bias;
        if (options is AiDotNet.SpeechRecognition.Specialized.MedicalASROptions medicalAsr)
        {
            // Scheduled sampling ramps linearly from 0 over 100,000 steps, so the probe's first steps sample with
            // probability ~0. Hold the probability, or ramp over one step.
            if (option == "ScheduledSamplingProbability") medicalAsr.ScheduledSamplingRampSteps = 0;
            if (option == "ScheduledSamplingRampSteps") medicalAsr.ScheduledSamplingRampSteps = 1;
            // The output delay belongs to the unidirectional CTC arm (section 4.1); the bidirectional LAS ignores it.
            if (option == "OutputDelay")
            {
                medicalAsr.DecoderType = AiDotNet.Enums.MedicalAsrDecoderType.Ctc;
                medicalAsr.Bidirectional = false;
            }
        }
        if (IsGenericOptions(options, typeof(AiDotNet.Models.Options.ChronosFinanceOptions<>)))
        {
            // One recipe step of four windows per TrainOnSeries call; the probe's own steps call it again.
            if (option is "BatchSize" or "MinPast" or "MissingValueProbability")
            {
                SetOption(options, "TrainingSteps", 1);
                SetOption(options, "BatchSize", 4);
            }
            // Start below the probe's gradient norm so the clip binds.
            if (option == "MaxGradNorm") SetOption(options, "MaxGradNorm", 1e-8);
        }
        if (options is AiDotNet.SpeechRecognition.LLMIntegrated.Phi4AudioOptions phi4 && option == "UseLongRope")
        {
            // The release's LongRoPE factors cover 48 rotary frequencies: Phi-4-mini's 128-wide heads at a 0.75 partial
            // rotary factor. One 128-wide head keeps the fixture small.
            phi4.LlmDim = 128;
            phi4.NumLlmAttentionHeads = 1;
            phi4.NumLlmKvHeads = 1;
        }
        if (options is AiDotNet.TextToSpeech.EndToEnd.MeloTTSOptions meloOptions && option == "BertHeads")
            // MeloTTS reads BERT's hidden_states[-3]: the fixture's two layers stop at the embeddings, before any
            // attention. Three layers put one attention layer in front of the features.
            meloOptions.BertLayers = 3;
        if (options is AiDotNet.Video.Options.ShiftNetOptions shiftNet && option is "CosineIterations" or "MinLearningRate")
            // Over 450,000 iterations the cosine moves the rate by ~1e-11 of itself in the probe's steps; a four-step
            // cosine decays at once and reaches its floor.
            shiftNet.CosineIterations = 4;
        if (options is AiDotNet.Video.Options.PSRTOptions psrt && option == "FlowFreezeIterations")
            // SPyNet joins training after 5,000 iterations; from one, the probe's second step trains it.
            psrt.FlowFreezeIterations = 1;
        if (options is AiDotNet.Video.Options.DualXVSROptions dualX
            && option is "AdversarialLearningRate" or "DiscriminatorLearningRate" or "AdversarialPixelWeight" or "GANWeight")
            // These act in the GAN stage, which follows 650,000 Charbonnier iterations.
            dualX.ReconstructionIterations = 0;
        if (options is AiDotNet.Video.Options.StreamDiffVSROptions streamDiff)
        {
            // The adversarial term joins at iteration 20,000 on every DiscriminatorInterval-th step (step 0 is one).
            if (option == "GanWeight") streamDiff.GanStartIteration = 0;
            // From step 1 the first adversarial step is excluded; half of it (0) includes it.
            if (option == "GanStartIteration") streamDiff.GanStartIteration = 1;
        }
        if (options is AiDotNet.Video.Options.MGLDVSROptions mgld)
        {
            // The decoder stage follows 100,000 diffusion iterations, and its discriminator joins at decoder step 501.
            if (option is "DecoderLearningRate" or "DifferenceWeight" or "EdgeWeight" or "TrainingFusionWeight"
                or "DiscriminatorStart" or "DiscriminatorWeight")
                mgld.DiffusionIterations = 0;
            if (option == "DiscriminatorWeight") mgld.DiscriminatorStart = 0;
            // From decoder step 1 the first step is excluded; half of it (0) includes it.
            if (option == "DiscriminatorStart") mgld.DiscriminatorStart = 1;
            // The upscale is max(MinimumSide / shorter side, ScaleFactor); the fixture's 8-pixel frames at 4x give 32, so
            // a minimum side only acts above 32.
            if (option == "MinimumSide") mgld.MinimumSide = 40;
        }
        if (options is AiDotNet.Video.Options.DAMVSROptions damVsr)
        {
            // The three stages run 60,000 ControlNet, 45,000 backward and then decoder iterations. Start each option's
            // probe where it acts: the backward stage for one step, or the decoder stage.
            if (option == "BackwardIterations")
            {
                damVsr.ControlNetIterations = 0;
                damVsr.BackwardIterations = 1;
            }
            if (option is "GANWeight" or "DecoderSamplingSteps")
            {
                damVsr.ControlNetIterations = 0;
                damVsr.BackwardIterations = 0;
            }
            // SDEdit at strength 0.6 over the fixture's two steps enters at the last, SigmaMin, where the Karras
            // schedule's maximum and exponent cancel. Four steps enter at an interior sigma both set.
            if (option is "SigmaMax" or "KarrasRho") damVsr.NumInferenceSteps = 4;
        }
        if (options is AiDotNet.Models.Options.DetectionBackboneOptions backbone && option == "MaxGradNorm")
            // Start below the fixture's gradient norm so the clip binds in the probe's training steps.
            backbone.MaxGradNorm = 1e-6;
        if (options is AiDotNet.SpeechRecognition.Specialized.CodeSwitchingASROptions codeSwitching
            && option is "MandarinCharVocabSize" or "EnglishBpeVocabSize")
        {
            // The fixture's 15-token inventory as 6 Mandarin characters, 8 English units and the blank, so the
            // per-language sizes place the Mandarin/English boundary.
            codeSwitching.MandarinCharVocabSize = 6;
            codeSwitching.EnglishBpeVocabSize = 8;
        }
        if (options is AiDotNet.ComputerVision.Segmentation.Foundation.ISam2Settings)
        {
            // DropPath and ImageSize default to null (the size preset's rate, the architecture's side); the generic
            // perturbations of null (1, 2) are out of range, so start each from an explicit value.
            if (option == "DropPath")
            {
                if (options is AiDotNet.Video.Options.SAM2Options sam2Drop) sam2Drop.DropPath = 0.1;
                if (options is AiDotNet.ComputerVision.Segmentation.Foundation.SAM21Options sam21Drop) sam21Drop.DropPath = 0.1;
                if (options is AiDotNet.ComputerVision.Segmentation.Medical.MedicalSAM2Options medicalDrop) medicalDrop.DropPath = 0.1;
            }
            if (option == "ImageSize")
            {
                if (options is AiDotNet.Video.Options.SAM2Options sam2Side) sam2Side.ImageSize = 32;
                if (options is AiDotNet.ComputerVision.Segmentation.Foundation.SAM21Options sam21Side) sam21Side.ImageSize = 64;
            }
            // SAM 2's 32-pixel fixture gives Hiera stage maps of 8, 4, 2 and 1 tokens, each inside its {8, 4, 2, 1}
            // window, so a global block attends exactly as a windowed one. Halve the first two windows so they cut the
            // 8- and 4-token maps; the third stays 2, since the last stage pools its queries 2x2 inside it.
            if (option == "HieraGlobalAttentionBlockIndexes" && options is AiDotNet.Video.Options.SAM2Options sam2Windows)
                sam2Windows.HieraWindowSizes = new[] { 4, 2, 2, 1 };
            // The fixture's explicit depths and global blocks override the size preset; leave them to the preset.
            if (option == "ModelSize" && options is AiDotNet.ComputerVision.Segmentation.Medical.MedicalSAM2Options medicalSize)
            {
                medicalSize.HieraStageDepths = null;
                medicalSize.HieraGlobalAttentionBlockIndexes = null;
            }
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.FishSpeechFamilyOptions fishOptions)
        {
            if (option == "ContextLength")
            {
                // A budget of 30 tokens past the release's 1,024 reserve, over segments of a few words.
                fishOptions.ContextLength = 1054;
                fishOptions.ChunkLength = 24;
            }
            // Fish Speech 1.5 trains without dropout, so its training seed would draw nothing.
            if (option == "TrainingSeed") fishOptions.DropoutRate = 0.1;
        }
        // The emotion model answers until its budget; a short one keeps the probe's generation cheap.
        if (options is AiDotNet.TextToSpeech.Latest.IndexTTS2Options index2Options && option == "EmotionMaxTokens")
            index2Options.EmotionMaxTokens = 6;
        if (options is AiDotNet.TextToSpeech.MultiModal.LlamaOmniOptions omniOptions)
        {
            // The CTC weight scales the speech loss only in the release's joint stage; top-p applies only when sampling.
            if (option == "CtcLossWeight") omniOptions.Stage = AiDotNet.TextToSpeech.MultiModal.LlamaOmniTrainingStage.Joint;
            if (option == "TopP") omniOptions.Temperature = 1.0;
        }
        // The prefix MLP's width exists only when P-tuning v2 reparameterizes the prefixes (prefix_projection).
        if (options is AiDotNet.TextToSpeech.DescriptionBased.PromptTTSOptions promptTts && option == "PrefixHiddenSize")
            promptTts.PrefixProjection = true;
        if (options is AiDotNet.TextToSpeech.StyleEmotion.StyleTTSZSOptions styleZsIstft && option is "DecoderType" or "IstftFftSize")
        {
            // iSTFTNet's hop is UpsampleRates' product times IstftHopSize and must equal the mel hop; an iSTFT hop of 1
            // keeps the fixture's 8-sample hop for both decoders, so the decoder type and the iSTFT size can change.
            styleZsIstft.IstftHopSize = 1;
            if (option == "IstftFftSize") styleZsIstft.DecoderType = AiDotNet.TextToSpeech.StyleEmotion.StyleTts2DecoderType.IstftNet;
        }
        if (options is AiDotNet.TextToSpeech.MultiModal.SpeechGPTOptions speechGptOptions)
        {
            // The LoRA adapters exist only in the chain-of-modality stage (section 3.3); the vocoder's GAN step needs its
            // discriminators, and its per-epoch decay acts within the probe's steps only at one update per epoch.
            if (option is "LoraRank" or "LoraAlpha" or "LoraDropout")
                speechGptOptions.Stage = AiDotNet.TextToSpeech.MultiModal.SpeechGptTrainingStage.ChainOfModality;
            if (option is "VocoderLearningRate" or "VocoderLearningRateDecay" or "VocoderSegmentSize" or "VocoderUpdatesPerEpoch")
                speechGptOptions.IncludeVocoderDiscriminators = true;
            // At the default 1,000 updates per epoch the probe's few updates never reach a decay; from one, the epoch
            // count (and so the decayed rate) differs from the second update.
            if (option is "VocoderLearningRateDecay" or "VocoderUpdatesPerEpoch") speechGptOptions.VocoderUpdatesPerEpoch = 1;
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.ChatTTSOptions chatTts && option == "CodeTopK")
        {
            // The fixture's 3 x 3 FSQ gives 10 codes, below every top-k, so the TopKLogitsWarper never cuts. 5 x 5 levels
            // give 26 codes; top-p 1 leaves top-k as the only cut; the code-path mel makes Predict read the sampled codes
            // (the hidden-state path reads only the states before each sample).
            chatTts.FsqLevels = new[] { 5, 5 };
            chatTts.CodeTopP = 1.0;
            chatTts.UseHiddenStateDecoder = false;
        }
        if (options is AiDotNet.TextToSpeech.CodecBased.DiaOptions dia && option == "EndToken")
        {
            // The fixture's 20-token tables hold the 16 codec codes and end/padding/start at 16-18, so every other end
            // token collides. Move padding and start to 18 and 19 so the end token has a free slot at 17.
            dia.PadToken = 18;
            dia.BeginToken = 19;
        }
        if (options is AiDotNet.ComputerVision.Segmentation.Video.EfficientTAMOptions tamSize && option == "ModelSize")
        {
            // The size picks the ViT's width and heads (ViT-S 384/6, ViT-Ti 192/3); the fixture's explicit width and heads
            // override both presets. Leave them to the size.
            tamSize.VitEmbeddingDimension = null;
            tamSize.VitHeadCount = null;
        }
        if (options is AiDotNet.SpeechRecognition.Robust.AVHuBERTOptions avMask && option == "AudioMaskProbability")
            // The probe's 12 audio frames hold three starts for 10-frame spans, and at least two spans are drawn, so every
            // probability masks two. Two-frame spans leave eleven starts, where p sets the count (p T / l + U).
            avMask.AudioMaskLength = 2;
        if (options is AiDotNet.Video.Options.RealBasicVSROptions realBasic && option == "InferenceCleaningThreshold")
            // The fixture's first cleaning pass leaves a mean |residue| between 8/255 and 16/255 (measured: thresholds
            // up to 8 run all three passes, from 16 one). From 11 the doubled and halved perturbations fall on
            // either side, so one of them changes the number of passes.
            realBasic.InferenceCleaningThreshold = 11;

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

        if (options is AiDotNet.TextToSpeech.CodecBased.VevoOptions vevo)
            VevoOptionFixture.PrepareOptions(vevo, option);

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
        if (options is AiDotNet.TextToSpeech.Latest.IndexTTS2Options index && (IsIndexEmotionArchitecture(option) || option == "EmotionMaxTokens"))
        {
            index.EmotionHidden = 16;
            index.EmotionLayers = 1;
            index.EmotionHeads = 2;
            index.EmotionKeyValueHeads = 1;
            index.EmotionHeadDim = 8;
            index.EmotionIntermediate = 32;
            index.EmotionLoraRank = 2;
            if (option != "EmotionMaxTokens") index.EmotionMaxTokens = 1;
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
        // The training crop acts only on a recording longer than it; two hops crop the fixture's eight-hop recording.
        if (options is AiDotNet.TextToSpeech.Vocoders.ISTFTNetOptions istftNet && option == "SegmentSize")
            istftNet.SegmentSize = checked(2 * istftNet.HopSize);

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
        if (options is AiDotNet.TextToSpeech.DescriptionBased.ParlerTTSOptions rotary && option == "RopeTheta")
            rotary.RotaryEmbeddings = true;                 // the base acts only on rotary positions (off in the release)
        if (options is AiDotNet.TextToSpeech.DescriptionBased.PromptTTSOptions seeded && option == "SamplingSeed")
            seeded.StyleDropout = 0.1;                        // the seed draws the style encoder's dropout masks
        if (options is AiDotNet.Audio.Foundations.W2vBertOptions semantic && option == "SemanticLayer")
            semantic.MaskedLayers = 2;                       // the fixture's one masked block leaves no other layer to read
        if (options is AiDotNet.TextToSpeech.VoiceCloning.ChatterboxOptions distilled && option == "DistillationTeacherSteps")
        {
            distilled.Variant = AiDotNet.TextToSpeech.VoiceCloning.ChatterboxVariant.Turbo;
            distilled.FlowTimesteps = 2;
        }
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
        if (option == "TimeMasks" && options.GetType().GetProperty("TimeMaskRatio") is { CanWrite: true } ratio
            && ratio.PropertyType == typeof(double))
            // A time mask is at most TimeMaskRatio of the frames wide; 5% of a few fixture frames rounds to zero width.
            ratio.SetValue(options, 0.5);
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
            && !(options is AiDotNet.TextToSpeech.Vocoders.HiFiGANOptions && (option is "LearningRateDecay" or "UpdatesPerEpoch"))
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
            and not AiDotNet.SpeechRecognition.NeMo.CanaryQwen<T>
            and not AiDotNet.SpeechRecognition.NeMo.NemotronSpeech<T>
            and not AiDotNet.SpeechRecognition.AlibabaASR.Paraformer<T>
            and not AiDotNet.SpeechRecognition.AlibabaASR.ParaformerLarge<T>
            and not AiDotNet.SpeechRecognition.AlibabaASR.SeACo<T>
            and not AiDotNet.SpeechRecognition.AlibabaASR.Qwen3ASR<T>
            and not AiDotNet.SpeechRecognition.AlibabaASR.Qwen3ASRSmall<T>
            and not AiDotNet.SpeechRecognition.AlibabaASR.FunASRNano<T>
            and not AiDotNet.SpeechRecognition.LLMIntegrated.GraniteSpeech<T>
            and not AiDotNet.SpeechRecognition.LLMIntegrated.SALM<T>
            and not AiDotNet.SpeechRecognition.Specialized.VoxtLM<T>
            and not AiDotNet.Finance.Forecasting.Foundation.Chronos<T>
            and not AiDotNet.Video.Denoising.ShiftNet<T>
            and not AiDotNet.Audio.Foundations.HuBERT<T>) return;
        // The generated fixtures inject constant-rate optimizers for convergence invariants, and a rebuilt probe
        // inherits the shell's optimizer, whose schedule was built from the shell's options (an ASR model's
        // WarmupSteps). Reconstruct option probes without that override so the paper optimizer reads its settings.
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
            case AiDotNet.TextToSpeech.MultiModal.Moshi<T> model:
                // Replay the factory's real Mimi k-means initialization on each configuration
                // rebuild; an empty RVQ decodes all sampled LM codes identically.
                MoshiFixture.SpreadCodes(model.Core);
                break;
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
