using AiDotNet.Safety.Adversarial;
using AiDotNet.Safety.Audio;
using AiDotNet.Safety.Compliance;
using AiDotNet.Safety.Fairness;
using AiDotNet.Safety.Guardrails;
using AiDotNet.Safety.Image;
using AiDotNet.Safety.Multimodal;
using AiDotNet.Safety.Text;
using AiDotNet.Safety.Video;
using AiDotNet.Safety.Watermarking;

namespace AiDotNet.Safety;

/// <summary>
/// Factory that constructs a <see cref="SafetyPipeline{T}"/> from a <see cref="SafetyConfig"/>.
/// </summary>
/// <remarks>
/// <para>
/// This factory reads the SafetyConfig and instantiates the appropriate safety modules
/// based on which features are enabled. It is called internally by the AiModelBuilder
/// during the build process.
/// </para>
/// <para>
/// <b>For Beginners:</b> You don't need to use this factory directly. When you call
/// <c>ConfigureSafety(...)</c> on the AiModelBuilder, this factory automatically creates
/// the right safety modules based on your settings.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class SafetyPipelineFactory<T>
{
    /// <summary>
    /// Creates a safety pipeline from the given configuration.
    /// </summary>
    /// <param name="config">The safety configuration. If null, defaults are used.</param>
    /// <returns>A configured safety pipeline with the appropriate modules registered.</returns>
    public static SafetyPipeline<T> Create(SafetyConfig? config = null)
    {
        var effectiveConfig = config ?? new SafetyConfig();
        var pipeline = new SafetyPipeline<T>(effectiveConfig);

        if (!effectiveConfig.EffectiveEnabled)
        {
            return pipeline;
        }

        // Register modules in order: guardrails first, then content analysis,
        // then adversarial/fairness, then multimodal, then compliance
        RegisterGuardrailModules(pipeline, effectiveConfig);
        RegisterTextModules(pipeline, effectiveConfig);
        RegisterImageModules(pipeline, effectiveConfig);
        RegisterAudioModules(pipeline, effectiveConfig);
        RegisterVideoModules(pipeline, effectiveConfig);
        RegisterWatermarkModules(pipeline, effectiveConfig);
        RegisterAdversarialModules(pipeline, effectiveConfig);
        RegisterFairnessModules(pipeline, effectiveConfig);
        RegisterMultimodalModules(pipeline, effectiveConfig);
        RegisterComplianceModules(pipeline, effectiveConfig);

        return pipeline;
    }

    private static void RegisterGuardrailModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Guardrails.EffectiveInputGuardrails)
        {
            pipeline.AddModule(new InputGuardrail<T>(
                maxInputLength: config.Guardrails.EffectiveMaxInputLength));
        }

        if (config.Guardrails.EffectiveOutputGuardrails)
        {
            pipeline.AddModule(new OutputGuardrail<T>());
        }

        if (config.Guardrails.EffectiveTopicRestrictions.Length > 0)
        {
            pipeline.AddModule(new TopicRestrictionGuardrail<T>(
                config.Guardrails.EffectiveTopicRestrictions));
        }
    }

    private static void RegisterTextModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Text.EffectiveToxicityDetection)
        {
            // Register ensemble toxicity detector (combines rule-based, embedding, and classifier)
            pipeline.AddModule(new EnsembleToxicityDetector<T>(config.Text.EffectiveToxicityThreshold));
        }

        if (config.Text.EffectivePIIDetection)
        {
            // Register composite PII detector (combines regex, NER, and context-aware)
            pipeline.AddModule(new CompositePIIDetector<T>());
        }

        if (config.Text.EffectiveJailbreakDetection)
        {
            // Register ensemble jailbreak detector (combines pattern, semantic, and gradient)
            pipeline.AddModule(new EnsembleJailbreakDetector<T>(
                sensitivity: config.Text.EffectiveJailbreakSensitivity));
        }

        if (config.Text.EffectiveHallucinationDetection)
        {
            pipeline.AddModule(new SelfConsistencyHallucinationDetector<T>());
        }

        if (config.Text.EffectiveCopyrightDetection)
        {
            pipeline.AddModule(new NgramCopyrightDetector<T>());
            pipeline.AddModule(new PerplexityMemorizationDetector<T>());
        }
    }

    private static void RegisterImageModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        bool nsfw = config.Image.EffectiveNSFWDetection;
        bool violence = config.Image.EffectiveViolenceDetection;

        if (nsfw || violence)
        {
            // Use ensemble image classifier for better accuracy
            double imageThreshold = Math.Min(config.Image.EffectiveNSFWThreshold,
                config.Image.EffectiveViolenceThreshold);
            pipeline.AddModule(new EnsembleImageSafetyClassifier<T>(imageThreshold));
        }

        if (config.Image.EffectiveDeepfakeDetection)
        {
            pipeline.AddModule(new FrequencyDeepfakeDetector<T>());
            pipeline.AddModule(new ConsistencyDeepfakeDetector<T>());
            pipeline.AddModule(new ProvenanceDeepfakeDetector<T>());
        }
    }

    private static void RegisterAudioModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Audio.EffectiveDeepfakeDetection)
        {
            pipeline.AddModule(new SpectralDeepfakeDetector<T>(
                defaultSampleRate: config.Audio.EffectiveSampleRate));
            pipeline.AddModule(new VoiceprintDeepfakeDetector<T>(
                sampleRate: config.Audio.EffectiveSampleRate));
            pipeline.AddModule(new WatermarkDeepfakeDetector<T>(
                sampleRate: config.Audio.EffectiveSampleRate));
        }

        if (config.Audio.EffectiveToxicSpeechDetection)
        {
            pipeline.AddModule(new TranscriptionToxicityDetector<T>(
                defaultSampleRate: config.Audio.EffectiveSampleRate));
            pipeline.AddModule(new AcousticToxicityDetector<T>(
                sampleRate: config.Audio.EffectiveSampleRate));
        }
    }

    private static void RegisterVideoModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Video.EffectiveContentModeration)
        {
            pipeline.AddModule(new MultimodalVideoModerator<T>(
                samplingRate: config.Video.EffectiveFrameSamplingRate));
        }

        if (config.Video.EffectiveDeepfakeDetection)
        {
            pipeline.AddModule(new TemporalConsistencyDetector<T>());
        }
    }

    private static void RegisterWatermarkModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Watermarking.EffectiveTextWatermarking || config.Watermarking.EffectiveDetectionMode)
        {
            pipeline.AddModule(new TextWatermarker<T>());
        }

        if (config.Watermarking.EffectiveImageWatermarking || config.Watermarking.EffectiveDetectionMode)
        {
            pipeline.AddModule(new ImageWatermarker<T>(
                watermarkStrength: config.Watermarking.EffectiveWatermarkStrength));
        }

        if (config.Watermarking.EffectiveAudioWatermarking || config.Watermarking.EffectiveDetectionMode)
        {
            pipeline.AddModule(new AudioWatermarker<T>(
                watermarkStrength: config.Watermarking.EffectiveWatermarkStrength));
        }
    }

    private static void RegisterAdversarialModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        // Adversarial robustness is automatically enabled when jailbreak detection or
        // image safety is enabled, as adversarial attacks target those systems
        if (config.Text.EffectiveJailbreakDetection)
        {
            pipeline.AddModule(new AdversarialRobustnessEvaluator<T>());
        }

        if (config.Image.EffectiveDeepfakeDetection)
        {
            pipeline.AddModule(new AdversarialImageEvaluator<T>());
        }
    }

    private static void RegisterFairnessModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Fairness.EffectiveDemographicParity)
        {
            pipeline.AddModule(new DemographicParityChecker<T>(
                protectedAttributes: config.Fairness.EffectiveProtectedAttributes.Length > 0
                    ? config.Fairness.EffectiveProtectedAttributes
                    : null));
        }

        if (config.Fairness.EffectiveEqualizedOdds)
        {
            pipeline.AddModule(new EqualizedOddsChecker<T>());
        }

        if (config.Fairness.EffectiveStereotypeDetection)
        {
            pipeline.AddModule(new StereotypeDetector<T>());
        }
    }

    private static void RegisterMultimodalModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        // Register cross-modal consistency checker if multiple modalities are active
        bool hasText = HasTextModality(config);
        bool hasImage = HasImageModality(config);
        bool hasAudio = HasAudioModality(config);

        // Cross-modal checking requires at least two active modalities
        int activeModalities = (hasText ? 1 : 0) + (hasImage ? 1 : 0) + (hasAudio ? 1 : 0);
        if (activeModalities >= 2)
        {
            pipeline.AddModule(new CrossModalConsistencyChecker<T>());
            pipeline.AddModule(new TextImageAlignmentChecker<T>());
        }
    }

    private static bool HasTextModality(SafetyConfig config) =>
        config.Text.EffectiveToxicityDetection ||
        config.Text.EffectivePIIDetection ||
        config.Text.EffectiveJailbreakDetection;

    private static bool HasImageModality(SafetyConfig config) =>
        config.Image.EffectiveNSFWDetection ||
        config.Image.EffectiveViolenceDetection;

    private static bool HasAudioModality(SafetyConfig config) =>
        config.Audio.EffectiveDeepfakeDetection ||
        config.Audio.EffectiveToxicSpeechDetection;

    private static void RegisterComplianceModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        if (config.Compliance.EffectiveEUAIAct)
        {
            pipeline.AddModule(new EUAIActComplianceChecker<T>(config));
        }

        if (config.Compliance.EffectiveGDPR)
        {
            pipeline.AddModule(new GDPRComplianceChecker<T>(config));
        }

        if (config.Compliance.EffectiveSOC2)
        {
            pipeline.AddModule(new SOC2ComplianceChecker<T>(config));
        }
    }
}
