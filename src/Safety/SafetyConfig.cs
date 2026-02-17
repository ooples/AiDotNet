using AiDotNet.Enums;

namespace AiDotNet.Safety;

/// <summary>
/// Master configuration for the comprehensive safety pipeline.
/// </summary>
/// <remarks>
/// <para>
/// This is the single configuration object for all safety features, accessed via
/// <c>AiModelBuilder.ConfigureSafety(Action&lt;SafetyConfig&gt;)</c>. It contains nested
/// sub-configs for each safety domain (text, image, audio, video, watermarking,
/// guardrails, fairness, compliance).
/// </para>
/// <para>
/// <b>For Beginners:</b> This is your one-stop safety control panel. You configure
/// everything through this single object:
/// <code>
/// .ConfigureSafety(safety =&gt;
/// {
///     safety.Enabled = true;
///     safety.Text.ToxicityDetection = true;
///     safety.Image.NSFWDetection = true;
///     safety.Guardrails.InputGuardrails = true;
/// })
/// </code>
/// All settings use nullable types with industry-standard defaults — if you don't set
/// something, a sensible default is used automatically.
/// </para>
/// </remarks>
public class SafetyConfig
{
    /// <summary>
    /// Gets or sets whether safety is enabled globally. Default: true.
    /// </summary>
    /// <remarks>
    /// When false, no safety modules run and the pipeline is skipped entirely.
    /// </remarks>
    public bool? Enabled { get; set; }

    /// <summary>
    /// Gets or sets the default action for safety violations when no module-specific action is configured.
    /// Default: Block.
    /// </summary>
    public SafetyAction? DefaultAction { get; set; }

    /// <summary>
    /// Gets or sets whether to throw an exception when unsafe input is detected. Default: true.
    /// </summary>
    /// <remarks>
    /// When true, a <see cref="SafetyViolationException"/> is thrown on unsafe input.
    /// When false, the safety report is attached to the result but processing continues.
    /// </remarks>
    public bool? ThrowOnUnsafeInput { get; set; }

    /// <summary>
    /// Gets or sets whether to throw an exception when unsafe output is detected. Default: false.
    /// </summary>
    /// <remarks>
    /// When true, a <see cref="SafetyViolationException"/> is thrown on unsafe output.
    /// When false, the safety report is attached to the result but the output is still returned.
    /// </remarks>
    public bool? ThrowOnUnsafeOutput { get; set; }

    /// <summary>
    /// Gets or sets the minimum severity level that triggers the configured action. Default: Medium.
    /// </summary>
    /// <remarks>
    /// Findings below this severity are logged but do not trigger blocking or other actions.
    /// </remarks>
    public SafetySeverity? MinimumActionSeverity { get; set; }

    /// <summary>
    /// Gets the text safety configuration (toxicity, PII, jailbreak, hallucination, copyright).
    /// </summary>
    public TextSafetyConfig Text { get; } = new();

    /// <summary>
    /// Gets the image safety configuration (NSFW, violence, deepfake).
    /// </summary>
    public ImageSafetyConfig Image { get; } = new();

    /// <summary>
    /// Gets the audio safety configuration (deepfake, toxic speech, voice protection).
    /// </summary>
    public AudioSafetyConfig Audio { get; } = new();

    /// <summary>
    /// Gets the video safety configuration (content moderation, temporal deepfake).
    /// </summary>
    public VideoSafetyConfig Video { get; } = new();

    /// <summary>
    /// Gets the watermarking configuration (text, image, audio watermarking).
    /// </summary>
    public WatermarkConfig Watermarking { get; } = new();

    /// <summary>
    /// Gets the guardrails configuration (input/output guardrails, topic restrictions).
    /// </summary>
    public GuardrailConfig Guardrails { get; } = new();

    /// <summary>
    /// Gets the fairness configuration (bias detection, demographic parity).
    /// </summary>
    public FairnessConfig Fairness { get; } = new();

    /// <summary>
    /// Gets the regulatory compliance configuration (EU AI Act, GDPR, SOC2).
    /// </summary>
    public ComplianceConfig Compliance { get; } = new();

    // -- Internal defaults resolution --

    internal bool EffectiveEnabled => Enabled ?? true;
    internal SafetyAction EffectiveDefaultAction => DefaultAction ?? SafetyAction.Block;
    internal bool EffectiveThrowOnUnsafeInput => ThrowOnUnsafeInput ?? true;
    internal bool EffectiveThrowOnUnsafeOutput => ThrowOnUnsafeOutput ?? false;
    internal SafetySeverity EffectiveMinimumActionSeverity => MinimumActionSeverity ?? SafetySeverity.Medium;
}

/// <summary>
/// Configuration for text safety modules.
/// </summary>
/// <remarks>
/// <para>
/// Controls which text safety checks are enabled and their sensitivity thresholds.
/// </para>
/// <para>
/// <b>For Beginners:</b> These settings control how text content is checked for safety issues.
/// Enable the checks you need — toxicity catches offensive language, PII detection finds
/// personal information, jailbreak detection catches prompt manipulation attempts.
/// </para>
/// </remarks>
public class TextSafetyConfig
{
    /// <summary>
    /// Gets or sets whether toxicity detection is enabled. Default: true.
    /// </summary>
    public bool? ToxicityDetection { get; set; }

    /// <summary>
    /// Gets or sets the toxicity score threshold (0-1). Content above this is flagged. Default: 0.7.
    /// </summary>
    public double? ToxicityThreshold { get; set; }

    /// <summary>
    /// Gets or sets whether PII (personally identifiable information) detection is enabled. Default: true.
    /// </summary>
    public bool? PIIDetection { get; set; }

    /// <summary>
    /// Gets or sets whether jailbreak/prompt injection detection is enabled. Default: true.
    /// </summary>
    public bool? JailbreakDetection { get; set; }

    /// <summary>
    /// Gets or sets the jailbreak detection sensitivity (0-1). Higher catches more but may false-positive. Default: 0.7.
    /// </summary>
    public double? JailbreakSensitivity { get; set; }

    /// <summary>
    /// Gets or sets whether hallucination detection is enabled. Default: false.
    /// </summary>
    /// <remarks>
    /// Hallucination detection requires reference context to compare against.
    /// It is disabled by default because it needs additional inputs.
    /// </remarks>
    public bool? HallucinationDetection { get; set; }

    /// <summary>
    /// Gets or sets whether copyright/memorization detection is enabled. Default: false.
    /// </summary>
    public bool? CopyrightDetection { get; set; }

    /// <summary>
    /// Gets or sets the languages to support for text safety checks. Default: ["en"].
    /// </summary>
    public string[]? Languages { get; set; }

    /// <summary>
    /// Gets or sets the maximum input text length. Inputs exceeding this are rejected. Default: 10000.
    /// </summary>
    public int? MaxInputLength { get; set; }

    // -- Internal defaults --
    internal bool EffectiveToxicityDetection => ToxicityDetection ?? true;
    internal double EffectiveToxicityThreshold => ToxicityThreshold ?? 0.7;
    internal bool EffectivePIIDetection => PIIDetection ?? true;
    internal bool EffectiveJailbreakDetection => JailbreakDetection ?? true;
    internal double EffectiveJailbreakSensitivity => JailbreakSensitivity ?? 0.7;
    internal bool EffectiveHallucinationDetection => HallucinationDetection ?? false;
    internal bool EffectiveCopyrightDetection => CopyrightDetection ?? false;
    internal string[] EffectiveLanguages => Languages ?? new[] { "en" };
    internal int EffectiveMaxInputLength => MaxInputLength ?? 10000;
}

/// <summary>
/// Configuration for image safety modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how images are checked for harmful content.
/// NSFW detection catches adult content, violence detection catches graphic imagery,
/// and deepfake detection identifies AI-manipulated faces.
/// </para>
/// <para>
/// <b>References:</b>
/// - UnsafeBench: GPT-4V achieves top F1 across 11 categories (Qu et al., 2024)
/// - Vision Transformers outperform CNNs for sensitive image classification (2024)
/// </para>
/// </remarks>
public class ImageSafetyConfig
{
    /// <summary>
    /// Gets or sets whether NSFW (sexual content) detection is enabled. Default: true.
    /// </summary>
    public bool? NSFWDetection { get; set; }

    /// <summary>
    /// Gets or sets the NSFW detection threshold (0-1). Content above this is flagged. Default: 0.8.
    /// </summary>
    /// <remarks>
    /// The default of 0.8 balances precision and recall. Lower values (0.5-0.7)
    /// catch more content but increase false positives.
    /// </remarks>
    public double? NSFWThreshold { get; set; }

    /// <summary>
    /// Gets or sets whether violence detection is enabled. Default: true.
    /// </summary>
    public bool? ViolenceDetection { get; set; }

    /// <summary>
    /// Gets or sets whether deepfake/AI-generated image detection is enabled. Default: false.
    /// </summary>
    public bool? DeepfakeDetection { get; set; }

    /// <summary>
    /// Gets or sets the type of image safety classifier to use. Default: null (auto-select).
    /// </summary>
    public string? ClassifierType { get; set; }

    // -- Internal defaults --
    internal bool EffectiveNSFWDetection => NSFWDetection ?? true;
    internal double EffectiveNSFWThreshold => NSFWThreshold ?? 0.8;
    internal bool EffectiveViolenceDetection => ViolenceDetection ?? true;
    internal bool EffectiveDeepfakeDetection => DeepfakeDetection ?? false;
}

/// <summary>
/// Configuration for audio safety modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how audio content is checked for safety.
/// Deepfake detection identifies cloned or synthetic voices, and toxic speech detection
/// catches harmful spoken content.
/// </para>
/// <para>
/// <b>References:</b>
/// - SafeEar: Privacy-preserving audio deepfake detection (ACM CCS 2024)
/// - AudioSeal: Localized watermarking for voice cloning detection (Meta AI, 2024)
/// </para>
/// </remarks>
public class AudioSafetyConfig
{
    /// <summary>
    /// Gets or sets whether audio deepfake detection is enabled. Default: false.
    /// </summary>
    public bool? DeepfakeDetection { get; set; }

    /// <summary>
    /// Gets or sets whether toxic speech detection is enabled. Default: false.
    /// </summary>
    public bool? ToxicSpeechDetection { get; set; }

    /// <summary>
    /// Gets or sets the expected sample rate in Hz. Default: 16000.
    /// </summary>
    public int? SampleRate { get; set; }

    // -- Internal defaults --
    internal bool EffectiveDeepfakeDetection => DeepfakeDetection ?? false;
    internal bool EffectiveToxicSpeechDetection => ToxicSpeechDetection ?? false;
    internal int EffectiveSampleRate => SampleRate ?? 16000;
}

/// <summary>
/// Configuration for video safety modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These settings control how video content is checked for safety.
/// Content moderation checks sampled frames for harmful imagery, while deepfake detection
/// analyzes temporal consistency to spot manipulated videos.
/// </para>
/// </remarks>
public class VideoSafetyConfig
{
    /// <summary>
    /// Gets or sets whether video deepfake detection is enabled. Default: false.
    /// </summary>
    public bool? DeepfakeDetection { get; set; }

    /// <summary>
    /// Gets or sets whether general video content moderation is enabled. Default: false.
    /// </summary>
    public bool? ContentModeration { get; set; }

    /// <summary>
    /// Gets or sets the frame sampling rate (frames per second to check). Default: 1.0.
    /// </summary>
    /// <remarks>
    /// Higher values are more thorough but slower. A value of 1.0 means one frame
    /// per second is checked. For short clips, consider higher values.
    /// </remarks>
    public double? FrameSamplingRate { get; set; }

    // -- Internal defaults --
    internal bool EffectiveDeepfakeDetection => DeepfakeDetection ?? false;
    internal bool EffectiveContentModeration => ContentModeration ?? false;
    internal double EffectiveFrameSamplingRate => FrameSamplingRate ?? 1.0;
}

/// <summary>
/// Configuration for watermarking modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Watermarking embeds invisible markers in AI-generated content
/// so it can later be identified as AI-generated. This is increasingly required by
/// regulations like the EU AI Act (Article 50).
/// </para>
/// <para>
/// <b>References:</b>
/// - SynthID-Text: Production text watermarking at scale (Google DeepMind, Nature 2024)
/// - SynthID-Image: Internet-scale image watermarking (Google DeepMind, 2025)
/// - Only 38% of AI generators implement adequate watermarking (2025)
/// </para>
/// </remarks>
public class WatermarkConfig
{
    /// <summary>
    /// Gets or sets whether text watermarking is enabled. Default: false.
    /// </summary>
    public bool? TextWatermarking { get; set; }

    /// <summary>
    /// Gets or sets whether image watermarking is enabled. Default: false.
    /// </summary>
    public bool? ImageWatermarking { get; set; }

    /// <summary>
    /// Gets or sets whether audio watermarking is enabled. Default: false.
    /// </summary>
    public bool? AudioWatermarking { get; set; }

    /// <summary>
    /// Gets or sets watermark detection mode (detect existing watermarks on input). Default: false.
    /// </summary>
    public bool? DetectionMode { get; set; }

    /// <summary>
    /// Gets or sets the watermark strength (0-1). Higher is more robust but may affect quality. Default: 0.5.
    /// </summary>
    public double? WatermarkStrength { get; set; }

    // -- Internal defaults --
    internal bool EffectiveTextWatermarking => TextWatermarking ?? false;
    internal bool EffectiveImageWatermarking => ImageWatermarking ?? false;
    internal bool EffectiveAudioWatermarking => AudioWatermarking ?? false;
    internal bool EffectiveDetectionMode => DetectionMode ?? false;
    internal double EffectiveWatermarkStrength => WatermarkStrength ?? 0.5;
}

/// <summary>
/// Configuration for input/output guardrails.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Guardrails are safety barriers that check content before and
/// after model processing. Input guardrails validate requests before they reach the model;
/// output guardrails validate responses before they reach the user.
/// </para>
/// <para>
/// <b>References:</b>
/// - ShieldGemma: LLM-based safety models (Google DeepMind, 2024)
/// - WildGuard: Open moderation covering 13 risk categories (Allen AI, 2024)
/// - Qwen3Guard: 85.3% accuracy, robust to prompt variation (Alibaba, 2025)
/// </para>
/// </remarks>
public class GuardrailConfig
{
    /// <summary>
    /// Gets or sets whether input guardrails are enabled. Default: true.
    /// </summary>
    public bool? InputGuardrails { get; set; }

    /// <summary>
    /// Gets or sets whether output guardrails are enabled. Default: true.
    /// </summary>
    public bool? OutputGuardrails { get; set; }

    /// <summary>
    /// Gets or sets restricted topics that should be blocked. Default: empty.
    /// </summary>
    public string[]? TopicRestrictions { get; set; }

    /// <summary>
    /// Gets or sets the maximum allowed input length. Default: 10000.
    /// </summary>
    public int? MaxInputLength { get; set; }

    // -- Internal defaults --
    internal bool EffectiveInputGuardrails => InputGuardrails ?? true;
    internal bool EffectiveOutputGuardrails => OutputGuardrails ?? true;
    internal string[] EffectiveTopicRestrictions => TopicRestrictions ?? Array.Empty<string>();
    internal int EffectiveMaxInputLength => MaxInputLength ?? 10000;
}

/// <summary>
/// Configuration for fairness and bias detection.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Fairness settings check whether the model treats all demographic
/// groups equitably. For example, ensuring the model doesn't give different quality
/// results for different genders or ethnicities.
/// </para>
/// <para>
/// <b>References:</b>
/// - BEATS: Comprehensive bias evaluation test suite for LLMs (2025)
/// - SB-Bench: Stereotype bias benchmark for multimodal models (2025)
/// - Demographic-targeted bias: race/ethnicity 55.6% exploitability (2025)
/// </para>
/// </remarks>
public class FairnessConfig
{
    /// <summary>
    /// Gets or sets whether demographic parity checking is enabled. Default: false.
    /// </summary>
    public bool? DemographicParity { get; set; }

    /// <summary>
    /// Gets or sets whether equalized odds checking is enabled. Default: false.
    /// </summary>
    public bool? EqualizedOdds { get; set; }

    /// <summary>
    /// Gets or sets the protected attributes to monitor. Default: empty.
    /// </summary>
    public string[]? ProtectedAttributes { get; set; }

    /// <summary>
    /// Gets or sets whether stereotype detection is enabled. Default: false.
    /// </summary>
    public bool? StereotypeDetection { get; set; }

    // -- Internal defaults --
    internal bool EffectiveDemographicParity => DemographicParity ?? false;
    internal bool EffectiveEqualizedOdds => EqualizedOdds ?? false;
    internal string[] EffectiveProtectedAttributes => ProtectedAttributes ?? Array.Empty<string>();
    internal bool EffectiveStereotypeDetection => StereotypeDetection ?? false;
}

/// <summary>
/// Configuration for regulatory compliance.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Compliance settings enable regulatory framework-specific checks.
/// The EU AI Act requires transparency and watermarking for AI-generated content.
/// GDPR requires PII detection and erasure support. SOC2 requires audit logging.
/// Enabling a compliance mode automatically enables the safety features required by that regulation.
/// </para>
/// </remarks>
public class ComplianceConfig
{
    /// <summary>
    /// Gets or sets whether EU AI Act compliance mode is enabled. Default: false.
    /// </summary>
    /// <remarks>
    /// When enabled, automatically enables: watermarking, risk classification,
    /// transparency reporting, and human oversight hooks.
    /// </remarks>
    public bool? EUAIAct { get; set; }

    /// <summary>
    /// Gets or sets whether GDPR compliance mode is enabled. Default: false.
    /// </summary>
    /// <remarks>
    /// When enabled, automatically enables: PII detection, data minimization checks,
    /// and right-to-erasure support.
    /// </remarks>
    public bool? GDPR { get; set; }

    /// <summary>
    /// Gets or sets whether SOC2 compliance mode is enabled. Default: false.
    /// </summary>
    /// <remarks>
    /// When enabled, automatically enables: audit logging, access logging,
    /// and safety event tracking.
    /// </remarks>
    public bool? SOC2 { get; set; }

    // -- Internal defaults --
    internal bool EffectiveEUAIAct => EUAIAct ?? false;
    internal bool EffectiveGDPR => GDPR ?? false;
    internal bool EffectiveSOC2 => SOC2 ?? false;
}
