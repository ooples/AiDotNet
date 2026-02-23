namespace AiDotNet.Enums;

/// <summary>
/// Comprehensive taxonomy of safety and harm categories for content classification.
/// </summary>
/// <remarks>
/// <para>
/// This enum defines a hierarchical taxonomy of content safety categories covering
/// all known types of harmful, inappropriate, or policy-violating content across
/// text, image, audio, and video modalities.
/// </para>
/// <para>
/// <b>For Beginners:</b> These are the specific types of harmful content that the
/// safety system can detect. Each category represents a different kind of risk,
/// from hate speech to deepfakes to copyright violations. You can configure which
/// categories to check for and what action to take for each.
/// </para>
/// <para>
/// <b>References:</b>
/// - UnsafeBench 11-category taxonomy (Qu et al., 2024)
/// - WildGuard 13-risk-category classification (Allen AI, 2024)
/// - OmniSafeBench-MM 9 risk domains with 50 fine-grained categories (2025)
/// - MM-SafetyBench 13 scenarios (Liu et al., ECCV 2024)
/// - EU AI Act risk classification (Articles 5, 6, 50, 52)
/// </para>
/// </remarks>
public enum SafetyCategory
{
    // === Sexual Content ===

    /// <summary>
    /// Sexually explicit content including pornography or graphic sexual acts.
    /// </summary>
    SexualExplicit,

    /// <summary>
    /// Sexually suggestive content that is not explicitly graphic.
    /// </summary>
    SexualSuggestive,

    /// <summary>
    /// Child sexual abuse material (CSAM). Critical severity — must always be blocked.
    /// </summary>
    SexualMinors,

    // === Violence ===

    /// <summary>
    /// Graphic depictions of violence, gore, or injury.
    /// </summary>
    ViolenceGraphic,

    /// <summary>
    /// Threats of violence against individuals or groups.
    /// </summary>
    ViolenceThreat,

    /// <summary>
    /// Content promoting or depicting weapons use.
    /// </summary>
    ViolenceWeapons,

    /// <summary>
    /// Content promoting, glorifying, or instructing self-harm.
    /// </summary>
    ViolenceSelfHarm,

    /// <summary>
    /// Content promoting, glorifying, or instructing suicide.
    /// </summary>
    ViolenceSuicide,

    /// <summary>
    /// Content promoting, glorifying, or instructing terrorism.
    /// </summary>
    ViolenceTerrorism,

    // === Hate and Discrimination ===

    /// <summary>
    /// Hate speech targeting protected groups based on race, religion, gender, etc.
    /// </summary>
    HateSpeech,

    /// <summary>
    /// Targeted harassment or bullying of individuals.
    /// </summary>
    Harassment,

    /// <summary>
    /// Discriminatory content or policies targeting protected groups.
    /// </summary>
    Discrimination,

    /// <summary>
    /// Stereotyping content that reinforces harmful generalizations.
    /// </summary>
    Stereotyping,

    /// <summary>
    /// Content that dehumanizes or degrades individuals or groups.
    /// </summary>
    Dehumanization,

    // === Dangerous Content ===

    /// <summary>
    /// Instructions or promotion of illegal activities.
    /// </summary>
    IllegalActivities,

    /// <summary>
    /// Instructions for manufacturing or using weapons.
    /// </summary>
    WeaponsInstructions,

    /// <summary>
    /// Instructions for manufacturing illegal drugs.
    /// </summary>
    DrugManufacturing,

    /// <summary>
    /// Malware, hacking tools, or cyberattack instructions.
    /// </summary>
    Malware,

    // === Deception ===

    /// <summary>
    /// Misinformation — unintentionally false or misleading content.
    /// </summary>
    Misinformation,

    /// <summary>
    /// Disinformation — deliberately false content intended to deceive.
    /// </summary>
    Disinformation,

    /// <summary>
    /// Impersonation of real individuals or organizations.
    /// </summary>
    Impersonation,

    /// <summary>
    /// Fraud, scam, or social engineering content.
    /// </summary>
    Fraud,

    /// <summary>
    /// Social engineering content designed to manipulate victims.
    /// </summary>
    SocialEngineering,

    // === Privacy ===

    /// <summary>
    /// Exposure of personally identifiable information (PII).
    /// </summary>
    PIIExposure,

    /// <summary>
    /// Doxxing — publishing private information to identify or locate someone.
    /// </summary>
    Doxxing,

    /// <summary>
    /// Content that enables mass surveillance.
    /// </summary>
    SurveillanceEnabling,

    // === AI-Specific Risks ===

    /// <summary>
    /// Prompt injection — attempts to override system instructions.
    /// </summary>
    PromptInjection,

    /// <summary>
    /// Jailbreak attempt — attempts to bypass safety measures.
    /// </summary>
    JailbreakAttempt,

    /// <summary>
    /// Hallucinated or fabricated content not grounded in facts or source material.
    /// </summary>
    Hallucination,

    /// <summary>
    /// Content that infringes copyright or reproduces protected works.
    /// </summary>
    CopyrightViolation,

    /// <summary>
    /// Leakage of memorized training data.
    /// </summary>
    TrainingDataLeakage,

    /// <summary>
    /// Attempts to extract or replicate the model itself.
    /// </summary>
    ModelExtraction,

    // === Regulated Advice ===

    /// <summary>
    /// Unqualified medical advice that could cause harm.
    /// </summary>
    MedicalAdvice,

    /// <summary>
    /// Unqualified legal advice that could cause harm.
    /// </summary>
    LegalAdvice,

    /// <summary>
    /// Unqualified financial advice that could cause harm.
    /// </summary>
    FinancialAdvice,

    // === Content Integrity ===

    /// <summary>
    /// Content detected as AI-generated (text, image, audio, or video).
    /// </summary>
    AIGenerated,

    /// <summary>
    /// Deepfake content — AI-generated or manipulated media impersonating real people.
    /// </summary>
    Deepfake,

    /// <summary>
    /// Content that has been digitally manipulated or altered.
    /// </summary>
    Manipulated,

    /// <summary>
    /// Content that contains a digital watermark (informational, not harmful).
    /// </summary>
    Watermarked,

    // === Fairness ===

    /// <summary>
    /// Biased content that treats demographic groups inequitably.
    /// </summary>
    Bias,

    /// <summary>
    /// Content that lacks transparency about AI involvement (regulatory compliance).
    /// </summary>
    TransparencyViolation,

    // === Policy ===

    /// <summary>
    /// Content that violates a topic restriction or custom policy rule.
    /// </summary>
    PolicyViolation
}
