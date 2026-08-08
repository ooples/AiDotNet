using AiDotNet.Enums;

namespace AiDotNet.Safety.Video;

/// <summary>
/// Maps the library's fine-grained <see cref="SafetyCategory"/> signals onto the six-category harm
/// taxonomy of arXiv:2411.05854.
/// </summary>
/// <remarks>
/// <para>
/// The taxonomy is deliberately coarse — six categories "based on their commonalities" — while
/// <see cref="SafetyCategory"/> is a long list of specific detectable signals. This is the join
/// between them: each taxonomy category owns the set of signals that evidence it.
/// </para>
/// <para>
/// The mapping follows the paper's own wording for each category. Signals that no taxonomy category
/// claims — model-integrity concerns such as PromptInjection or ModelExtraction, or provenance
/// labels such as Watermarked — are intentionally unmapped: they are not harms to a viewer of a
/// video, and the paper excludes anything not discernible as harm from the content itself.
/// </para>
/// <para>
/// <b>For Beginners:</b> The detectors report very specific things ("hate speech", "graphic
/// violence", "fraud"). The taxonomy groups those into six broad kinds of harm. This file records
/// which specific findings count as evidence for which broad kind.
/// </para>
/// </remarks>
/// <remarks>
/// INTERNAL. This is a join table between two internal vocabularies, consumed only by
/// <see cref="MultimodalVideoModerator{T}"/>. Users reach safety through AiModelBuilder /
/// AiModelResult and never name a <c>HarmCategory</c>, so publishing it would freeze a mapping
/// that is expected to grow as detectors are added.
/// </remarks>
internal static class HarmTaxonomyMap
{
    /// <summary>
    /// Returns the taxonomy category a signal evidences, or <c>null</c> when the signal is not a
    /// content harm under this taxonomy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// EXCLUDED BY DESIGN, NOT BY OMISSION. A signal that maps to <c>null</c> raises no video-level
    /// finding, so an accidental omission is a silent safety regression. Fourteen
    /// <see cref="SafetyCategory"/> members are deliberately unmapped: thirteen are model-integrity
    /// or provenance signals rather than harms a viewer suffers from the CONTENT, and one is
    /// deployment-defined rather than a fixed harm.
    /// </para>
    /// <list type="bullet">
    /// <item><description>Prompt/model integrity: <c>PromptInjection</c>, <c>JailbreakAttempt</c>,
    /// <c>TrainingDataLeakage</c>, <c>ModelExtraction</c></description></item>
    /// <item><description>Provenance: <c>AIGenerated</c>, <c>Watermarked</c></description></item>
    /// <item><description>Legal/contractual: <c>CopyrightViolation</c>, <c>LegalAdvice</c></description></item>
    /// <item><description>Privacy/security of third parties: <c>PIIExposure</c>,
    /// <c>SurveillanceEnabling</c>, <c>Malware</c></description></item>
    /// <item><description>Process concerns: <c>Bias</c>, <c>TransparencyViolation</c></description></item>
    /// <item><description>Deployment-defined: <c>PolicyViolation</c>. Its own definition is
    /// "content that violates a topic restriction or CUSTOM policy rule", so what it means is set by
    /// the operator rather than by this taxonomy. Every fixed harm category would therefore be a
    /// guess: a custom policy may restrict cooking videos or share prices, neither of which is
    /// hate, sexual, physical, addictive, information or clickbait harm. Mapping it to any of the
    /// six would reproduce exactly the wrong-but-SPECIFIC label the removal of the catch-all arm
    /// below was meant to prevent, so it raises no video-level finding here and is reported by
    /// whichever module owns the custom policy.</description></item>
    /// </list>
    /// <para>
    /// The taxonomy explicitly scopes itself to harm "discernible from the content itself", which is
    /// why these are out. Any OTHER unmapped member is a bug: map it, or add it here with a reason.
    /// </para>
    /// </remarks>
    public static HarmCategory? ToHarmCategory(SafetyCategory category) => category switch
    {
        // "fake news, misinformation, disinformation, conspiracy theories, unverified medical
        // treatments, and unproven scientific myths"
        SafetyCategory.Misinformation => HarmCategory.Information,
        SafetyCategory.Disinformation => HarmCategory.Information,
        SafetyCategory.MedicalAdvice => HarmCategory.Information,
        SafetyCategory.Hallucination => HarmCategory.Information,
        SafetyCategory.Impersonation => HarmCategory.Information,
        SafetyCategory.Deepfake => HarmCategory.Information,
        SafetyCategory.Manipulated => HarmCategory.Information,

        // "insults and obscenities, identity attacks, and hate speech based on gender, race,
        // ethnicity, age, religion, political ideology, disability, or sexual orientation"
        SafetyCategory.HateSpeech => HarmCategory.HateAndHarassment,
        SafetyCategory.Harassment => HarmCategory.HateAndHarassment,
        SafetyCategory.Discrimination => HarmCategory.HateAndHarassment,
        SafetyCategory.Stereotyping => HarmCategory.HateAndHarassment,
        SafetyCategory.Dehumanization => HarmCategory.HateAndHarassment,
        SafetyCategory.Doxxing => HarmCategory.HateAndHarassment,

        // "excessive gaming, gambling, or substance use (drugs, smoking, or alcohol)"
        SafetyCategory.DrugManufacturing => HarmCategory.Addictive,

        // "exaggerated headlines intended to boost click rates, unverified financial schemes, and
        // sensational gossip or defamatory videos"
        SafetyCategory.Fraud => HarmCategory.Clickbait,
        SafetyCategory.SocialEngineering => HarmCategory.Clickbait,
        SafetyCategory.FinancialAdvice => HarmCategory.Clickbait,

        // "erotic scenes, depictions of sexual acts and nudity, and videos of sexual abuse"
        SafetyCategory.SexualExplicit => HarmCategory.Sexual,
        SafetyCategory.SexualSuggestive => HarmCategory.Sexual,
        SafetyCategory.SexualMinors => HarmCategory.Sexual,

        // "dangerous behaviors and graphic violence, including self-injury, suicide, eating
        // disorders, and dangerous challenges"
        SafetyCategory.ViolenceGraphic => HarmCategory.Physical,
        SafetyCategory.ViolenceThreat => HarmCategory.Physical,
        SafetyCategory.ViolenceWeapons => HarmCategory.Physical,
        SafetyCategory.ViolenceSelfHarm => HarmCategory.Physical,
        SafetyCategory.ViolenceSuicide => HarmCategory.Physical,
        SafetyCategory.ViolenceTerrorism => HarmCategory.Physical,
        SafetyCategory.IllegalActivities => HarmCategory.Physical,
        SafetyCategory.WeaponsInstructions => HarmCategory.Physical,

        _ => null,
    };

    /// <summary>
    /// A representative <see cref="SafetyCategory"/> to report a taxonomy-level finding under, so a
    /// harm category can be surfaced through the existing <c>SafetyFinding</c> contract.
    /// </summary>
    public static SafetyCategory RepresentativeSignal(HarmCategory harm) => harm switch
    {
        HarmCategory.Information => SafetyCategory.Misinformation,
        HarmCategory.HateAndHarassment => SafetyCategory.HateSpeech,
        HarmCategory.Addictive => SafetyCategory.DrugManufacturing,
        HarmCategory.Clickbait => SafetyCategory.Fraud,
        HarmCategory.Sexual => SafetyCategory.SexualExplicit,
        HarmCategory.Physical => SafetyCategory.ViolenceGraphic,

        // NO CATCH-ALL. This returned Misinformation for any unlisted HarmCategory, so a new
        // taxonomy member would have been reported under a wrong and SPECIFIC category -- a
        // mislabelled finding is harder to spot than a missing one, and reads as a real detection.
        // Throwing makes the omission fail during development instead.
        _ => throw new ArgumentOutOfRangeException(
            nameof(harm), harm,
            "No representative signal is registered for this harm category. Add one here when a "
            + "taxonomy member is added; defaulting would report the finding under the wrong category."),
    };
}
