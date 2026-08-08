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
public static class HarmTaxonomyMap
{
    /// <summary>
    /// Returns the taxonomy category a signal evidences, or <c>null</c> when the signal is not a
    /// content harm under this taxonomy.
    /// </summary>
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
        _ => SafetyCategory.Misinformation,
    };
}
