namespace AiDotNet.Enums;

/// <summary>Names one short, overridable phrase that prompt sections are assembled from.</summary>
/// <remarks>
/// <para>
/// Fragments are the sentence-sized pieces a prompt builder chooses between: whether fitness improved or
/// declined, what kind of example a program is, why an artifact section is present. Keeping them separate from
/// the templates in <see cref="ProgramPromptTemplateKey"/> means wording can be adjusted or translated without
/// touching the structure of the prompt.
/// </para>
/// <para>
/// The reference OpenEvolve implementation ships forty-one fragments in a JSON file, roughly ten of which no code
/// path ever reads, and at least one of which (<c>inspiration_changes_prefix</c>) is formatted with a
/// <c>{changes}</c> argument its text does not contain, so the change description is silently dropped from every
/// prompt. This enumeration lists only fragments that are actually rendered, and every fragment's placeholders are
/// validated when the template set is built, so that class of silent loss cannot occur.
/// </para>
/// <para><b>For Beginners:</b> Think of these as the stock phrases in a form letter. The letter's layout comes
/// from a template; the phrases are the parts that change depending on the situation, such as "Fitness improved"
/// versus "Fitness declined". You can replace any phrase with your own wording. Some phrases have blanks in them,
/// written as <c>{name}</c>, which are filled in with real numbers when the prompt is built; if your replacement
/// leaves out a blank the phrase needs, you are told right away.</para>
/// </remarks>
public enum ProgramPromptFragmentKey
{
    /// <summary>Reports that fitness rose against the previous attempt. Placeholders: <c>previous</c>, <c>current</c>.</summary>
    FitnessImproved = 0,

    /// <summary>Reports that fitness fell against the previous attempt. Placeholders: <c>previous</c>, <c>current</c>.</summary>
    FitnessDeclined = 1,

    /// <summary>Reports that fitness is unchanged. Placeholder: <c>current</c>.</summary>
    FitnessStable = 2,

    /// <summary>Names the region of the feature space being explored. Placeholder: <c>features</c>.</summary>
    ExploringRegion = 3,

    /// <summary>States that no feature coordinates are available.</summary>
    NoFeatureCoordinates = 4,

    /// <summary>Suggests simplification once a program grows past a threshold. Placeholder: <c>threshold</c>.</summary>
    CodeTooLong = 5,

    /// <summary>The fallback guidance used when nothing more specific applies.</summary>
    NoSpecificGuidance = 6,

    /// <summary>Points at nearby unoccupied archive cells worth reaching. Placeholder: <c>cells</c>.</summary>
    CoverageHint = 7,

    /// <summary>Stands in for a missing description of what an attempt changed.</summary>
    AttemptUnknownChanges = 8,

    /// <summary>Describes an attempt whose measured values all improved.</summary>
    AttemptAllMetricsImproved = 9,

    /// <summary>Describes an attempt whose measured values all regressed.</summary>
    AttemptAllMetricsRegressed = 10,

    /// <summary>Describes an attempt with both improvements and regressions.</summary>
    AttemptMixedMetrics = 11,

    /// <summary>Introduces the strong measurements of a top program.</summary>
    TopProgramMetricsPrefix = 12,

    /// <summary>The heading above the diverse-programs section.</summary>
    DiverseProgramsTitle = 13,

    /// <summary>Introduces the measurements of a diverse program.</summary>
    DiverseProgramMetricsPrefix = 14,

    /// <summary>Labels an inspiration chosen for being different from the parent.</summary>
    InspirationTypeDiverse = 15,

    /// <summary>Labels an inspiration that arrived from another island.</summary>
    InspirationTypeMigrant = 16,

    /// <summary>Labels an inspiration drawn at random.</summary>
    InspirationTypeRandom = 17,

    /// <summary>Labels an inspiration in the highest score band.</summary>
    InspirationTypeHighPerformer = 18,

    /// <summary>Labels an inspiration in the upper-middle score band.</summary>
    InspirationTypeAlternative = 19,

    /// <summary>Labels an inspiration in the lower-middle score band.</summary>
    InspirationTypeExperimental = 20,

    /// <summary>Labels an inspiration in the lowest score band.</summary>
    InspirationTypeExploratory = 21,

    /// <summary>Introduces an inspiration's change description. Placeholder: <c>changes</c>.</summary>
    InspirationChangesPrefix = 22,

    /// <summary>Highlights an unusually strong measurement. Placeholders: <c>name</c>, <c>value</c>.</summary>
    InspirationMetricsExcellent = 23,

    /// <summary>Highlights a measurement where the program takes a different route. Placeholder: <c>name</c>.</summary>
    InspirationMetricsAlternative = 24,

    /// <summary>The fallback description of an inspiration with no standout features. Placeholder: <c>type</c>.</summary>
    InspirationNoFeatures = 25,

    /// <summary>The heading above the execution-output section.</summary>
    ArtifactTitle = 26,

    /// <summary>Marks where an over-long artifact was cut. Placeholder: <c>bytes</c>.</summary>
    ArtifactTruncated = 27,

    /// <summary>Labels an example short enough to count as a concise implementation.</summary>
    InspirationConciseImplementation = 28,

    /// <summary>Labels an example long enough to count as a comprehensive implementation.</summary>
    InspirationComprehensiveImplementation = 29
}
