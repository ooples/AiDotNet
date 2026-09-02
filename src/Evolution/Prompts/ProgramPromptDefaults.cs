using AiDotNet.Enums;

namespace AiDotNet.Evolution.Prompts;

/// <summary>
/// The prompt texts, phrases, and placeholder contracts that ship with the library, compiled into the assembly
/// so they can never be missing or shadowed by a stale file on disk.
/// </summary>
/// <remarks>
/// Every template declares two lists: the placeholders the prompt builder always supplies to it, and the
/// placeholders it may not drop without breaking the section it wraps. Every fragment declares the exact
/// arguments it is given, which must all appear in its text. Those declarations are what makes configure-time
/// validation possible.
/// </remarks>
internal static class ProgramPromptDefaults
{
    private static readonly string[] Empty = Array.Empty<string>();

    private static readonly string[] UserVariables =
    {
        "metrics", "fitness_score", "feature_coords", "feature_dimensions", "improvement_areas",
        "evolution_history", "current_program", "language", "artifacts", "task_description",
        "diagnostics", "search_marker", "divider_marker", "replace_marker", "evolve_block_instructions"
    };

    private static readonly string[] SystemVariables = { "language", "feature_dimensions", "task_description" };
    private static readonly string[] EvaluatorSystemVariables = { "language", "task_description" };
    private static readonly string[] HistoryVariables = { "previous_attempts", "top_programs", "inspirations_section" };
    private static readonly string[] AttemptVariables = { "attempt_number", "changes", "performance", "outcome" };
    private static readonly string[] TopProgramVariables = { "program_number", "score", "language", "program_snippet", "key_features" };
    private static readonly string[] InspirationsSectionVariables = { "inspiration_programs" };
    private static readonly string[] InspirationProgramVariables = { "program_number", "score", "program_type", "language", "program_snippet", "unique_features" };
    private static readonly string[] EvaluationVariables = { "current_program", "language", "criteria", "response_schema" };
    private static readonly string[] ChangesWrapperVariables = { "system_message", "system_message_changes_description" };
    private static readonly string[] UserChangesWrapperVariables = { "user_message", "changes_description" };

    private static readonly string[] RequiredCurrentProgram = { "current_program" };
    private static readonly string[] RequiredTopPrograms = { "top_programs" };
    private static readonly string[] RequiredInspirationPrograms = { "inspiration_programs" };
    private static readonly string[] RequiredProgramSnippet = { "program_snippet" };
    private static readonly string[] RequiredSystemMessage = { "system_message" };
    private static readonly string[] RequiredUserMessageAndChanges = { "user_message", "changes_description" };

    /// <summary>Gets the shipped text for a template key.</summary>
    internal static string TemplateText(ProgramPromptTemplateKey key) => key switch
    {
        ProgramPromptTemplateKey.SystemMessage => SystemMessageText,
        ProgramPromptTemplateKey.EvaluatorSystemMessage => EvaluatorSystemMessageText,
        ProgramPromptTemplateKey.DiffUser => DiffUserText,
        ProgramPromptTemplateKey.FullRewriteUser => FullRewriteUserText,
        ProgramPromptTemplateKey.EvolutionHistory => EvolutionHistoryText,
        ProgramPromptTemplateKey.PreviousAttempt => PreviousAttemptText,
        ProgramPromptTemplateKey.TopProgram => TopProgramText,
        ProgramPromptTemplateKey.InspirationsSection => InspirationsSectionText,
        ProgramPromptTemplateKey.InspirationProgram => InspirationProgramText,
        ProgramPromptTemplateKey.Evaluation => EvaluationText,
        ProgramPromptTemplateKey.SystemMessageChangesDescription => SystemMessageChangesDescriptionText,
        ProgramPromptTemplateKey.SystemMessageWithChangesDescription => SystemMessageWithChangesDescriptionText,
        ProgramPromptTemplateKey.UserMessageWithChangesDescription => UserMessageWithChangesDescriptionText,
        _ => throw new ArgumentException($"'{key}' is not a defined prompt template key.", nameof(key))
    };

    /// <summary>Gets the file stem a template override is read from.</summary>
    internal static string TemplateFileStem(ProgramPromptTemplateKey key) => key switch
    {
        ProgramPromptTemplateKey.SystemMessage => "system_message",
        ProgramPromptTemplateKey.EvaluatorSystemMessage => "evaluator_system_message",
        ProgramPromptTemplateKey.DiffUser => "diff_user",
        ProgramPromptTemplateKey.FullRewriteUser => "full_rewrite_user",
        ProgramPromptTemplateKey.EvolutionHistory => "evolution_history",
        ProgramPromptTemplateKey.PreviousAttempt => "previous_attempt",
        ProgramPromptTemplateKey.TopProgram => "top_program",
        ProgramPromptTemplateKey.InspirationsSection => "inspirations_section",
        ProgramPromptTemplateKey.InspirationProgram => "inspiration_program",
        ProgramPromptTemplateKey.Evaluation => "evaluation",
        ProgramPromptTemplateKey.SystemMessageChangesDescription => "system_message_changes_description",
        ProgramPromptTemplateKey.SystemMessageWithChangesDescription => "system_message_with_changes_description",
        ProgramPromptTemplateKey.UserMessageWithChangesDescription => "user_message_with_changes_description",
        _ => throw new ArgumentException($"'{key}' is not a defined prompt template key.", nameof(key))
    };

    /// <summary>Gets the placeholders the prompt builder always supplies to a template.</summary>
    internal static IReadOnlyList<string> SuppliedPlaceholders(ProgramPromptTemplateKey key) => key switch
    {
        ProgramPromptTemplateKey.SystemMessage => SystemVariables,
        ProgramPromptTemplateKey.EvaluatorSystemMessage => EvaluatorSystemVariables,
        ProgramPromptTemplateKey.DiffUser => UserVariables,
        ProgramPromptTemplateKey.FullRewriteUser => UserVariables,
        ProgramPromptTemplateKey.EvolutionHistory => HistoryVariables,
        ProgramPromptTemplateKey.PreviousAttempt => AttemptVariables,
        ProgramPromptTemplateKey.TopProgram => TopProgramVariables,
        ProgramPromptTemplateKey.InspirationsSection => InspirationsSectionVariables,
        ProgramPromptTemplateKey.InspirationProgram => InspirationProgramVariables,
        ProgramPromptTemplateKey.Evaluation => EvaluationVariables,
        ProgramPromptTemplateKey.SystemMessageChangesDescription => Empty,
        ProgramPromptTemplateKey.SystemMessageWithChangesDescription => ChangesWrapperVariables,
        ProgramPromptTemplateKey.UserMessageWithChangesDescription => UserChangesWrapperVariables,
        _ => throw new ArgumentException($"'{key}' is not a defined prompt template key.", nameof(key))
    };

    /// <summary>Gets the placeholders a template may not drop without breaking the section it wraps.</summary>
    internal static IReadOnlyList<string> RequiredPlaceholders(ProgramPromptTemplateKey key) => key switch
    {
        ProgramPromptTemplateKey.SystemMessage => Empty,
        ProgramPromptTemplateKey.EvaluatorSystemMessage => Empty,
        ProgramPromptTemplateKey.DiffUser => RequiredCurrentProgram,
        ProgramPromptTemplateKey.FullRewriteUser => RequiredCurrentProgram,
        ProgramPromptTemplateKey.EvolutionHistory => RequiredTopPrograms,
        ProgramPromptTemplateKey.PreviousAttempt => Empty,
        ProgramPromptTemplateKey.TopProgram => RequiredProgramSnippet,
        ProgramPromptTemplateKey.InspirationsSection => RequiredInspirationPrograms,
        ProgramPromptTemplateKey.InspirationProgram => RequiredProgramSnippet,
        ProgramPromptTemplateKey.Evaluation => RequiredCurrentProgram,
        ProgramPromptTemplateKey.SystemMessageChangesDescription => Empty,
        ProgramPromptTemplateKey.SystemMessageWithChangesDescription => RequiredSystemMessage,
        ProgramPromptTemplateKey.UserMessageWithChangesDescription => RequiredUserMessageAndChanges,
        _ => throw new ArgumentException($"'{key}' is not a defined prompt template key.", nameof(key))
    };

    /// <summary>Gets the shipped text for a phrase key.</summary>
    internal static string FragmentText(ProgramPromptFragmentKey key) => key switch
    {
        ProgramPromptFragmentKey.FitnessImproved => "Fitness improved: {previous} -> {current}",
        ProgramPromptFragmentKey.FitnessDeclined => "Fitness declined: {previous} -> {current}. Consider revising the most recent changes.",
        ProgramPromptFragmentKey.FitnessStable => "Fitness unchanged at {current}",
        ProgramPromptFragmentKey.ExploringRegion => "Exploring the {features} region of the solution space",
        ProgramPromptFragmentKey.NoFeatureCoordinates => "No feature coordinates are available",
        ProgramPromptFragmentKey.CodeTooLong => "Consider simplifying: the program is longer than {threshold} characters",
        ProgramPromptFragmentKey.NoSpecificGuidance => "Focus on improving fitness while keeping the solution distinctive",
        ProgramPromptFragmentKey.CoverageHint => "These nearby archive cells are still empty and worth reaching: {cells}",
        ProgramPromptFragmentKey.AttemptUnknownChanges => "Unknown changes",
        ProgramPromptFragmentKey.AttemptAllMetricsImproved => "Every measurement improved",
        ProgramPromptFragmentKey.AttemptAllMetricsRegressed => "Every measurement regressed",
        ProgramPromptFragmentKey.AttemptMixedMetrics => "Mixed results",
        ProgramPromptFragmentKey.TopProgramMetricsPrefix => "Strong on",
        ProgramPromptFragmentKey.DiverseProgramsTitle => "Diverse Programs",
        ProgramPromptFragmentKey.DiverseProgramMetricsPrefix => "Alternative approach to",
        ProgramPromptFragmentKey.InspirationTypeDiverse => "Diverse",
        ProgramPromptFragmentKey.InspirationTypeMigrant => "Migrant",
        ProgramPromptFragmentKey.InspirationTypeRandom => "Random",
        ProgramPromptFragmentKey.InspirationTypeHighPerformer => "High-Performer",
        ProgramPromptFragmentKey.InspirationTypeAlternative => "Alternative",
        ProgramPromptFragmentKey.InspirationTypeExperimental => "Experimental",
        ProgramPromptFragmentKey.InspirationTypeExploratory => "Exploratory",
        ProgramPromptFragmentKey.InspirationChangesPrefix => "Modification: {changes}",
        ProgramPromptFragmentKey.InspirationMetricsExcellent => "Excellent {name} ({value})",
        ProgramPromptFragmentKey.InspirationMetricsAlternative => "Alternative {name} approach",
        ProgramPromptFragmentKey.InspirationNoFeatures => "{type} approach to the problem",
        ProgramPromptFragmentKey.ArtifactTitle => "Last Execution Output",
        ProgramPromptFragmentKey.ArtifactTruncated => "... truncated at {bytes} bytes",
        ProgramPromptFragmentKey.InspirationConciseImplementation => "Concise implementation",
        ProgramPromptFragmentKey.InspirationComprehensiveImplementation => "Comprehensive implementation",
        _ => throw new ArgumentException($"'{key}' is not a defined prompt fragment key.", nameof(key))
    };

    /// <summary>Gets the name a phrase override is read under inside the fragments file.</summary>
    internal static string FragmentName(ProgramPromptFragmentKey key) => key switch
    {
        ProgramPromptFragmentKey.FitnessImproved => "fitness_improved",
        ProgramPromptFragmentKey.FitnessDeclined => "fitness_declined",
        ProgramPromptFragmentKey.FitnessStable => "fitness_stable",
        ProgramPromptFragmentKey.ExploringRegion => "exploring_region",
        ProgramPromptFragmentKey.NoFeatureCoordinates => "no_feature_coordinates",
        ProgramPromptFragmentKey.CodeTooLong => "code_too_long",
        ProgramPromptFragmentKey.NoSpecificGuidance => "no_specific_guidance",
        ProgramPromptFragmentKey.CoverageHint => "coverage_hint",
        ProgramPromptFragmentKey.AttemptUnknownChanges => "attempt_unknown_changes",
        ProgramPromptFragmentKey.AttemptAllMetricsImproved => "attempt_all_metrics_improved",
        ProgramPromptFragmentKey.AttemptAllMetricsRegressed => "attempt_all_metrics_regressed",
        ProgramPromptFragmentKey.AttemptMixedMetrics => "attempt_mixed_metrics",
        ProgramPromptFragmentKey.TopProgramMetricsPrefix => "top_program_metrics_prefix",
        ProgramPromptFragmentKey.DiverseProgramsTitle => "diverse_programs_title",
        ProgramPromptFragmentKey.DiverseProgramMetricsPrefix => "diverse_program_metrics_prefix",
        ProgramPromptFragmentKey.InspirationTypeDiverse => "inspiration_type_diverse",
        ProgramPromptFragmentKey.InspirationTypeMigrant => "inspiration_type_migrant",
        ProgramPromptFragmentKey.InspirationTypeRandom => "inspiration_type_random",
        ProgramPromptFragmentKey.InspirationTypeHighPerformer => "inspiration_type_high_performer",
        ProgramPromptFragmentKey.InspirationTypeAlternative => "inspiration_type_alternative",
        ProgramPromptFragmentKey.InspirationTypeExperimental => "inspiration_type_experimental",
        ProgramPromptFragmentKey.InspirationTypeExploratory => "inspiration_type_exploratory",
        ProgramPromptFragmentKey.InspirationChangesPrefix => "inspiration_changes_prefix",
        ProgramPromptFragmentKey.InspirationMetricsExcellent => "inspiration_metrics_excellent",
        ProgramPromptFragmentKey.InspirationMetricsAlternative => "inspiration_metrics_alternative",
        ProgramPromptFragmentKey.InspirationNoFeatures => "inspiration_no_features",
        ProgramPromptFragmentKey.ArtifactTitle => "artifact_title",
        ProgramPromptFragmentKey.ArtifactTruncated => "artifact_truncated",
        ProgramPromptFragmentKey.InspirationConciseImplementation => "inspiration_concise_implementation",
        ProgramPromptFragmentKey.InspirationComprehensiveImplementation => "inspiration_comprehensive_implementation",
        _ => throw new ArgumentException($"'{key}' is not a defined prompt fragment key.", nameof(key))
    };

    /// <summary>Gets the exact arguments a phrase is given, all of which its text must use.</summary>
    internal static IReadOnlyList<string> FragmentArguments(ProgramPromptFragmentKey key) => key switch
    {
        ProgramPromptFragmentKey.FitnessImproved => PreviousAndCurrent,
        ProgramPromptFragmentKey.FitnessDeclined => PreviousAndCurrent,
        ProgramPromptFragmentKey.FitnessStable => CurrentOnly,
        ProgramPromptFragmentKey.ExploringRegion => FeaturesOnly,
        ProgramPromptFragmentKey.NoFeatureCoordinates => Empty,
        ProgramPromptFragmentKey.CodeTooLong => ThresholdOnly,
        ProgramPromptFragmentKey.NoSpecificGuidance => Empty,
        ProgramPromptFragmentKey.CoverageHint => CellsOnly,
        ProgramPromptFragmentKey.AttemptUnknownChanges => Empty,
        ProgramPromptFragmentKey.AttemptAllMetricsImproved => Empty,
        ProgramPromptFragmentKey.AttemptAllMetricsRegressed => Empty,
        ProgramPromptFragmentKey.AttemptMixedMetrics => Empty,
        ProgramPromptFragmentKey.TopProgramMetricsPrefix => Empty,
        ProgramPromptFragmentKey.DiverseProgramsTitle => Empty,
        ProgramPromptFragmentKey.DiverseProgramMetricsPrefix => Empty,
        ProgramPromptFragmentKey.InspirationTypeDiverse => Empty,
        ProgramPromptFragmentKey.InspirationTypeMigrant => Empty,
        ProgramPromptFragmentKey.InspirationTypeRandom => Empty,
        ProgramPromptFragmentKey.InspirationTypeHighPerformer => Empty,
        ProgramPromptFragmentKey.InspirationTypeAlternative => Empty,
        ProgramPromptFragmentKey.InspirationTypeExperimental => Empty,
        ProgramPromptFragmentKey.InspirationTypeExploratory => Empty,
        ProgramPromptFragmentKey.InspirationChangesPrefix => ChangesOnly,
        ProgramPromptFragmentKey.InspirationMetricsExcellent => NameAndValue,
        ProgramPromptFragmentKey.InspirationMetricsAlternative => NameOnly,
        ProgramPromptFragmentKey.InspirationNoFeatures => TypeOnly,
        ProgramPromptFragmentKey.ArtifactTitle => Empty,
        ProgramPromptFragmentKey.ArtifactTruncated => BytesOnly,
        ProgramPromptFragmentKey.InspirationConciseImplementation => Empty,
        ProgramPromptFragmentKey.InspirationComprehensiveImplementation => Empty,
        _ => throw new ArgumentException($"'{key}' is not a defined prompt fragment key.", nameof(key))
    };

    private static readonly string[] PreviousAndCurrent = { "previous", "current" };
    private static readonly string[] CurrentOnly = { "current" };
    private static readonly string[] FeaturesOnly = { "features" };
    private static readonly string[] ThresholdOnly = { "threshold" };
    private static readonly string[] CellsOnly = { "cells" };
    private static readonly string[] ChangesOnly = { "changes" };
    private static readonly string[] NameAndValue = { "name", "value" };
    private static readonly string[] NameOnly = { "name" };
    private static readonly string[] TypeOnly = { "type" };
    private static readonly string[] BytesOnly = { "bytes" };

    private const string SystemMessageText = """
        You are an expert software engineer improving a program in small, verifiable steps.
        Your goal is to raise the FITNESS SCORE while keeping the population of programs diverse across the feature dimensions the search tracks.
        A higher score and a genuinely different approach are both valuable outcomes.
        Reply in the requested format only, with no commentary outside it.
        """;

    private const string EvaluatorSystemMessageText = """
        You are an expert code reviewer.
        Analyse the program you are given and score it systematically against the stated criteria.
        Reply with the requested JSON object only, with no commentary outside it.
        """;

    private const string DiffUserText = """
        # Current Program Information
        - Fitness: {fitness_score}
        - Feature coordinates: {feature_coords}
        - Focus areas: {improvement_areas}
        - Measurements: {metrics}
        {task_description}
        {diagnostics}
        {artifacts}

        # Program Evolution History
        {evolution_history}

        # Current Program
        ```{language}
        {current_program}
        ```

        # Task
        Suggest improvements to the program that raise its FITNESS SCORE.
        The search keeps solutions diverse across these dimensions: {feature_dimensions}
        Solutions with similar fitness but different feature coordinates are valuable.
        {evolve_block_instructions}
        You MUST express every change as an edit block in exactly this form:

        {search_marker}
        the exact lines that are in the program now
        {divider_marker}
        the lines that replace them
        {replace_marker}

        Each search section must match the current program exactly, including indentation and blank lines.
        You may supply several blocks. Put your reasoning before the blocks, never inside them.

        IMPORTANT: do not rewrite the whole program; make targeted changes.
        """;

    private const string FullRewriteUserText = """
        # Current Program Information
        - Fitness: {fitness_score}
        - Feature coordinates: {feature_coords}
        - Focus areas: {improvement_areas}
        - Measurements: {metrics}
        {task_description}
        {diagnostics}
        {artifacts}

        # Program Evolution History
        {evolution_history}

        # Current Program
        ```{language}
        {current_program}
        ```

        # Task
        Rewrite the program so that it reaches a higher FITNESS SCORE.
        The search keeps solutions diverse across these dimensions: {feature_dimensions}
        Solutions with similar fitness but different feature coordinates are valuable.
        {evolve_block_instructions}
        Keep the same inputs and outputs as the current program, and improve the implementation behind them.
        Return the complete new program inside a single fenced block:

        ```{language}
        your rewritten program here
        ```
        """;

    private const string EvolutionHistoryText = """
        ## Previous Attempts

        {previous_attempts}

        ## Top Performing Programs

        {top_programs}

        {inspirations_section}
        """;

    private const string PreviousAttemptText = """
        ### Attempt {attempt_number}
        - Changes: {changes}
        - Measurements: {performance}
        - Outcome: {outcome}
        """;

    private const string TopProgramText = """
        ### Program {program_number} (fitness {score})
        ```{language}
        {program_snippet}
        ```
        Key features: {key_features}
        """;

    private const string InspirationsSectionText = """
        ## Inspiration Programs

        These programs take different approaches and may suggest ideas worth borrowing:

        {inspiration_programs}
        """;

    private const string InspirationProgramText = """
        ### Inspiration {program_number} (fitness {score}, type {program_type})
        ```{language}
        {program_snippet}
        ```
        Unique approach: {unique_features}
        """;

    private const string EvaluationText = """
        Score the program below against each of these criteria, from 0.0 (worst) to 1.0 (best):
        {criteria}

        Program under review:
        ```{language}
        {current_program}
        ```

        Return exactly one JSON object and nothing else, matching this shape:
        {response_schema}
        """;

    private const string SystemMessageChangesDescriptionText = """
        Important: keep the "Changes Description" block up to date. Replace its previous contents in the same answer, using the same edit format as the rest of your changes. If you skip this step your changes are discarded, because the search depends on that description.
        Write the description in plain language as a short numbered list; another model, or you in a later round, has to read and update it.
        """;

    private const string SystemMessageWithChangesDescriptionText = """
        {system_message}

        {system_message_changes_description}
        """;

    private const string UserMessageWithChangesDescriptionText = """
        {user_message}


        # Changes Description
        Update the contents of the block below in this same answer, or every change you propose is discarded.
        ```text
        {changes_description}
        ```
        """;
}
