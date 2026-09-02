using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>Binds program genomes, a fitness evaluator, and program descriptors into one evolution task.</summary>
/// <remarks>
/// <para>
/// The task supplies the three things <c>EvolutionEngine&lt;ProgramGenome&gt;</c> cannot know by itself: what a
/// valid candidate looks like, which candidates count as the same, and how good one is.
/// <see cref="CanonicalizeAsync"/> uses <see cref="ProgramGenome.Id"/>, the SHA-256 of the normalized source, so a
/// child that differs from its parent only in line endings or trailing white space is recognized as a duplicate
/// before any evaluation is paid for. <see cref="EvaluateAsync"/> enforces the configured program bounds, delegates
/// scoring to the <see cref="IProgramFitnessEvaluator"/>, and then merges the descriptor set's coordinates into the
/// result.
/// </para>
/// <para>
/// Descriptor merging follows the reference implementation's precedence: a coordinate the evaluator computed
/// itself always wins over a built-in descriptor of the same name, so an evaluator that measures real runtime can
/// override the static token-count proxy without any reconfiguration. A candidate longer than
/// <see cref="ProgramEvolutionOptions.MaxProgramChars"/>, or one whose evolve-block markers are damaged while
/// <see cref="ProgramEvolutionOptions.EnforceEvolveBlocks"/> is set, is returned as
/// <see cref="EvolutionEvaluationStatus.Rejected"/> without being executed at all.
/// </para>
/// <para><b>For Beginners:</b> This is the adapter that lets the generic evolution engine evolve source code. You
/// give it your scoring rule and, optionally, the pigeonhole axes you want the archive to use; it handles
/// recognizing duplicate programs, refusing candidates that broke the rules, and combining your score with the
/// descriptor coordinates. Everything about running the code stays in your evaluator, so nothing here executes
/// generated program text.</para>
/// </remarks>
public sealed class ProgramEvolutionTask : IEvolutionTask<ProgramGenome>
{
    private readonly IProgramFitnessEvaluator _evaluator;
    private readonly ProgramDescriptorSet _descriptors;
    private readonly ProgramEvolutionOptions _options;
    private readonly EvolveBlockMarkers _markers;

    /// <summary>Initializes a program-evolution task.</summary>
    /// <param name="evaluator">The evaluator that scores candidates.</param>
    /// <param name="descriptors">The descriptors that place candidates in the archive; <c>null</c> means none.</param>
    /// <param name="options">Program bounds and evolve-block settings; <c>null</c> uses the defaults.</param>
    /// <param name="id">A stable task identifier.</param>
    /// <exception cref="ArgumentNullException"><paramref name="evaluator"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> is empty or white space, or <paramref name="options"/> is invalid.</exception>
    public ProgramEvolutionTask(
        IProgramFitnessEvaluator evaluator,
        ProgramDescriptorSet? descriptors = null,
        ProgramEvolutionOptions? options = null,
        string id = "program-evolution")
    {
        Guard.NotNull(evaluator);
        Guard.NotNullOrWhiteSpace(id);

        ProgramEvolutionOptions effective = (options ?? new ProgramEvolutionOptions()).Clone();
        effective.Validate();

        _evaluator = evaluator;
        _descriptors = descriptors ?? ProgramDescriptorSet.Empty();
        _options = effective;
        _markers = effective.ResolveEvolveBlockMarkers();
        Id = id.Trim();
        VersionHash = BuildVersionHash(effective, _descriptors, evaluator);
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <inheritdoc/>
    public string EvaluatorVersionHash => _evaluator.VersionHash;

    /// <summary>Gets the descriptors merged into every completed evaluation.</summary>
    public ProgramDescriptorSet Descriptors => _descriptors;

    /// <summary>Gets a copy of the program bounds and evolve-block settings this task enforces.</summary>
    /// <returns>An independent copy; mutating it does not affect the task.</returns>
    public ProgramEvolutionOptions GetOptions() => _options.Clone();

    /// <inheritdoc/>
    public ValueTask<EvolutionCanonicalGenome<ProgramGenome>> CanonicalizeAsync(
        ProgramGenome genome,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(genome);
        cancellationToken.ThrowIfCancellationRequested();
        return new ValueTask<EvolutionCanonicalGenome<ProgramGenome>>(
            new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id));
    }

    /// <inheritdoc/>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(
        EvolutionCandidate<ProgramGenome> candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);

        ProgramGenome genome = candidate.CanonicalGenome.Genome;
        if (genome.NormalizedSource.Length > _options.MaxProgramChars)
        {
            return new EvolutionTaskResult(
                EvolutionEvaluationStatus.Rejected,
                diagnostics: new[]
                {
                    new EvolutionDiagnostic(
                        "program_too_long",
                        "The candidate is " +
                        genome.NormalizedSource.Length.ToString(CultureInfo.InvariantCulture) +
                        " characters, above the configured limit of " +
                        _options.MaxProgramChars.ToString(CultureInfo.InvariantCulture) + ".")
                });
        }

        if (_options.EnforceEvolveBlocks)
        {
            EvolveBlockExtractionResult extraction = EvolveBlock.Extract(genome.Source, _markers);
            if (!extraction.IsWellFormed || !extraction.HasRegions)
            {
                return new EvolutionTaskResult(
                    EvolutionEvaluationStatus.Rejected,
                    diagnostics: new[]
                    {
                        new EvolutionDiagnostic(
                            "program_evolve_block_invalid",
                            "Evolve blocks are enforced but the candidate's markers are " +
                            extraction.Status.ToString() + ".")
                    });
            }
        }

        EvolutionTaskResult result = await _evaluator
            .EvaluateAsync(genome, context, cancellationToken)
            .ConfigureAwait(false);

        if (result is null)
        {
            return EvolutionTaskResult.Failed(
                "program_evaluator_returned_null", "The fitness evaluator returned no result.");
        }

        if (_descriptors.Count == 0 || result.Status != EvolutionEvaluationStatus.Completed) return result;

        var merged = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, double> pair in _descriptors.Compute(genome)) merged[pair.Key] = pair.Value;
        foreach (KeyValuePair<string, double> pair in result.Descriptors) merged[pair.Key] = pair.Value;

        return new EvolutionTaskResult(
            result.Status,
            result.Quality,
            result.Direction,
            merged,
            result.Objectives,
            result.ConstraintViolations,
            result.CostUnits,
            result.Diagnostics);
    }

    private static string BuildVersionHash(
        ProgramEvolutionOptions options,
        ProgramDescriptorSet descriptors,
        IProgramFitnessEvaluator evaluator)
    {
        EvolveBlockMarkers markers = options.ResolveEvolveBlockMarkers();
        var components = new List<string>
        {
            "program-evolution-task-v2",
            options.Language.ToString(),
            markers.ToString(),
            options.EnforceEvolveBlocks ? "enforce" : "free",
            options.MaxProgramChars.ToString(CultureInfo.InvariantCulture),
            descriptors.VersionHash,

            // The evaluator's own version travels separately as EvaluatorVersionHash, but its identity belongs in
            // the task hash too: swapping one evaluator for another with the same version must not silently pass a
            // checkpoint-compatibility check.
            evaluator.Id
        };

        return "program-task-" + EvolutionHash.Combine(components);
    }
}
