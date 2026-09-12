using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>Validates setup and freshly checks the first seed before requesting any model proposals.</summary>
    /// <param name="maximumInputOutputCases">Separate preflight bound on public test cases, between 1 and 4096.</param>
    /// <param name="cancellationToken">Cancellation honored by the configured evaluator/runner.</param>
    /// <returns>Separate preflight evidence; this work does not consume the search's MaxEvaluationAttempts.</returns>
    /// <remarks>
    /// Executes at most one correctness evaluation and one additional fitness evaluation. Built-in runners retain
    /// their configured per-dispatch limits; opaque caller-owned providers must enforce their own resource limits.
    /// No proposal/chat request is made here, but caller-supplied evaluators may themselves use external services.
    /// Correctness requires a maximize-one result with no violations or declared measurement reuse, not merely
    /// Completed. Script/custom fitness requires explicit correctness or public input/output examples.
    /// Resource-accounted/persistent-fitness preflight needs coordinated reservations and is refused for now.
    /// Resume file integrity is checked; full resume compatibility remains the engine's responsibility before search.
    /// </remarks>
    public async Task<ProgramEvolutionPreflightResult> PreflightProgramEvolutionAsync(
        int maximumInputOutputCases = 256, CancellationToken cancellationToken = default)
    {
        if (maximumInputOutputCases is < 1 or > 4096) throw new ArgumentOutOfRangeException(nameof(maximumInputOutputCases));
        cancellationToken.ThrowIfCancellationRequested();
        var programs = (_programEvolutionOptions ?? throw new InvalidOperationException("ConfigureProgramEvolution has not been called.")).Clone();
        if (_evolutionSeedOptions is { } seeds)
            for (int index = 0; index < seeds.ProgramSources.Count; index++) programs.SeedPrograms.Insert(index, seeds.ProgramSources[index]);
        programs.Validate();
        if (programs.CustomVariation is null && _chatClient is null)
            throw new InvalidOperationException("Program evolution requires a configured chat client or custom variation operator.");
        if (programs.ResourceAccounting is not null || programs.CustomFitnessEvaluator is PersistentProgramFitnessEvaluator)
            throw new NotSupportedException("Resource-accounted and persistent-fitness preflight requires coordinated preflight reservations; it is not yet supported.");
        if (programs.TestCases.Count > maximumInputOutputCases)
            throw new ArgumentException("The public test set exceeds the separate preflight test-case budget.");
        var runOptions = ResolveProgramEvolutionOptions(programs);
        if (runOptions.MaxEvaluationAttempts <= 0 || runOptions.MaxProposals <= 0 || runOptions.MaxGenerations <= 0)
            throw new ArgumentException("Preflight requires positive search evaluation, proposal and generation budgets.");
        var genomes = programs.CreateSeedGenomes();
        if (genomes.Count == 0) throw new ArgumentException("Preflight requires at least one seed program.");
        ProgramGenome seed = genomes[0];
        var report = new ProgramEvolutionPreflightResult { SeedGenomeId = seed.Id, InputOutputCases = programs.TestCases.Count };
        var locations = ResolveEvolutionLocations(runOptions);
        if (runOptions.Resume)
        {
            if (locations.CheckpointPath is null ||
                await new JsonEvolutionCheckpointStore(locations.CheckpointPath).LoadLatestAsync(runOptions.RunId, cancellationToken).ConfigureAwait(false) is null)
                throw new ArgumentException("Resume requires an existing integrity-checked checkpoint for this run.");
        }
        foreach (string directory in new[] { locations.OutputDirectory, Path.GetDirectoryName(locations.CheckpointPath),
                     Path.GetDirectoryName(locations.TracePath) }.OfType<string>().Distinct(StringComparer.Ordinal))
        {
            Directory.CreateDirectory(directory);
            string probe = Path.Combine(directory, ".preflight-" + Guid.NewGuid().ToString("N"));
            using var stream = new FileStream(probe, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1, FileOptions.DeleteOnClose);
            stream.WriteByte(0);
            stream.Flush(flushToDisk: true);
        }
        report.OutputLocationsChecked = true;
        ProcessProgramExecutionEngine? owned = null;
        ProcessProgramExecutionEngine? correctnessOwned = null;
        try
        {
            IProgramFitnessEvaluator fitness = CreateProgramEvaluator(programs, out owned);
            IProgramFitnessEvaluator? correctness = _programCorrectnessEvaluator is { } configured
                ? new VersionPinnedProgramFitnessEvaluator(configured) : fitness as SandboxedProgramFitnessEvaluator;
            if (correctness is null && programs.TestCases.Count > 0)
            {
                IProgramExecutionEngine? runner = _programExecutionEngine ?? owned;
                if (runner is null)
                {
                    if (_programSandboxOptions is not null && programs.HasExplicitSandbox)
                        throw new ArgumentException("Configure the program sandbox in one place, not twice.");
                    correctnessOwned = new ProcessProgramExecutionEngine(_programSandboxOptions ?? programs.Sandbox);
                    runner = correctnessOwned;
                }
                correctness = new SandboxedProgramFitnessEvaluator(runner, programs.TestCases);
            }
            if (correctness is null) { report.Code = "correctness_not_configured"; return report; }
            report.CorrectnessIdentity = EvolutionHash.Combine(new[] { correctness.Id, correctness.VersionHash });
            report.FitnessIdentity = EvolutionHash.Combine(new[] { fitness.Id, fitness.VersionHash });
            var lineage = new EvolutionLineage(null, null, "preflight", null, 0, 0, runOptions.Seed);
            var descriptors = programs.CreateDescriptorSet();
            var correctnessTask = new ProgramEvolutionTask(correctness, descriptors, programs);
            var canonical = await correctnessTask.CanonicalizeAsync(seed, cancellationToken).ConfigureAwait(false);
            var candidate = new EvolutionCandidate<ProgramGenome>(0, canonical, lineage);
            var context = new EvolutionEvaluationContext(0, runOptions.Seed, 0, 1);
            EvolutionTaskResult validation = await correctnessTask.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
            report.CorrectnessStatus = validation.Status;
            report.CorrectnessCostUnits = validation.CostUnits;
            if (validation.Status != EvolutionEvaluationStatus.Completed || validation.Quality != 1 ||
                validation.Direction != EvolutionOptimizationDirection.Maximize || validation.ConstraintViolations.Any(value => value > 0) ||
                validation.MeasurementOrigin is { Kind: not EvolutionMeasurementOriginKind.Measured })
            { report.Code = "seed_correctness_failed"; return report; }
            cancellationToken.ThrowIfCancellationRequested();
            report.SharedCorrectnessAndFitness = ReferenceEquals(correctness, fitness);
            var fitnessTask = report.SharedCorrectnessAndFitness ? correctnessTask : new ProgramEvolutionTask(fitness, descriptors, programs);
            EvolutionTaskResult measured = validation;
            if (!report.SharedCorrectnessAndFitness)
            {
                measured = await fitnessTask.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
                report.AdditionalFitnessStatus = measured.Status;
                report.AdditionalFitnessCostUnits = measured.CostUnits;
            }
            var evaluation = new EvolutionEvaluation(0, canonical.Id, measured.Status, measured.Quality, measured.Direction,
                measured.Descriptors, measured.Objectives, measured.ConstraintViolations,
                new EvolutionEvaluationCost(TimeSpan.Zero, 1, measured.CostUnits), lineage, EvolutionCacheStatus.NotChecked,
                Array.Empty<EvolutionDiagnostic>(), fitnessTask.VersionHash, fitnessTask.EvaluatorVersionHash,
                EvolutionHash.Compute("preflight-not-a-search-checkpoint"), measured.Metrics);
            report.IsReady = runOptions.CreateArchive<ProgramGenome>().TryAdd(candidate, evaluation) == EvolutionArchiveInsertionResult.Inserted;
            report.Code = report.IsReady ? "ready" : "seed_fitness_or_archive_rejected";
            return report;
        }
        finally { correctnessOwned?.Dispose(); owned?.Dispose(); }
    }
}
