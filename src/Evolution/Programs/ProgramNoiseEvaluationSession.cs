using AiDotNet.Configuration;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution.Programs;

/// <summary>Runs correctness-gated fresh program screening, automatic rejection audits and incumbent challenges.</summary>
/// <remarks>Keep this controller and its confirmation evaluator outside the proposing process. All backends must
/// perform fresh work and own isolation/reset semantics; no cache decorator or engine is used. The ledger is shared,
/// so do not additionally meter these callbacks. A report authorizes neither deployment nor archive mutation.</remarks>
public sealed class ProgramNoiseEvaluationSession
{
    private readonly ProgramNoiseEvaluationOptions _options;
    private readonly EvolutionResourceLedger _ledger;
    private readonly EvolutionReplicateRunner<ProgramGenome> _screen;
    private readonly EvolutionIncumbentChallenge<ProgramGenome> _challenge;
    private readonly EvolutionRejectionAudit<ProgramGenome> _audit;

    /// <summary>Creates a version-pinned session. Public correctness and hidden confirmation correctness are separate backends.</summary>
    public ProgramNoiseEvaluationSession(ProgramNoiseEvaluationOptions options, EvolutionResourceLedger ledger,
        IProgramFitnessEvaluator correctness, IProgramFitnessEvaluator screen, IProgramFitnessEvaluator full,
        IProgramFitnessEvaluator confirmationCorrectness, IProgramFitnessEvaluator confirmation)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = options;
        _ledger = ledger;
        var cheap = Gate(correctness, screen);
        var search = Gate(correctness, full);
        var hidden = Gate(confirmationCorrectness, confirmation);
        var plan = options.ScreenPlan;
        string screenVersion = EvolutionHash.Combine(new[] { cheap.VersionHash, EvolutionHash.EncodeDouble(options.ScreenThreshold) });
        _screen = new(screenVersion, plan, ledger, (g, c, t) => cheap.EvaluateAsync(g, c.EvaluationContext, t));
        _challenge = new(search.VersionHash, hidden.VersionHash, ledger, options.SearchSamples, options.ConfirmationSamples,
            options.MaximumChallenges, plan.MinimumQuality, plan.MaximumQuality, options.MinimumImprovement,
            plan.MaximumCostPerSample, (g, c, t) => search.EvaluateAsync(g, c.EvaluationContext, t),
            (g, c, t) => hidden.EvaluateAsync(g, c.EvaluationContext, t), options.Confidence, plan.Direction);
        _audit = new(hidden.VersionHash, ledger, options.AuditCandidates, options.ConfirmationSamples,
            plan.MinimumQuality, plan.MaximumQuality, options.UsefulThreshold, plan.MaximumCostPerSample,
            (g, c, t) => hidden.EvaluateAsync(g, c.EvaluationContext, t), options.Confidence, plan.Direction);
        VersionHash = EvolutionHash.Combine(new[] { "program-noise-session-v1", _screen.VersionHash, _challenge.VersionHash, _audit.VersionHash });
    }

    /// <summary>Gets the complete frozen evaluator and policy identity.</summary>
    public string VersionHash { get; }

    /// <summary>Screens a bounded frozen population, then audits the preselected rejects before returning.</summary>
    /// <remarks>Choose auditSeed independently before screening; never retry a batch under a new identity to shop for outcomes.
    /// Incomplete screens are retained separately and are not evidence of a successful rejection.</remarks>
    public async ValueTask<ProgramNoiseScreenReport> ScreenAndAuditAsync(string batchId, IEnumerable<ProgramGenome> candidates,
        ulong searchSeed, ulong auditSeed, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(batchId) || batchId.Length > 128 || batchId.Any(char.IsControl))
            throw new ArgumentException("Declare a bounded printable batch identity.", nameof(batchId));
        if (candidates is null) throw new ArgumentNullException(nameof(candidates));
        var population = candidates.Take(129).ToArray();
        if (population.Length is < 1 or > 128 || population.Any(g => g is null) || population.Select(g => g.Id).Distinct().Count() != population.Length)
            throw new ArgumentException("Require 1..128 distinct programs.", nameof(candidates));
        cancellationToken.ThrowIfCancellationRequested();
        string operation = EvolutionHash.Combine(new[] { VersionHash, batchId });
        using (var claim = _ledger.TryReserve("program-noise-screen/" + operation, EvolutionResourceStage.Setup,
            EvolutionResources.Empty, EvolutionResources.Empty) ?? throw new EvolutionResourceBudgetException(operation))
            claim.Complete(EvolutionResources.Empty);
        var rows = new List<ProgramNoiseScreenEntry>();
        foreach (var genome in population.OrderBy(g => g.Id, StringComparer.Ordinal))
        {
            if (cancellationToken.IsCancellationRequested) break;
            var result = await _screen.RunAsync(new(genome, genome.Id), new EvolutionEvaluationContext(0, searchSeed, 0, 1),
                batchId, cancellationToken: cancellationToken).ConfigureAwait(false);
            bool passed = result.IsComplete && (_options.ScreenPlan.Direction == EvolutionOptimizationDirection.Maximize
                ? result.MeanQuality >= _options.ScreenThreshold : result.MeanQuality <= _options.ScreenThreshold);
            rows.Add(new(genome, result, passed));
            if (cancellationToken.IsCancellationRequested) break;
        }
        var rejected = rows.Where(row => row.Measurements.IsComplete && !row.Passed)
            .Select(row => new EvolutionCanonicalGenome<ProgramGenome>(row.Genome, row.Genome.Id)).ToArray();
        EvolutionRejectionAuditReport? audit = null;
        if (rejected.Length > 0 && !cancellationToken.IsCancellationRequested)
            audit = await _audit.RunAsync(batchId, rejected, auditSeed, cancellationToken).ConfigureAwait(false);
        return new(population.Length, rows, audit, rejected.Length == 0 || audit is not null, cancellationToken.IsCancellationRequested);
    }

    /// <summary>Requests fresh full-fidelity incumbent/challenger search followed by independent confirmation.</summary>
    public ValueTask<EvolutionIncumbentChallengeReport> ChallengeAsync(int slot, ProgramGenome candidate, ProgramGenome incumbent,
        ulong seed, CancellationToken cancellationToken = default)
    {
        if (candidate is null) throw new ArgumentNullException(nameof(candidate));
        if (incumbent is null) throw new ArgumentNullException(nameof(incumbent));
        return _challenge.RunAsync(slot, new(candidate, candidate.Id), new(incumbent, incumbent.Id),
            new EvolutionEvaluationContext(0, seed, 0, 1), cancellationToken);
    }

    private static IProgramFitnessEvaluator Gate(IProgramFitnessEvaluator correctness, IProgramFitnessEvaluator fitness) =>
        new CorrectnessGatedProgramFitnessEvaluator(Fresh(correctness), Fresh(fitness));

    private static IProgramFitnessEvaluator Fresh(IProgramFitnessEvaluator evaluator)
    {
        var pinned = new VersionPinnedProgramFitnessEvaluator(evaluator);
        return new DelegateProgramFitnessEvaluator(async (g, c, t) =>
        {
            var result = await pinned.EvaluateAsync(g, c, t).ConfigureAwait(false);
            if (result is null) throw new InvalidOperationException("Fresh evaluator returned no receipt.");
            // Correctness metadata would otherwise be dropped when merging fitness: reject reused checks here too.
            if (result.MeasurementOrigin is not null)
                throw new InvalidOperationException("Noise session requires raw fresh receipts, not cached or preaggregated evidence.");
            return result;
        }, versionHash: pinned.VersionHash);
    }
}

/// <summary>One cheap-screen result, including its fresh measurement and cost evidence.</summary>
public sealed class ProgramNoiseScreenEntry
{
    internal ProgramNoiseScreenEntry(ProgramGenome genome, EvolutionReplicationReport measurements, bool passed)
    { Genome = genome; Measurements = measurements; Passed = passed; }
    /// <summary>Gets the exact immutable program screened.</summary>
    public ProgramGenome Genome { get; }
    /// <summary>Gets the fresh cheap-screen batch, including incomplete results.</summary>
    public EvolutionReplicationReport Measurements { get; }
    /// <summary>Gets whether a complete screen met the threshold, not permission to promote.</summary>
    public bool Passed { get; }
}

/// <summary>Screen decisions and separately retained full-fidelity rejection evidence.</summary>
public sealed class ProgramNoiseScreenReport
{
    internal ProgramNoiseScreenReport(int requested, IEnumerable<ProgramNoiseScreenEntry> entries, EvolutionRejectionAuditReport? audit, bool auditDispatched, bool canceled)
    { Requested = requested; Entries = Array.AsReadOnly(entries.ToArray()); Audit = audit; AuditDispatched = auditDispatched; Canceled = canceled; }
    /// <summary>Gets the frozen requested population size, including programs not reached after cancellation.</summary>
    public int Requested { get; }
    /// <summary>Gets all dispatched screens.</summary>
    public IReadOnlyList<ProgramNoiseScreenEntry> Entries { get; }
    /// <summary>Gets rejection evidence; null if no complete rejects existed or cancellation prevented auditing.</summary>
    public EvolutionRejectionAuditReport? Audit { get; }
    /// <summary>Gets whether an audit was dispatched when required; completion must also be checked.</summary>
    public bool AuditDispatched { get; }
    /// <summary>Gets whether cancellation prevented a successful complete workflow.</summary>
    public bool Canceled { get; }
    /// <summary>Gets whether every screen and required audit completed, without classifying inconclusive bounds as success.</summary>
    public bool IsComplete => !Canceled && Entries.Count == Requested && Entries.All(row => row.Measurements.IsComplete) && AuditDispatched &&
        (Audit is null || Audit.Entries.All(row => row.FullEvaluation?.IsComplete == true));
    /// <summary>Gets all reported screen and audit work, including conservative unknown charges.</summary>
    public decimal ChargedCostUnits => Entries.Sum(row => row.Measurements.ChargedCostUnits) + (Audit?.ChargedCostUnits ?? 0);
}
