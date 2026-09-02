using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs.Novelty;

/// <summary>Consults a novelty policy before an inner evaluator is allowed to spend anything on a candidate.</summary>
/// <remarks>
/// <para>
/// This is how a novelty gate reaches a run without any change to the evolution engine or the archive: it is an
/// <see cref="IProgramFitnessEvaluator"/> that wraps the real one. A candidate the policy rejects is returned as
/// <see cref="EvolutionEvaluationStatus.Rejected"/> with a bounded diagnostic and the inner evaluator is never
/// called, so the saving is the whole cost of an evaluation — a sandboxed run, a test suite, a judging call —
/// rather than merely an archive slot.
/// </para>
/// <para>
/// That ordering is the substantive difference from the reference implementation, which gates on insertion: there,
/// a candidate is proposed, evaluated in full, and only then compared for novelty and dropped, so a rejected
/// duplicate has already consumed its evaluation. It is also why the check must be cheap, which is exactly what the
/// policy's structural rung is.
/// </para>
/// <para>
/// The gate remembers the genomes it has accepted, newest last, up to
/// <see cref="EmbeddingNoveltyOptions.MaxTrackedGenomes"/>, then discards the oldest. That memory is its own: it is
/// not the engine's archive and cannot see elites this evaluator did not admit, migrants, or anything restored from
/// a checkpoint. An engine-side gate consulting the live archive would compare against strictly more, and would
/// need the engine to call the policy itself.
/// </para>
/// <para><b>For Beginners:</b> Wrap your scoring evaluator in this and near-duplicate candidates stop costing you
/// anything: they are turned away before your tests ever run. It keeps a list of the candidates it has let through
/// and compares each new one against that list. <see cref="RejectedCount"/> tells you how many evaluations it
/// saved.</para>
/// </remarks>
public sealed class NoveltyGatingProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    /// <summary>The diagnostic code reported for a candidate the policy turned away.</summary>
    public const string RejectionCode = "program_not_novel";

    private readonly IProgramFitnessEvaluator _inner;
    private readonly ProgramNoveltyPolicy _policy;
    private readonly List<ProgramGenome> _accepted = new();
    private readonly HashSet<string> _acceptedIds = new(StringComparer.Ordinal);
    private readonly object _gate = new();

    private long _acceptedCount;
    private long _rejectedCount;
    private ProgramNoveltyDecision? _lastDecision;

    /// <summary>Initializes a gating evaluator.</summary>
    /// <param name="inner">The evaluator that scores candidates the policy admits.</param>
    /// <param name="policy">The novelty policy consulted first; <c>null</c> uses a free structural policy.</param>
    /// <param name="id">A stable evaluator identifier.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inner"/> or <paramref name="id"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="id"/> is empty or white space.</exception>
    public NoveltyGatingProgramFitnessEvaluator(
        IProgramFitnessEvaluator inner,
        ProgramNoveltyPolicy? policy = null,
        string id = "novelty-gating-program-evaluator")
    {
        Guard.NotNull(inner);
        Guard.NotNullOrWhiteSpace(id);

        _inner = inner;
        _policy = policy ?? new ProgramNoveltyPolicy();
        Id = id.Trim();
        VersionHash = "novelty-gate-" + EvolutionHash.Combine(new[]
        {
            "novelty-gating-program-evaluator-v1",
            inner.Id,
            inner.VersionHash,
            _policy.VersionHash
        });
    }

    /// <inheritdoc/>
    public string Id { get; }

    /// <inheritdoc/>
    public string VersionHash { get; }

    /// <summary>Gets the evaluator that scores candidates the policy admits.</summary>
    public IProgramFitnessEvaluator Inner => _inner;

    /// <summary>Gets the policy consulted before every evaluation.</summary>
    public ProgramNoveltyPolicy Policy => _policy;

    /// <summary>Gets how many candidates were admitted and forwarded to the inner evaluator.</summary>
    public long AcceptedCount => Interlocked.Read(ref _acceptedCount);

    /// <summary>Gets how many candidates were turned away without any inner evaluation.</summary>
    public long RejectedCount => Interlocked.Read(ref _rejectedCount);

    /// <summary>Gets how many accepted genomes are currently remembered.</summary>
    public int TrackedCount
    {
        get
        {
            lock (_gate) return _accepted.Count;
        }
    }

    /// <summary>Returns the most recent novelty decision, or <c>null</c> before the first evaluation.</summary>
    /// <returns>The last decision this gate produced.</returns>
    public ProgramNoveltyDecision? GetLastDecision()
    {
        lock (_gate) return _lastDecision;
    }

    /// <summary>Adds a genome to the remembered set without evaluating it.</summary>
    /// <param name="genome">The genome to remember, typically a seed the run started from.</param>
    /// <exception cref="ArgumentNullException"><paramref name="genome"/> is <c>null</c>.</exception>
    public void Remember(ProgramGenome genome)
    {
        Guard.NotNull(genome);
        lock (_gate) Track(genome);
    }

    /// <inheritdoc/>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(
        ProgramGenome candidate,
        EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidate);
        Guard.NotNull(context);

        ProgramGenome[] known;
        lock (_gate) known = _accepted.ToArray();

        ProgramNoveltyDecision decision = await _policy
            .EvaluateAsync(candidate, known, cancellationToken)
            .ConfigureAwait(false);

        lock (_gate) _lastDecision = decision;

        if (!decision.IsNovel)
        {
            Interlocked.Increment(ref _rejectedCount);
            return new EvolutionTaskResult(
                EvolutionEvaluationStatus.Rejected,
                diagnostics: new[] { new EvolutionDiagnostic(RejectionCode, BuildRejectionMessage(decision)) });
        }

        Interlocked.Increment(ref _acceptedCount);
        lock (_gate) Track(candidate);

        return await _inner.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
    }

    private static string BuildRejectionMessage(ProgramNoveltyDecision decision) =>
        "The candidate was turned away by the novelty gate at the " + decision.DecidedBy +
        " stage: " + decision.Reason + ". Structural comparisons: " +
        decision.StructuralComparisons.ToString(CultureInfo.InvariantCulture) +
        ", embedding requests: " + decision.EmbeddingRequests.ToString(CultureInfo.InvariantCulture) +
        ", judging requests: " + decision.JudgeRequests.ToString(CultureInfo.InvariantCulture) + ".";

    private void Track(ProgramGenome genome)
    {
        if (!_acceptedIds.Add(genome.Id)) return;
        _accepted.Add(genome);
        while (_accepted.Count > _policy.Options.MaxTrackedGenomes)
        {
            _acceptedIds.Remove(_accepted[0].Id);
            _accepted.RemoveAt(0);
        }
    }
}
