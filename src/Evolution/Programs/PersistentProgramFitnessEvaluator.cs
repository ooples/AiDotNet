using System.Globalization;
using System.IO;
using System.Text;
using AiDotNet.Interfaces;

namespace AiDotNet.Evolution.Programs;

/// <summary>Reuses explicitly eligible program-fitness samples while leaving current correctness checks outside the cache.</summary>
/// <remarks>
/// Supply this decorator as ProgramEvolutionOptions.CustomFitnessEvaluator and disable the engine's
/// EnableEvaluationCache. The facade then runs correctness before every fitness lookup. The scope identifies
/// the inner fitness domain: TaskId is inner.Id; TaskVersion and EvaluatorVersion are inner.VersionHash;
/// codec facets identify ProgramGenomeCodec. All other scope facets are explicit caller declarations.
/// The producer must supply fresh MeasurementOrigin with this scope; missing/reused evidence is never relabeled.
/// Each run requires a unique runId and a ledger declaring cache_store_invocations and program_evidence_invocations.
/// Neither logical counter measures physical I/O or evaluator cost_units. Clock and evidence callbacks are
/// caller-owned, thread-safe and bounded. This does not attest hardware, isolation or stochastic stationarity.
/// </remarks>
public sealed class PersistentProgramFitnessEvaluator : IProgramFitnessEvaluator
{
    /// <summary>The resource counting dispatched raw-evidence store methods, including failures.</summary>
    public const string EvidenceInvocationResource = "program_evidence_invocations";
    private static readonly EvolutionResources OneEvidenceCall = EvolutionResources.Of(EvidenceInvocationResource, 1);
    private static readonly UTF8Encoding Utf8 = new(false, true);
    private readonly IProgramFitnessEvaluator _inner;
    private readonly IProgramMeasurementEvidenceStore _evidence;
    private readonly EvolutionPersistentEvaluationCache _cache;
    private readonly EvolutionResourceLedger _ledger;
    private readonly ProgramGenomeCodec _codec = new();
    private readonly string _innerId, _innerVersion, _evidenceVersion, _measurementVersion, _runId;
    private readonly Func<DateTimeOffset> _utcNow;
    private readonly bool _forceFresh;

    /// <summary>Creates explicit, caller-owned persistent fitness reuse; no work or store calls occur here.</summary>
    public PersistentProgramFitnessEvaluator(IProgramFitnessEvaluator inner, IEvolutionEvaluationStore store,
        IProgramMeasurementEvidenceStore evidence, EvolutionReuseScope scope, EvolutionEvaluationReusePolicy policy,
        EvolutionResourceLedger ledger, string measurementVersion, string runId, bool forceFresh = false,
        Func<DateTimeOffset>? utcNow = null)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _evidence = evidence ?? throw new ArgumentNullException(nameof(evidence));
        Scope = scope ?? throw new ArgumentNullException(nameof(scope));
        _ledger = ledger ?? throw new ArgumentNullException(nameof(ledger));
        _innerId = inner.Id; _innerVersion = inner.VersionHash; _evidenceVersion = evidence.VersionHash;
        foreach (string value in new[] { _innerId, _innerVersion, _evidenceVersion, measurementVersion, runId })
            VersionPinnedProgramFitnessEvaluator.ValidateIdentity(value, nameof(value));
        if (scope.TaskId != _innerId || scope.TaskVersion != _innerVersion || scope.EvaluatorVersion != _innerVersion ||
            scope.CodecId != _codec.Id || scope.CodecVersion != _codec.VersionHash)
            throw new ArgumentException("The reuse scope must identify the current inner fitness domain and program codec.", nameof(scope));
        if (!ledger.Limits.Amounts.ContainsKey(EvidenceInvocationResource))
            throw new ArgumentException("The ledger must declare program_evidence_invocations.", nameof(ledger));
        _cache = new EvolutionPersistentEvaluationCache(store, policy, ledger);
        _measurementVersion = EvolutionHash.Combine(new[] { measurementVersion, _evidenceVersion });
        _runId = runId; _forceFresh = forceFresh; _utcNow = utcNow ?? (() => DateTimeOffset.UtcNow);
        VersionHash = EvolutionHash.Combine(new[] { "persistent-program-fitness-v1", _innerId, _innerVersion,
            scope.StableKey, _cache.VersionHash, _measurementVersion, forceFresh ? "fresh" : "reuse" });
    }

    /// <inheritdoc/>
    public string Id => "persistent-program-fitness";
    /// <inheritdoc/>
    public string VersionHash { get; }
    /// <summary>Gets the explicit fitness-domain applicability scope required on fresh producer evidence.</summary>
    public EvolutionReuseScope Scope { get; }

    /// <inheritdoc/>
    public async ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
        CancellationToken cancellationToken = default)
    {
        if (candidate is null) throw new ArgumentNullException(nameof(candidate));
        if (context is null) throw new ArgumentNullException(nameof(context));
        cancellationToken.ThrowIfCancellationRequested(); CheckIdentity();
        if (_cache.Policy.Mode == EvolutionEvaluationReuseMode.Disabled)
            return await EvaluateFreshAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        string payload = _codec.Serialize(candidate);
        // Oversized valid programs remain evaluable, but are deliberately ineligible for bounded persistence.
        if (Utf8.GetByteCount(payload) > EvolutionRepertoire.MaximumPayloadBytes)
            return await EvaluateFreshAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        ProgramGenome decoded = _codec.Deserialize(payload);
        if (decoded.Id != candidate.Id || _codec.Serialize(decoded) != payload)
            throw new InvalidOperationException("Program fitness reuse requires an exact canonical codec round trip.");
        var key = new EvolutionEvaluationCacheKey(Scope, candidate.Id, EvolutionHash.Compute(payload), _measurementVersion);
        string operationId = EvolutionHash.Combine(new[] { _runId, context.EvaluationId.ToString(CultureInfo.InvariantCulture),
            context.AttemptCount.ToString(CultureInfo.InvariantCulture) });
        DateTimeOffset lookupTime = _utcNow();
        CheckIdentity();
        var lookup = await _cache.LookupAsync(key, operationId, lookupTime, _forceFresh, cancellationToken).ConfigureAwait(false);
        CheckIdentity();
        if (lookup.ReusedResult is { } reused && await EvidenceCallAsync(key, operationId, "verify",
            () => _evidence.VerifyAsync(candidate, reused, lookup.EvidenceSha256!, context, cancellationToken), false,
            cancellationToken).ConfigureAwait(false))
        {
            // A slow evidence verifier must not extend the original acquisition's lifetime.
            DateTimeOffset acceptedAt = _utcNow();
            CheckIdentity(); cancellationToken.ThrowIfCancellationRequested();
            if (acceptedAt.Offset != TimeSpan.Zero) throw new ArgumentException("Declare the reuse clock in UTC.");
            TimeSpan age = acceptedAt - reused.MeasurementOrigin!.ObservedAt;
            if (age >= TimeSpan.Zero && age <= _cache.Policy.MaximumAge) return reused;
        }

        EvolutionTaskResult measured = await EvaluateFreshAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        if (measured.Status != EvolutionEvaluationStatus.Completed ||
            measured.ConstraintViolations.Any(value => value > 0) || measured.MeasurementOrigin is not { } origin ||
            origin.Kind != EvolutionMeasurementOriginKind.Measured || origin.ScopeKey != Scope.StableKey)
            return measured;
        string? digest = await EvidenceCallAsync(key, operationId, "retain",
            () => _evidence.RetainAsync(candidate, measured, context, cancellationToken), (string?)null,
            cancellationToken).ConfigureAwait(false);
        if (digest is null) return measured;
        // Invalid provider metadata declines persistence without discarding the already paid fresh result.
        EvolutionEvaluationCacheRecord record;
        try { record = new EvolutionEvaluationCacheRecord(key, measured, digest); }
        catch (ArgumentException) { return measured; }
        await _cache.TryStoreAsync(record, operationId, cancellationToken).ConfigureAwait(false);
        CheckIdentity();
        return measured;
    }

    private async ValueTask<EvolutionTaskResult> EvaluateFreshAsync(ProgramGenome candidate, EvolutionEvaluationContext context,
        CancellationToken cancellationToken)
    {
        CheckIdentity(); cancellationToken.ThrowIfCancellationRequested();
        var result = await _inner.EvaluateAsync(candidate, context, cancellationToken).ConfigureAwait(false);
        CheckIdentity(); cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    private async ValueTask<TResult> EvidenceCallAsync<TResult>(EvolutionEvaluationCacheKey key, string operationId, string action,
        Func<ValueTask<TResult>> call, TResult unavailable, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested(); CheckIdentity();
        using EvolutionResourceReservation? reservation = _ledger.TryReserve(
            "program-evidence/" + EvolutionHash.Combine(new[] { operationId, key.StableKey, action }),
            EvolutionResourceStage.Persistence, OneEvidenceCall, OneEvidenceCall);
        if (reservation is null) return unavailable;
        EvolutionResourceOutcome outcome = EvolutionResourceOutcome.Failed;
        try
        {
            TResult result = await call().ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested(); CheckIdentity();
            outcome = EvolutionResourceOutcome.Completed;
            return result;
        }
        catch (OperationCanceledException) { outcome = EvolutionResourceOutcome.Canceled; throw; }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or FormatException)
        { return unavailable; }
        finally { reservation.Complete(OneEvidenceCall, outcome); }
    }

    private void CheckIdentity()
    {
        if (_inner.Id != _innerId || _inner.VersionHash != _innerVersion || _evidence.VersionHash != _evidenceVersion)
            throw new InvalidOperationException("Program fitness or raw-evidence semantics changed during persistent reuse.");
    }
}
