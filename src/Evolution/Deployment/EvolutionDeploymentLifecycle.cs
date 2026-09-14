using System.Text.Json;
using JsonSerializer = System.Text.Json.JsonSerializer;
using JsonException = System.Text.Json.JsonException;

namespace AiDotNet.Evolution.Deployment;

/// <summary>Explicit program/model promotion, batch-boundary fallback, bounded retuning and monitored rollback.</summary>
/// <remarks>
/// No hidden worker or paid provider is started. Applications drain retuning through an idle gate and dispatch through
/// returned selections. In-flight operations are not revoked. Abandoned validation/search never publishes; its capacity
/// remains occupied until it settles. The short file-commit phase is synchronous and not forcibly interrupted.
/// </remarks>
public sealed class EvolutionDeploymentLifecycle
{
    private readonly object _gate = new();
    private readonly SemaphoreSlim _operation = new(1, 1);
    private readonly EvolutionDeploymentArtifactRegistry _registry;
    private readonly EvolutionDeploymentPolicy _policy;
    private readonly Func<EvolutionDeploymentEnvelope, EvolutionDeployableArtifact> _fallback;
    private readonly Func<EvolutionDeployableArtifact, int, CancellationToken, ValueTask<EvolutionDeploymentMeasurement>> _evaluate;
    private EvolutionDeploymentEnvelope _observed;
    private EvolutionDeploymentEnvelope? _pending;
    private EvolutionDeployableArtifact? _cached;
    private long _epoch;
    private int _retunes;
    private DateTimeOffset? _lastRetune, _lastObservation;
    private string? _monitoredRevision;
    private readonly List<MonitoringWindow> _windows = new();
    private bool _storageFaulted;

    /// <summary>Creates an application-owned lifecycle with an independent held-out evaluator and valid per-envelope fallback.</summary>
    public EvolutionDeploymentLifecycle(EvolutionDeploymentArtifactRegistry registry, EvolutionDeploymentEnvelope initialEnvelope,
        EvolutionDeploymentPolicy policy, Func<EvolutionDeploymentEnvelope, EvolutionDeployableArtifact> knownValidFallback,
        Func<EvolutionDeployableArtifact, int, CancellationToken, ValueTask<EvolutionDeploymentMeasurement>> evaluator)
    {
        _registry = registry ?? throw new ArgumentNullException(nameof(registry));
        _observed = initialEnvelope ?? throw new ArgumentNullException(nameof(initialEnvelope));
        _policy = policy ?? throw new ArgumentNullException(nameof(policy));
        _fallback = knownValidFallback ?? throw new ArgumentNullException(nameof(knownValidFallback));
        _evaluate = evaluator ?? throw new ArgumentNullException(nameof(evaluator));
    }
    /// <summary>Gets lifetime retuning admissions, including canceled, failed and abandoned work.</summary>
    public int AdmittedRetunes { get { lock (_gate) return _retunes; } }
    /// <summary>Gets whether one coalesced revalidation/retuning request is pending.</summary>
    public bool RetuneRequested { get { lock (_gate) return _pending is not null; } }
    /// <summary>Gets whether unreadable/corrupt storage latched this controller onto its fallback.</summary>
    public bool StorageFaulted { get { lock (_gate) return _storageFaulted; } }

    /// <summary>Refreshes applicability and persistent quarantine at a controlled dispatch boundary.</summary>
    public EvolutionDeploymentSelection Select(EvolutionDeploymentEnvelope observed)
    {
        if (observed is null) throw new ArgumentNullException(nameof(observed));
        long epoch;
        lock (_gate)
        {
            if (_observed.Key != observed.Key) { _observed = observed; _epoch++; }
            epoch = _epoch;
        }
        EvolutionDeploymentSelection selection;
        try
        {
            bool faulted;
            lock (_gate) faulted = _storageFaulted;
            selection = faulted || !_policy.AllowBestEffortPersistence ? Fallback(observed, null) : Select(_registry.ReadSlot(), observed);
        }
        catch (Exception error) when (StorageFailure(error) || error is ArgumentException)
        {
            lock (_gate) _storageFaulted = true;
            selection = Fallback(observed, null);
        }
        lock (_gate)
        {
            if (_epoch == epoch)
            {
                _pending = selection.IsFallback && !_storageFaulted && _policy.AllowBestEffortPersistence ? observed : null;
                return selection;
            }
        }
        return Fallback(observed, null);
    }

    /// <summary>Runs a fixed, fresh paired comparison and atomically activates only an explicitly authorized improvement.</summary>
    public Task<EvolutionDeploymentDecision> PromoteAsync(EvolutionDeployableArtifact candidate, CancellationToken cancellationToken = default)
    {
        if (candidate is null) throw new ArgumentNullException(nameof(candidate));
        long epoch;
        lock (_gate) epoch = _epoch;
        return ExecuteAsync(token => PrepareAsync(candidate, epoch, token), cancellationToken);
    }

    /// <summary>Drains one coalesced request through explicit idle admission and a bounded private search driver.</summary>
    /// <remarks>Use the first-party program/AutoML retuners to enforce engine caps; custom drivers must enforce the supplied request internally.</remarks>
    public Task<EvolutionDeploymentDecision> RetunePendingAsync(
        Func<EvolutionDeploymentRetuneRequest, CancellationToken, Task<EvolutionDeployableArtifact>> retuner,
        Func<CancellationToken, Task> idleGate, CancellationToken cancellationToken = default)
    {
        if (retuner is null) throw new ArgumentNullException(nameof(retuner));
        if (idleGate is null) throw new ArgumentNullException(nameof(idleGate));
        return ExecuteAsync(async token =>
        {
            EvolutionDeploymentEnvelope requested;
            long epoch;
            lock (_gate)
            {
                if (_pending is null || _storageFaulted) return new Prepared("NotRequested");
                var now = DateTimeOffset.UtcNow;
                if (_retunes >= _policy.MaximumRetunes || _lastRetune is { } last && now - last < _policy.Cooldown)
                    return new Prepared("BudgetDenied");
                requested = _pending; _pending = null; epoch = _epoch; _retunes++; _lastRetune = now;
            }
            await idleGate(token).ConfigureAwait(false);
            token.ThrowIfCancellationRequested();
            var candidate = await retuner(new EvolutionDeploymentRetuneRequest(requested, _policy), token).ConfigureAwait(false)
                ?? throw new InvalidOperationException("Retuner returned no deployable artifact.");
            token.ThrowIfCancellationRequested();
            if (candidate.Envelope.Key != requested.Key) throw new InvalidDataException("Retuner changed the admitted applicability envelope.");
            return await PrepareAsync(candidate, epoch, token).ConfigureAwait(false);
        }, cancellationToken);
    }

    private async Task<EvolutionDeploymentDecision> ExecuteAsync(Func<CancellationToken, Task<Prepared>> prepare, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!_policy.AllowBestEffortPersistence) return new("PersistencePolicyDenied", false);
        if (!await _operation.WaitAsync(0, cancellationToken).ConfigureAwait(false)) return new("Busy", false);
        var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Task<Prepared>? work = null;
        bool deferred = false;
        try
        {
            deadline.CancelAfter(_policy.Timeout);
            work = Task.Run(() => prepare(deadline.Token), deadline.Token);
            using var wait = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            wait.CancelAfter(_policy.Timeout + _policy.GracePeriod);
            try
            {
                if (await Task.WhenAny(work, Task.Delay(System.Threading.Timeout.Infinite, wait.Token)).ConfigureAwait(false) != work)
                { cancellationToken.ThrowIfCancellationRequested(); return new("Abandoned", false); }
                Prepared result = await work.ConfigureAwait(false);
                deadline.Token.ThrowIfCancellationRequested();
                if (result.Outcome != "Approved") return new(result.Outcome, false, result.Candidate?.Id, result.EvidenceId);
                var candidate = result.Candidate ?? throw new InvalidOperationException("Missing prepared artifact.");
                lock (_gate)
                {
                    if (_epoch != result.Epoch || _observed.Key != candidate.Envelope.Key || _storageFaulted)
                        return new("Stale", false, candidate.Id, result.EvidenceId);
                    deadline.Token.ThrowIfCancellationRequested();
                    if (!_registry.TryPromote(result.Revision, candidate, result.Incumbent!, result.EvidenceId!))
                        return new("Stale", false, candidate.Id, result.EvidenceId);
                    _cached = candidate; _pending = null; _windows.Clear(); _monitoredRevision = null;
                    return new("Promoted", true, candidate.Id, result.EvidenceId);
                }
            }
            finally { wait.Cancel(); }
        }
        finally
        {
            if (work is { IsCompleted: false } pending)
            {
                deferred = true;
                _ = pending.ContinueWith(finished => { _ = finished.Exception; deadline.Dispose(); _operation.Release(); },
                    CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);
            }
            if (!deferred) { _ = work?.Exception; deadline.Dispose(); _operation.Release(); }
        }
    }

    private async Task<Prepared> PrepareAsync(EvolutionDeployableArtifact candidate, long epoch, CancellationToken token)
    {
        lock (_gate)
            if (_epoch != epoch || _observed.Key != candidate.Envelope.Key || _storageFaulted) return new("Stale");
        var slot = _registry.ReadSlot();
        var incumbent = Select(slot, candidate.Envelope).Artifact;
        if (candidate.Kind != incumbent.Kind) throw new InvalidDataException("Candidate and incumbent use different deployment kinds.");
        if (_registry.IsQuarantined(candidate.Id) || _registry.IsQuarantined(incumbent.Id)) return new("Quarantined");
        _registry.Stage(candidate); _registry.Stage(incumbent);
        var pairs = new DeploymentMeasurementPair[_policy.PairedSamples];
        var observations = new List<object>(2 * _policy.PairedSamples);
        async Task<EvolutionDeploymentMeasurement> Measure(EvolutionDeployableArtifact artifact, int index)
        {
            token.ThrowIfCancellationRequested();
            var measurement = await _evaluate(artifact, index, token).ConfigureAwait(false)
                ?? throw new InvalidOperationException("Deployment evaluator returned no measurement or cost receipt.");
            observations.Add(new { ArtifactId = artifact.Id, PairIndex = index, measurement.Direction, Measurement = DeploymentRawMeasurement.From(measurement) });
            token.ThrowIfCancellationRequested();
            return measurement;
        }
        for (int i = 0; i < pairs.Length; i++)
        {
            // Alternate order without changing the shared pair index supplied to the independent protocol.
            var first = await Measure(i % 2 == 0 ? candidate : incumbent, i).ConfigureAwait(false);
            if (!Usable(first)) return Rejected("InvalidValidation");
            var second = await Measure(i % 2 == 0 ? incumbent : candidate, i).ConfigureAwait(false);
            if (!Usable(second)) return Rejected("InvalidValidation");
            pairs[i] = new DeploymentMeasurementPair
            { Candidate = DeploymentRawMeasurement.From(i % 2 == 0 ? first : second), Incumbent = DeploymentRawMeasurement.From(i % 2 == 0 ? second : first) };
        }
        var evidence = new DeploymentValidationEvidence
        {
            CandidateId = candidate.Id, IncumbentId = incumbent.Id, EnvelopeKey = candidate.Envelope.Key,
            Direction = _policy.Direction, MinimumMeanGain = _policy.MinimumMeanGain, MaximumPValue = _policy.MaximumPValue,
            MaximumP95LatencyRatio = _policy.MaximumP95LatencyRatio, Pairs = pairs
        };
        evidence.Validate();
        string evidenceId = _registry.RetainEvidence(JsonSerializer.SerializeToUtf8Bytes(evidence));
        return new Prepared(evidence.Qualifies(_policy) ? "Approved" : "InsufficientImprovement", candidate, incumbent, slot.Revision, epoch, evidenceId);

        Prepared Rejected(string reason) => new(reason, candidate, incumbent, slot.Revision, epoch,
            _registry.RetainEvidence(JsonSerializer.SerializeToUtf8Bytes(new
            { SchemaVersion = 1, Kind = "rejected-deployment-validation", CandidateId = candidate.Id, IncumbentId = incumbent.Id,
                EnvelopeKey = candidate.Envelope.Key, Reason = reason, Observations = observations })));
    }

    /// <summary>Attributes a fixed raw window to an exact dispatch revision and quarantines after consecutive regressions.</summary>
    public async Task<EvolutionDeploymentDecision> ObserveAsync(EvolutionDeploymentSelection observed,
        IReadOnlyList<EvolutionDeploymentMeasurement> measurements, DateTimeOffset observedAt, CancellationToken cancellationToken = default)
    {
        if (observed is null) throw new ArgumentNullException(nameof(observed));
        if (measurements is null || measurements.Count != _policy.MonitoringSamples || observedAt == default)
            throw new ArgumentException("A bounded, timestamped monitoring window is required.");
        var copy = measurements.ToArray();
        if (copy.Any(value => value is null || !value.IsFresh || value.Direction != _policy.Direction))
            throw new ArgumentException("Monitoring requires fresh measurements in the configured objective direction.");
        if (!await _operation.WaitAsync(0, cancellationToken).ConfigureAwait(false)) return new("Busy", false);
        try
        {
            var slot = _registry.ReadSlot();
            long epoch;
            lock (_gate)
            {
                if (observed.IsFallback || slot.Revision != observed.Revision || slot.ActiveId != observed.Artifact.Id ||
                    _observed.Key != observed.Artifact.Envelope.Key || _storageFaulted) return new("Stale", false);
                epoch = _epoch;
            }
            var validation = DeploymentEncoding.Parse<DeploymentValidationEvidence>(_registry.ReadEvidence(slot.ValidationEvidenceId!));
            validation.Validate();
            if (!validation.AllPassed || validation.EnvelopeKey != observed.Artifact.Envelope.Key || validation.Direction != _policy.Direction ||
                (slot.CandidateEvidenceSide ? validation.CandidateId : validation.IncumbentId) != observed.Artifact.Id)
                throw new InvalidDataException("Active validation evidence does not describe the observed deployment.");
            double quality = DeploymentValidationEvidence.Mean(copy.Select(value => value.Quality).ToArray());
            long p95 = DeploymentValidationEvidence.P95(copy.Select(value => value.Elapsed.Ticks).ToArray());
            double loss = _policy.Direction == EvolutionOptimizationDirection.Maximize
                ? validation.Mean(slot.CandidateEvidenceSide) - quality : quality - validation.Mean(slot.CandidateEvidenceSide);
            bool regression = copy.Any(value => !value.CorrectnessPassed) || loss > _policy.MaximumQualityDrop ||
                p95 / (double)validation.P95(slot.CandidateEvidenceSide) > _policy.MaximumMonitoredLatencyRatio;
            lock (_gate)
            {
                if (_monitoredRevision != slot.Revision) { _monitoredRevision = slot.Revision; _lastObservation = null; _windows.Clear(); }
                if (_lastObservation is { } last && observedAt <= last) throw new ArgumentException("Monitoring windows must advance in time.");
                _lastObservation = observedAt;
                if (!regression) { _windows.Clear(); return new("Healthy", false); }
                if (_windows.Count == _policy.ConsecutiveRegressions) _windows.RemoveAt(0);
                _windows.Add(new MonitoringWindow { ObservedAtUtc = observedAt.ToUniversalTime(), Measurements = copy.Select(DeploymentRawMeasurement.From).ToArray() });
                if (_windows.Count < _policy.ConsecutiveRegressions) return new("Monitoring", false);
            }
            string evidence = _registry.RetainEvidence(JsonSerializer.SerializeToUtf8Bytes(new
            {
                SchemaVersion = 1, Kind = "deployment-regression", ArtifactId = observed.Artifact.Id, observed.Revision,
                EnvelopeKey = observed.Artifact.Envelope.Key, slot.ValidationEvidenceId, slot.CandidateEvidenceSide,
                _policy.MaximumQualityDrop, _policy.MaximumMonitoredLatencyRatio, _policy.ConsecutiveRegressions,
                Windows = _windows.ToArray()
            }));
            cancellationToken.ThrowIfCancellationRequested();
            lock (_gate)
            {
                if (_epoch != epoch) return new("Stale", false, observed.Artifact.Id, evidence);
                bool applied = _registry.TryQuarantine(slot.Revision, observed.Artifact.Id, evidence, observed.Artifact.Envelope, out var prior);
                if (!applied) return new("Stale", false, observed.Artifact.Id, evidence);
                _cached = prior; _windows.Clear(); _pending = prior is null ? _observed : null;
                return new(prior is null ? "QuarantinedFallback" : "RolledBack", prior is not null, prior?.Id, evidence);
            }
        }
        catch (Exception error) when (StorageFailure(error)) { lock (_gate) { _storageFaulted = true; _pending = null; } throw; }
        finally { _operation.Release(); }
    }

    private bool Usable(EvolutionDeploymentMeasurement value) => value.CorrectnessPassed && value.IsFresh && value.HasQuality && value.Direction == _policy.Direction;
    private EvolutionDeploymentSelection Select(EvolutionDeploymentArtifactRegistry.Slot slot, EvolutionDeploymentEnvelope envelope)
    {
        if (slot.ActiveId is null || slot.EnvelopeKey != envelope.Key || _registry.IsQuarantined(slot.ActiveId)) return Fallback(envelope, slot.Revision);
        EvolutionDeployableArtifact? cached;
        lock (_gate) cached = _cached;
        var artifact = cached is not null && cached.Id == slot.ActiveId ? cached : _registry.Load(slot.ActiveId, envelope);
        lock (_gate) _cached = artifact;
        return new(artifact, slot.Revision, false);
    }
    private EvolutionDeploymentSelection Fallback(EvolutionDeploymentEnvelope envelope, string? revision)
    {
        var artifact = _fallback(envelope) ?? throw new InvalidOperationException("No known-valid deployment fallback was supplied.");
        if (artifact.Envelope.Key != envelope.Key) throw new InvalidOperationException("Fallback applicability differs from the observed environment.");
        return new(artifact, revision, true);
    }
    private static bool StorageFailure(Exception error) => error is IOException or InvalidDataException or UnauthorizedAccessException or JsonException;
    private sealed class Prepared
    {
        internal Prepared(string outcome, EvolutionDeployableArtifact? candidate = null, EvolutionDeployableArtifact? incumbent = null,
            string? revision = null, long epoch = 0, string? evidenceId = null)
        { Outcome = outcome; Candidate = candidate; Incumbent = incumbent; Revision = revision; Epoch = epoch; EvidenceId = evidenceId; }
        internal string Outcome { get; }
        internal EvolutionDeployableArtifact? Candidate { get; }
        internal EvolutionDeployableArtifact? Incumbent { get; }
        internal string? Revision { get; }
        internal long Epoch { get; }
        internal string? EvidenceId { get; }
    }
    private sealed class MonitoringWindow
    {
        public DateTimeOffset ObservedAtUtc { get; set; }
        public DeploymentRawMeasurement[] Measurements { get; set; } = Array.Empty<DeploymentRawMeasurement>();
    }
}
