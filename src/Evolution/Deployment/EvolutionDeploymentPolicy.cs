namespace AiDotNet.Evolution.Deployment;

/// <summary>One fresh independent validation observation, with declared direction and measured evaluation cost.</summary>
public sealed class EvolutionDeploymentMeasurement
{
    /// <summary>Creates a raw observation; providers remain responsible for truthful correctness, timing and cost.</summary>
    public EvolutionDeploymentMeasurement(bool correctnessPassed, double quality, EvolutionOptimizationDirection direction,
        TimeSpan elapsed, double costUnits, bool isFresh = true, bool hasQuality = true)
    {
        if (!Finite(quality) || !Finite(costUnits) || costUnits < 0 || elapsed <= TimeSpan.Zero)
            throw new ArgumentException("Deployment observations require finite quality/cost and positive elapsed time.");
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        if (correctnessPassed && !hasQuality) throw new ArgumentException("A passed observation requires a measured objective.");
        CorrectnessPassed = correctnessPassed; Quality = quality; Direction = direction;
        Elapsed = elapsed; CostUnits = costUnits; IsFresh = isFresh; HasQuality = hasQuality;
    }
    /// <summary>Gets whether independent correctness checks passed.</summary>
    public bool CorrectnessPassed { get; }
    /// <summary>Gets the measured objective value.</summary>
    public double Quality { get; }
    /// <summary>Gets whether quality was reported; a failed evaluation can explicitly lack a score.</summary>
    public bool HasQuality { get; }
    /// <summary>Gets the objective direction.</summary>
    public EvolutionOptimizationDirection Direction { get; }
    /// <summary>Gets elapsed time in the declared protocol's measurement scope.</summary>
    public TimeSpan Elapsed { get; }
    /// <summary>Gets measured cost in application-defined units, not inferred money.</summary>
    public double CostUnits { get; }
    /// <summary>Gets whether this is a new measurement rather than reused search evidence.</summary>
    public bool IsFresh { get; }
    internal static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}

/// <summary>Explicit application authorization and fixed per-decision validation/lifetime-retuning bounds.</summary>
/// <remarks>
/// A one-sided exact paired sign test checks directional improvement under independent paired samples.
/// Mean gain and P95 gates are separate practical constraints. This is not a family-wise superiority claim across repeated decisions.
/// The registry currently offers best-effort directory persistence, which must be explicitly authorized.
/// </remarks>
public sealed class EvolutionDeploymentPolicy
{
    /// <summary>Creates a frozen policy; no setting is inferred from a benchmark winner.</summary>
    public EvolutionDeploymentPolicy(bool allowBestEffortPersistence, EvolutionOptimizationDirection direction,
        int pairedSamples = 7, double minimumMeanGain = 0, double maximumPValue = 0.05,
        double maximumP95LatencyRatio = 1.1, int maximumRetunes = 3, int searchEvaluations = 50,
        int searchProposals = 100, TimeSpan? timeout = null, TimeSpan? gracePeriod = null,
        TimeSpan? cooldown = null, int monitoringSamples = 7, int consecutiveRegressions = 2,
        double maximumQualityDrop = 0.01, double maximumMonitoredLatencyRatio = 1.2)
    {
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), direction)) throw new ArgumentOutOfRangeException(nameof(direction));
        if (pairedSamples is < 5 or > 64 || monitoringSamples is < 1 or > 64 || consecutiveRegressions is < 1 or > 64)
            throw new ArgumentOutOfRangeException(nameof(pairedSamples), "Validation and monitoring windows must be bounded.");
        if (!EvolutionDeploymentMeasurement.Finite(minimumMeanGain) || minimumMeanGain < 0 ||
            !EvolutionDeploymentMeasurement.Finite(maximumPValue) || maximumPValue <= 0 || maximumPValue >= 0.5 ||
            !EvolutionDeploymentMeasurement.Finite(maximumP95LatencyRatio) || maximumP95LatencyRatio <= 0 ||
            !EvolutionDeploymentMeasurement.Finite(maximumQualityDrop) || maximumQualityDrop < 0 ||
            !EvolutionDeploymentMeasurement.Finite(maximumMonitoredLatencyRatio) || maximumMonitoredLatencyRatio <= 1)
            throw new ArgumentException("Invalid finite deployment decision thresholds.");
        if (maximumRetunes is < 1 or > 1024 || searchEvaluations is < 1 or > 1000000 ||
            searchProposals < searchEvaluations || searchProposals > 1000000)
            throw new ArgumentException("Retuning requires finite evaluation, proposal and lifetime admission limits.");
        Timeout = timeout ?? TimeSpan.FromMinutes(2);
        GracePeriod = gracePeriod ?? TimeSpan.FromSeconds(5);
        Cooldown = cooldown ?? TimeSpan.FromMinutes(10);
        if (Timeout <= TimeSpan.Zero || Timeout > TimeSpan.FromDays(1) || GracePeriod < TimeSpan.Zero ||
            GracePeriod > TimeSpan.FromMinutes(5) || Cooldown < TimeSpan.Zero || Cooldown > TimeSpan.FromDays(30))
            throw new ArgumentException("Invalid deployment time bounds.");
        AllowBestEffortPersistence = allowBestEffortPersistence; Direction = direction; PairedSamples = pairedSamples;
        MinimumMeanGain = minimumMeanGain; MaximumPValue = maximumPValue; MaximumP95LatencyRatio = maximumP95LatencyRatio;
        MaximumRetunes = maximumRetunes; SearchEvaluations = searchEvaluations; SearchProposals = searchProposals;
        MonitoringSamples = monitoringSamples; ConsecutiveRegressions = consecutiveRegressions;
        MaximumQualityDrop = maximumQualityDrop; MaximumMonitoredLatencyRatio = maximumMonitoredLatencyRatio;
    }
    /// <summary>Gets explicit acceptance of flushed-file/atomic-rename storage without a power-loss directory guarantee.</summary>
    public bool AllowBestEffortPersistence { get; }
    /// <summary>Gets the objective direction.</summary>
    public EvolutionOptimizationDirection Direction { get; }
    /// <summary>Gets the fixed number of independent candidate/incumbent validation pairs.</summary>
    public int PairedSamples { get; }
    /// <summary>Gets the minimum mean gain in objective units, after applying direction.</summary>
    public double MinimumMeanGain { get; }
    /// <summary>Gets the maximum one-sided paired sign-test p-value.</summary>
    public double MaximumPValue { get; }
    /// <summary>Gets the maximum candidate/incumbent P95 latency ratio.</summary>
    public double MaximumP95LatencyRatio { get; }
    /// <summary>Gets the controller-lifetime retuning admission cap; failures/cancellation are not refunded.</summary>
    public int MaximumRetunes { get; }
    /// <summary>Gets the maximum search evaluation/training attempts per retune, separate from final comparison.</summary>
    public int SearchEvaluations { get; }
    /// <summary>Gets the maximum search proposals per retune.</summary>
    public int SearchProposals { get; }
    /// <summary>Gets the cooperative per-operation deadline.</summary>
    public TimeSpan Timeout { get; }
    /// <summary>Gets the additional wait before reporting abandoned uncooperative work.</summary>
    public TimeSpan GracePeriod { get; }
    /// <summary>Gets the minimum interval between retuning admissions.</summary>
    public TimeSpan Cooldown { get; }
    /// <summary>Gets the fixed monitoring-window size.</summary>
    public int MonitoringSamples { get; }
    /// <summary>Gets consecutive violating windows required before quarantine.</summary>
    public int ConsecutiveRegressions { get; }
    /// <summary>Gets the allowed mean quality loss relative to the exact active validation evidence.</summary>
    public double MaximumQualityDrop { get; }
    /// <summary>Gets the allowed P95 ratio relative to the exact active validation evidence.</summary>
    public double MaximumMonitoredLatencyRatio { get; }
}

/// <summary>A bounded, immutable retuning admission passed to a program or AutoML search driver.</summary>
public sealed class EvolutionDeploymentRetuneRequest
{
    internal EvolutionDeploymentRetuneRequest(EvolutionDeploymentEnvelope envelope, EvolutionDeploymentPolicy policy)
    { Envelope = envelope; MaximumEvaluations = policy.SearchEvaluations; MaximumProposals = policy.SearchProposals; Timeout = policy.Timeout; }
    /// <summary>Gets the environment for which the private search must produce a candidate.</summary>
    public EvolutionDeploymentEnvelope Envelope { get; }
    /// <summary>Gets the admitted search evaluation/training limit.</summary>
    public int MaximumEvaluations { get; }
    /// <summary>Gets the admitted proposal limit.</summary>
    public int MaximumProposals { get; }
    /// <summary>Gets the cooperative search timeout, including idle admission.</summary>
    public TimeSpan Timeout { get; }
}

/// <summary>An explicit deployment decision with retained raw evidence, never an implied authorization to merge or publish packages.</summary>
public sealed class EvolutionDeploymentDecision
{
    internal EvolutionDeploymentDecision(string outcome, bool activated, string? artifactId = null, string? evidenceId = null)
    { Outcome = outcome; Activated = activated; ArtifactId = artifactId; EvidenceId = evidenceId; }
    /// <summary>Gets the terminal decision or scheduling reason.</summary>
    public string Outcome { get; }
    /// <summary>Gets whether this operation changed the persisted active implementation.</summary>
    public bool Activated { get; }
    /// <summary>Gets the selected candidate or rollback artifact, when applicable.</summary>
    public string? ArtifactId { get; }
    /// <summary>Gets the immutable raw evidence digest, when an evaluation or regression decision was retained.</summary>
    public string? EvidenceId { get; }
}
