namespace AiDotNet.Evolution.Programs;

/// <summary>Connects a nonblocking, thread-safe live telemetry source before program evolution starts.</summary>
/// <remarks>Sources must return immediately without I/O or waiting for another engine event. The caller owns the observer.</remarks>
public interface IProgramEvolutionTelemetryObserver
{
    /// <summary>Registers the live source; consumers must stop querying it when their run finishes.</summary>
    void SetTelemetrySource(Func<ProgramEvolutionTelemetrySnapshot> source);
}

/// <summary>Detached current-segment usage and configured runner queue counters, not a billing receipt.</summary>
public sealed class ProgramEvolutionTelemetrySnapshot
{
    /// <summary>Creates a bounded snapshot; null queue values mean the adapter supplies no telemetry.</summary>
    public ProgramEvolutionTelemetrySnapshot(string operatorId, string? configuredModelId,
        ProgramEvolutionLlmUsage usage, int? queuedExecutions = null, int? activeExecutions = null)
    {
        if (string.IsNullOrWhiteSpace(operatorId) || operatorId.Length > 256 || operatorId.Any(char.IsControl))
            throw new ArgumentException("Operator identity must be bounded and printable.", nameof(operatorId));
        if (configuredModelId is not null && (configuredModelId.Length > 256 || configuredModelId.Any(char.IsControl)))
            throw new ArgumentException("Model identity must be bounded and printable.", nameof(configuredModelId));
        if (queuedExecutions < 0 || activeExecutions < 0) throw new ArgumentOutOfRangeException(nameof(queuedExecutions));
        OperatorId = operatorId;
        ConfiguredModelId = configuredModelId;
        Usage = usage ?? throw new ArgumentNullException(nameof(usage));
        QueuedExecutions = queuedExecutions;
        ActiveExecutions = activeExecutions;
    }

    /// <summary>Gets the configured operator identity, not arbitrary event text.</summary>
    public string OperatorId { get; }
    /// <summary>Gets the configured client's model identity; response-level routing may differ.</summary>
    public string? ConfiguredModelId { get; }
    /// <summary>Gets provider-reported counters; absent token reporting must not be interpreted as free usage.</summary>
    public ProgramEvolutionLlmUsage Usage { get; }
    /// <summary>Gets waiting requests on the configured runner instance, not an OS or cluster queue.</summary>
    public int? QueuedExecutions { get; }
    /// <summary>Gets active requests on the configured runner instance.</summary>
    public int? ActiveExecutions { get; }
}
