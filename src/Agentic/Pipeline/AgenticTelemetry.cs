using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// The instrumentation source for the agentic subsystem: a named <see cref="ActivitySource"/> (traces) and
/// <see cref="Meter"/> (metrics) following OpenTelemetry GenAI semantic conventions. Any OpenTelemetry
/// exporter subscribed to the source name <see cref="SourceName"/> collects chat spans and token metrics with
/// no extra wiring.
/// </summary>
/// <remarks>
/// <para>
/// A named <see cref="ActivitySource"/>/<see cref="Meter"/> is the idiomatic .NET instrumentation pattern:
/// emission is essentially free when nothing is listening, and standard OpenTelemetry collectors enable it by
/// name. <see cref="TelemetryChatMiddleware"/> is the producer.
/// </para>
/// <para><b>For Beginners:</b> This is the labeled channel the system broadcasts "what the model did" on —
/// timings, token counts, finish reasons. Point your monitoring/dashboard at the name and you get visibility
/// without changing agent code.
/// </para>
/// </remarks>
public static class AgenticTelemetry
{
    /// <summary>The OpenTelemetry source/meter name for the agentic subsystem.</summary>
    public const string SourceName = "AiDotNet.Agentic";

    /// <summary>The activity source that emits chat spans.</summary>
    public static readonly ActivitySource Source = new(SourceName);

    /// <summary>The meter that emits chat metrics.</summary>
    public static readonly Meter Meter = new(SourceName);

    /// <summary>Counts chat operations (GenAI convention: gen_ai.client.operation.count).</summary>
    public static readonly Counter<long> OperationCount =
        Meter.CreateCounter<long>("gen_ai.client.operation.count");

    /// <summary>Records token usage per call (GenAI convention: gen_ai.client.token.usage).</summary>
    public static readonly Histogram<long> TokenUsage =
        Meter.CreateHistogram<long>("gen_ai.client.token.usage");
}
