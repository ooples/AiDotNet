using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System.Diagnostics;
using System.Diagnostics.Metrics;

namespace AiDotNet.Serving.Observability;

/// <summary>
/// Opt-in OpenTelemetry registration scaffold for the AiDotNet serving host.
/// </summary>
/// <remarks>
/// <para>
/// The serving project deliberately does NOT take a hard dependency on
/// <c>OpenTelemetry.Extensions.Hosting</c> or its exporters — those
/// packages add a non-trivial transitive surface that consumers who
/// don't want OpenTelemetry shouldn't pay for. Instead, this class
/// exposes the <see cref="ActivitySource"/> + <see cref="Meter"/>
/// instances the serving stack emits to, and an opt-in registration
/// hook the host can call after adding its own OpenTelemetry
/// configuration.
/// </para>
/// <para>
/// Typical consumer wiring in their own <c>Program.cs</c> (after adding
/// the OpenTelemetry NuGet packages they want):
/// <code>
/// builder.Services.AddOpenTelemetry()
///     .ConfigureResource(r =&gt; r.AddService("AiDotNet.Serving"))
///     .WithTracing(t =&gt; t.AddSource(AiDotNetServingTelemetry.ActivitySourceName))
///     .WithMetrics(m =&gt; m.AddMeter(AiDotNetServingTelemetry.MeterName));
/// </code>
/// </para>
/// <para>
/// The serving controllers and pipeline middleware should emit to
/// <see cref="AiDotNetServingTelemetry.ActivitySource"/> for spans and
/// <see cref="AiDotNetServingTelemetry.Meter"/> for counters / histograms.
/// As of the audit-hardening PR these instruments are wired up but not
/// yet broadly instrumented across every controller — full coverage is
/// tracked as a separate work item under the audit follow-up family.
/// </para>
/// </remarks>
public static class AiDotNetServingTelemetry
{
    /// <summary>
    /// Name of the ActivitySource for tracing serving operations.
    /// Consumers wire this into their OpenTelemetry tracing pipeline.
    /// </summary>
    public const string ActivitySourceName = "AiDotNet.Serving";

    /// <summary>
    /// Name of the Meter for serving metrics (counters, histograms).
    /// Consumers wire this into their OpenTelemetry metrics pipeline.
    /// </summary>
    public const string MeterName = "AiDotNet.Serving";

    /// <summary>
    /// Shared ActivitySource for the serving host. Controllers and
    /// middleware should call <c>ActivitySource.StartActivity(...)</c>
    /// at request boundaries.
    /// </summary>
    public static readonly ActivitySource ActivitySource = new(ActivitySourceName);

    /// <summary>
    /// Shared Meter for the serving host. Use this to create counters,
    /// histograms, gauges, etc. for metric emission.
    /// </summary>
    public static readonly Meter Meter = new(MeterName);

    /// <summary>
    /// Disposes the static <see cref="ActivitySource"/> and
    /// <see cref="Meter"/> singletons. Reserved for test-teardown
    /// scenarios where the serving host is repeatedly created and torn
    /// down inside the same process; production hosts rely on
    /// app-domain teardown for cleanup and should NOT call this.
    /// </summary>
    /// <remarks>
    /// Both <see cref="ActivitySource"/> and <see cref="Meter"/>
    /// implement <see cref="System.IDisposable"/> but are exposed as
    /// static readonly singletons because the OpenTelemetry SDK
    /// expects long-lived instruments. Repeated factory-create /
    /// dispose cycles in integration tests can otherwise leak
    /// instrument-side listeners; tests that exercise that lifecycle
    /// should call this at teardown. After disposal the singletons
    /// remain referenced — re-entering the host without restarting
    /// the process will use the disposed instances, so this is
    /// strictly a "test teardown right before app-domain exit" hook,
    /// not a "reset between tests" hook.
    /// </remarks>
    public static void DisposeTelemetry()
    {
        ActivitySource.Dispose();
        Meter.Dispose();
    }

    /// <summary>
    /// Reserved opt-in registration point for AiDotNet.Serving's
    /// OpenTelemetry-facing configuration. Currently a no-op — the
    /// instruments consumers actually wire into their OpenTelemetry
    /// pipeline are the static <see cref="ActivitySource"/> and
    /// <see cref="Meter"/> fields above (referenced by name via
    /// <see cref="ActivitySourceName"/> / <see cref="MeterName"/>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method exists so consumers can call
    /// <c>services.AddAiDotNetServingObservability(configuration)</c>
    /// from their host setup TODAY and have that call automatically
    /// pick up future wiring (TelemetryOptions binding, dependency-
    /// injected redaction policies, sampling-override services) without
    /// the consumer needing to add a second registration line then.
    /// </para>
    /// <para>
    /// Until that wiring lands the method does NOT bind configuration,
    /// register hosted services, add named-options, or register
    /// instruments — it returns the supplied service collection
    /// unchanged. The XML doc here matches that contract exactly so a
    /// consumer reading IntelliSense isn't misled about what calling
    /// the method does today.
    /// </para>
    /// </remarks>
    /// <param name="services">The serving host's service collection.</param>
    /// <param name="configuration">The serving host's configuration.</param>
    /// <returns>The same service collection for chaining.</returns>
    public static IServiceCollection AddAiDotNetServingObservability(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        // No-op today; the method exists so consumers can register
        // against a stable opt-in point now and pick up future wiring
        // automatically. See the XML doc above for the contract.
        return services;
    }
}
