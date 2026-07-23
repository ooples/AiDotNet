using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Threading;

namespace AiDotNet.Serving.Observability;

/// <summary>
/// Process-wide serving metrics rendered in the Prometheus text exposition format (the same format vLLM and
/// TGI expose at <c>/metrics</c>), so operators can scrape AiDotNet serving with a standard Prometheus stack.
/// </summary>
/// <remarks>
/// <para>
/// This is rendered natively (no third-party exporter dependency) so the library carries no beta/preview
/// package. Request-level counters and latency histograms are recorded from the OpenAI-compatible request
/// path; live batcher gauges (throughput, queue depth, batch utilization) are folded in at scrape time from
/// the running <c>IRequestBatcher</c>'s performance snapshot.
/// </para>
/// <para><b>For Beginners:</b> Prometheus is the industry-standard tool for collecting server metrics. This
/// class produces the plain-text page it reads (requests served, tokens generated, how long requests take),
/// so you can point Grafana at your AiDotNet server and get the same dashboards people build for vLLM.
/// </para>
/// </remarks>
public static class ServingMetrics
{
    // Cumulative counters (monotonic). Interlocked for lock-free recording on the hot path.
    private static long _requestsSuccess;
    private static long _requestsError;
    private static long _promptTokens;
    private static long _generationTokens;

    private static readonly PromHistogram RequestDuration = PromHistogram.LatencySeconds();
    private static readonly PromHistogram TimeToFirstToken = PromHistogram.LatencySeconds();
    private static readonly PromHistogram TimePerOutputToken = PromHistogram.SmallLatencySeconds();

    /// <summary>
    /// Records a completed generation request. <paramref name="ttftSeconds"/> and <paramref name="tpotSeconds"/>
    /// are optional (null when the path cannot measure them, e.g. non-streaming has no first-token timing).
    /// </summary>
    public static void RecordRequest(
        bool success, int promptTokens, int generationTokens,
        double durationSeconds, double? ttftSeconds = null, double? tpotSeconds = null)
    {
        if (success) Interlocked.Increment(ref _requestsSuccess);
        else Interlocked.Increment(ref _requestsError);

        if (promptTokens > 0) Interlocked.Add(ref _promptTokens, promptTokens);
        if (generationTokens > 0) Interlocked.Add(ref _generationTokens, generationTokens);

        if (durationSeconds >= 0) RequestDuration.Observe(durationSeconds);
        if (ttftSeconds is { } ttft && ttft >= 0) TimeToFirstToken.Observe(ttft);
        if (tpotSeconds is { } tpot && tpot >= 0) TimePerOutputToken.Observe(tpot);
    }

    /// <summary>Resets all counters and histograms. Internal: exists only for test isolation — resetting
    /// process-wide metrics at runtime is a footgun, so it is not part of the supported public surface.</summary>
    internal static void Reset()
    {
        Interlocked.Exchange(ref _requestsSuccess, 0);
        Interlocked.Exchange(ref _requestsError, 0);
        Interlocked.Exchange(ref _promptTokens, 0);
        Interlocked.Exchange(ref _generationTokens, 0);
        RequestDuration.Reset();
        TimeToFirstToken.Reset();
        TimePerOutputToken.Reset();
    }

    /// <summary>
    /// Renders the current metrics as a Prometheus text-exposition (version 0.0.4) page. When
    /// <paramref name="batcherMetrics"/> (the running batcher's <c>GetPerformanceMetrics()</c> snapshot) is
    /// supplied, its live gauges (throughput, batch size, queue depth, utilization, latency percentiles) are
    /// included as <c>gauge</c> series.
    /// </summary>
    public static string RenderPrometheus(IReadOnlyDictionary<string, object>? batcherMetrics = null)
    {
        var sb = new StringBuilder(4096);

        Counter(sb, "aidotnet_serving_requests_total",
            "Total completed generation requests by outcome.",
            ("outcome=\"success\"", Volatile.Read(ref _requestsSuccess)),
            ("outcome=\"error\"", Volatile.Read(ref _requestsError)));

        Counter(sb, "aidotnet_serving_prompt_tokens_total",
            "Total prompt (input) tokens processed.", (null, Volatile.Read(ref _promptTokens)));
        Counter(sb, "aidotnet_serving_generation_tokens_total",
            "Total generated (output) tokens produced.", (null, Volatile.Read(ref _generationTokens)));

        RequestDuration.Render(sb, "aidotnet_serving_request_duration_seconds",
            "End-to-end request latency in seconds.");
        TimeToFirstToken.Render(sb, "aidotnet_serving_time_to_first_token_seconds",
            "Time to first streamed token in seconds.");
        TimePerOutputToken.Render(sb, "aidotnet_serving_time_per_output_token_seconds",
            "Per-output-token latency (inter-token latency) in seconds.");

        if (batcherMetrics is not null)
        {
            RenderBatcherGauges(sb, batcherMetrics);
        }

        return sb.ToString();
    }

    private static void RenderBatcherGauges(StringBuilder sb, IReadOnlyDictionary<string, object> m)
    {
        // Map the batcher's live performance snapshot to Prometheus gauges. Missing keys are skipped so the
        // page stays valid across batcher implementations.
        Gauge(sb, "aidotnet_serving_throughput_requests_per_second",
            "Recent request throughput (requests/second).", m, "throughputRequestsPerSecond");
        Gauge(sb, "aidotnet_serving_batch_size_avg",
            "Average scheduled batch size.", m, "averageBatchSize");
        Gauge(sb, "aidotnet_serving_queue_depth_avg",
            "Average scheduler queue depth (waiting requests).", m, "averageQueueDepth");
        Gauge(sb, "aidotnet_serving_batch_utilization_percent",
            "Batch utilization (non-padding fraction) as a percentage.", m, "batchUtilizationPercent");
        Gauge(sb, "aidotnet_serving_latency_p50_ms",
            "Batcher-observed p50 request latency in milliseconds.", m, "latencyP50Ms");
        Gauge(sb, "aidotnet_serving_latency_p95_ms",
            "Batcher-observed p95 request latency in milliseconds.", m, "latencyP95Ms");
        Gauge(sb, "aidotnet_serving_latency_p99_ms",
            "Batcher-observed p99 request latency in milliseconds.", m, "latencyP99Ms");
        Gauge(sb, "aidotnet_serving_uptime_seconds",
            "Serving process uptime in seconds.", m, "uptimeSeconds");
    }

    private static void Counter(StringBuilder sb, string name, string help, params (string? Labels, long Value)[] series)
    {
        sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
        sb.Append("# TYPE ").Append(name).Append(" counter\n");
        foreach (var (labels, value) in series)
        {
            sb.Append(name);
            if (!string.IsNullOrEmpty(labels)) sb.Append('{').Append(labels).Append('}');
            sb.Append(' ').Append(value.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }
    }

    private static void Gauge(StringBuilder sb, string name, string help, IReadOnlyDictionary<string, object> m, string key)
    {
        if (!m.TryGetValue(key, out var raw)) return;
        if (!TryToDouble(raw, out double value)) return;
        sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
        sb.Append("# TYPE ").Append(name).Append(" gauge\n");
        sb.Append(name).Append(' ').Append(FormatDouble(value)).Append('\n');
    }

    private static bool TryToDouble(object raw, out double value)
    {
        switch (raw)
        {
            case double d: value = d; return true;
            case float f: value = f; return true;
            case long l: value = l; return true;
            case int i: value = i; return true;
            default:
                return double.TryParse(
                    raw?.ToString(), NumberStyles.Any, CultureInfo.InvariantCulture, out value);
        }
    }

    internal static string FormatDouble(double value)
    {
        if (double.IsNaN(value)) return "NaN";
        if (double.IsPositiveInfinity(value)) return "+Inf";
        if (double.IsNegativeInfinity(value)) return "-Inf";
        return value.ToString("0.######", CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// A minimal thread-safe fixed-bucket histogram rendered in the Prometheus cumulative-bucket layout.
    /// Bucket counts and the running sum are updated lock-free with <see cref="Interlocked"/>.
    /// </summary>
    private sealed class PromHistogram
    {
        private readonly double[] _bounds; // ascending upper bounds (le), excluding +Inf
        private readonly long[] _counts;   // per-bucket counts; _counts[^1] is the +Inf overflow bucket
        private long _sumMicros;           // sum of observed seconds, stored as microseconds for Interlocked

        private PromHistogram(double[] bounds)
        {
            _bounds = bounds;
            _counts = new long[bounds.Length + 1]; // + overflow (+Inf) bucket
        }

        public static PromHistogram LatencySeconds() => new(new[]
        {
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60
        });

        public static PromHistogram SmallLatencySeconds() => new(new[]
        {
            0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1
        });

        public void Observe(double seconds)
        {
            Interlocked.Add(ref _sumMicros, (long)(seconds * 1_000_000.0));
            int idx = _bounds.Length; // default: overflow bucket
            for (int i = 0; i < _bounds.Length; i++)
            {
                if (seconds <= _bounds[i]) { idx = i; break; }
            }
            Interlocked.Increment(ref _counts[idx]);
        }

        public void Reset()
        {
            for (int i = 0; i < _counts.Length; i++) Interlocked.Exchange(ref _counts[i], 0);
            Interlocked.Exchange(ref _sumMicros, 0);
        }

        public void Render(StringBuilder sb, string name, string help)
        {
            sb.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            sb.Append("# TYPE ").Append(name).Append(" histogram\n");

            long cumulative = 0;
            for (int i = 0; i < _bounds.Length; i++)
            {
                cumulative += Volatile.Read(ref _counts[i]);
                sb.Append(name).Append("_bucket{le=\"")
                  .Append(_bounds[i].ToString("0.######", CultureInfo.InvariantCulture))
                  .Append("\"} ").Append(cumulative.ToString(CultureInfo.InvariantCulture)).Append('\n');
            }
            cumulative += Volatile.Read(ref _counts[^1]); // +Inf overflow
            long total = cumulative;
            sb.Append(name).Append("_bucket{le=\"+Inf\"} ").Append(total.ToString(CultureInfo.InvariantCulture)).Append('\n');

            double sumSeconds = Volatile.Read(ref _sumMicros) / 1_000_000.0;
            sb.Append(name).Append("_sum ").Append(FormatDouble(sumSeconds)).Append('\n');
            sb.Append(name).Append("_count ").Append(total.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }
    }
}
