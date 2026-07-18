using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace AiDotNet.Serving.Benchmarks;

/// <summary>Per-request measurement captured by a backend adapter.</summary>
public sealed class RequestResult
{
    /// <summary>Index of the originating request spec.</summary>
    public int Index { get; set; }

    /// <summary>Whether the request completed without error.</summary>
    public bool Success { get; set; }

    /// <summary>Error message when <see cref="Success"/> is false.</summary>
    public string? Error { get; set; }

    /// <summary>Dispatch time relative to run start, in ms.</summary>
    public double DispatchMs { get; set; }

    /// <summary>Completion time relative to run start, in ms.</summary>
    public double EndMs { get; set; }

    /// <summary>Time-to-first-token relative to dispatch, in ms. Null for non-streaming backends.</summary>
    public double? TtftMs { get; set; }

    /// <summary>Number of prompt tokens (from server usage, or the synthesized prompt length).</summary>
    public int PromptTokens { get; set; }

    /// <summary>Number of generated tokens.</summary>
    public int OutputTokens { get; set; }

    /// <summary>Token arrival times relative to dispatch, in ms (streaming backends only).</summary>
    public List<double> TokenArrivalsMs { get; } = new();

    /// <summary>End-to-end latency in ms.</summary>
    public double E2EMs => EndMs - DispatchMs;

    /// <summary>Mean time-per-output-token (excludes the first token / TTFT), in ms.</summary>
    public double? TpotMs =>
        (TtftMs is double t && OutputTokens > 1) ? (E2EMs - t) / (OutputTokens - 1) : null;

    /// <summary>Inter-token latencies (gaps between consecutive streamed tokens), in ms.</summary>
    public IEnumerable<double> InterTokenLatencies()
    {
        for (int i = 1; i < TokenArrivalsMs.Count; i++)
            yield return TokenArrivalsMs[i] - TokenArrivalsMs[i - 1];
    }
}

/// <summary>Aggregated benchmark result with the standard serving metrics.</summary>
public sealed class BenchmarkReport
{
    public string Backend { get; set; } = "";
    public string Model { get; set; } = "";
    public double DurationSec { get; set; }
    public int Completed { get; set; }
    public int Failed { get; set; }
    public bool Streaming { get; set; }

    public long TotalPromptTokens { get; set; }
    public long TotalOutputTokens { get; set; }

    public double RequestThroughput { get; set; }      // req/s
    public double OutputThroughput { get; set; }        // output tok/s
    public double TotalTokenThroughput { get; set; }    // (prompt+output) tok/s

    public Stat Ttft { get; set; } = Stat.Empty;        // ms
    public Stat Tpot { get; set; } = Stat.Empty;        // ms (per-request mean)
    public Stat Itl { get; set; } = Stat.Empty;         // ms (all inter-token gaps)
    public Stat E2E { get; set; } = Stat.Empty;         // ms

    public double GoodputPerSec { get; set; }           // req/s meeting both SLAs
    public double SlaTtftMs { get; set; }
    public double SlaTpotMs { get; set; }

    /// <summary>Summary statistic bundle (mean + percentiles), all in ms.</summary>
    public sealed class Stat
    {
        public double Mean { get; set; }
        public double P50 { get; set; }
        public double P90 { get; set; }
        public double P99 { get; set; }
        public int Count { get; set; }
        public static Stat Empty => new();

        public static Stat From(IReadOnlyList<double> values)
        {
            if (values.Count == 0) return Empty;
            var sorted = values.ToArray();
            Array.Sort(sorted);
            return new Stat
            {
                Count = sorted.Length,
                Mean = sorted.Average(),
                P50 = Percentile(sorted, 50),
                P90 = Percentile(sorted, 90),
                P99 = Percentile(sorted, 99),
            };
        }

        // Nearest-rank percentile over an already-sorted array.
        private static double Percentile(double[] sorted, double p)
        {
            if (sorted.Length == 0) return double.NaN;
            if (sorted.Length == 1) return sorted[0];
            double rank = (p / 100.0) * (sorted.Length - 1);
            int lo = (int)Math.Floor(rank);
            int hi = (int)Math.Ceiling(rank);
            if (lo == hi) return sorted[lo];
            double frac = rank - lo;
            return sorted[lo] + frac * (sorted[hi] - sorted[lo]); // linear interpolation
        }
    }

    /// <summary>Computes the aggregate report from raw per-request results and the measured wall-clock duration.</summary>
    public static BenchmarkReport Compute(BenchmarkOptions o, IReadOnlyList<RequestResult> results, double durationSec)
    {
        var ok = results.Where(r => r.Success).ToList();
        var report = new BenchmarkReport
        {
            Backend = o.Backend + (o.Backend == "openai" ? $"/{o.Mode}" : ""),
            Model = o.Model,
            DurationSec = durationSec,
            Completed = ok.Count,
            Failed = results.Count - ok.Count,
            Streaming = ok.Any(r => r.TtftMs.HasValue),
            SlaTtftMs = o.SlaTtftMs,
            SlaTpotMs = o.SlaTpotMs,
        };

        report.TotalPromptTokens = ok.Sum(r => (long)r.PromptTokens);
        report.TotalOutputTokens = ok.Sum(r => (long)r.OutputTokens);

        if (durationSec > 0)
        {
            report.RequestThroughput = ok.Count / durationSec;
            report.OutputThroughput = report.TotalOutputTokens / durationSec;
            report.TotalTokenThroughput = (report.TotalPromptTokens + report.TotalOutputTokens) / durationSec;
        }

        report.Ttft = Stat.From(ok.Where(r => r.TtftMs.HasValue).Select(r => r.TtftMs!.Value).ToList());
        report.Tpot = Stat.From(ok.Where(r => r.TpotMs.HasValue).Select(r => r.TpotMs!.Value).ToList());
        report.Itl = Stat.From(ok.SelectMany(r => r.InterTokenLatencies()).ToList());
        report.E2E = Stat.From(ok.Select(r => r.E2EMs).ToList());

        // Goodput: requests that met BOTH SLAs, normalized by wall-clock.
        int good = ok.Count(r =>
            (r.TtftMs ?? 0) <= o.SlaTtftMs &&
            (r.TpotMs ?? 0) <= o.SlaTpotMs);
        report.GoodputPerSec = durationSec > 0 ? good / durationSec : 0;

        return report;
    }

    private static string F(double v) => double.IsNaN(v) ? "   n/a" : v.ToString("0.00", CultureInfo.InvariantCulture);

    /// <summary>Renders a console-friendly report block.</summary>
    public string ToConsole()
    {
        var sb = new StringBuilder();
        sb.AppendLine("============================ RESULTS ============================");
        sb.AppendLine($"  backend                : {Backend}");
        sb.AppendLine($"  model                  : {Model}");
        sb.AppendLine($"  duration               : {DurationSec.ToString("0.00", CultureInfo.InvariantCulture)} s");
        sb.AppendLine($"  completed / failed     : {Completed} / {Failed}");
        sb.AppendLine($"  streaming metrics      : {(Streaming ? "yes" : "NO (backend is non-streaming: TTFT/ITL/TPOT unavailable)")}");
        sb.AppendLine($"  prompt / output tokens : {TotalPromptTokens} / {TotalOutputTokens}");
        sb.AppendLine("  --------------------------------------------------------------");
        sb.AppendLine($"  request throughput     : {RequestThroughput.ToString("0.00", CultureInfo.InvariantCulture)} req/s");
        sb.AppendLine($"  output throughput      : {OutputThroughput.ToString("0.00", CultureInfo.InvariantCulture)} tok/s");
        sb.AppendLine($"  total token throughput : {TotalTokenThroughput.ToString("0.00", CultureInfo.InvariantCulture)} tok/s");
        sb.AppendLine("  --------------------------------------------------------------");
        sb.AppendLine("  latency (ms)             mean      p50      p90      p99");
        sb.AppendLine($"  TTFT                  {F(Ttft.Mean),8} {F(Ttft.P50),8} {F(Ttft.P90),8} {F(Ttft.P99),8}");
        sb.AppendLine($"  TPOT (per req)        {F(Tpot.Mean),8} {F(Tpot.P50),8} {F(Tpot.P90),8} {F(Tpot.P99),8}");
        sb.AppendLine($"  ITL  (per token)      {F(Itl.Mean),8} {F(Itl.P50),8} {F(Itl.P90),8} {F(Itl.P99),8}");
        sb.AppendLine($"  E2E                   {F(E2E.Mean),8} {F(E2E.P50),8} {F(E2E.P90),8} {F(E2E.P99),8}");
        sb.AppendLine("  --------------------------------------------------------------");
        sb.AppendLine($"  goodput (TTFT<={SlaTtftMs}ms, TPOT<={SlaTpotMs}ms) : {GoodputPerSec.ToString("0.00", CultureInfo.InvariantCulture)} req/s");
        sb.Append("================================================================");
        return sb.ToString();
    }

    /// <summary>Serializes the report to indented JSON.</summary>
    public string ToJson() =>
        JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
}
