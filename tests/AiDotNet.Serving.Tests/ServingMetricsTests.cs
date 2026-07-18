using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Observability;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Verifies the native Prometheus text-exposition renderer for serving metrics. <see cref="ServingMetrics"/>
/// holds process-wide static state; xUnit runs the methods of a single test class sequentially and no other
/// test class touches it, so <see cref="ServingMetrics.Reset"/> at the top of each test keeps them isolated.
/// </summary>
public sealed class ServingMetricsTests
{
    [Fact]
    public void RenderPrometheus_Counters_ReflectRecordedRequests()
    {
        ServingMetrics.Reset();
        ServingMetrics.RecordRequest(success: true, promptTokens: 10, generationTokens: 5, durationSeconds: 0.2);
        ServingMetrics.RecordRequest(success: true, promptTokens: 7, generationTokens: 3, durationSeconds: 0.1);
        ServingMetrics.RecordRequest(success: false, promptTokens: 0, generationTokens: 0, durationSeconds: 0.05);

        string page = ServingMetrics.RenderPrometheus();

        Assert.Contains("aidotnet_serving_requests_total{outcome=\"success\"} 2", page);
        Assert.Contains("aidotnet_serving_requests_total{outcome=\"error\"} 1", page);
        Assert.Contains("aidotnet_serving_prompt_tokens_total 17", page);
        Assert.Contains("aidotnet_serving_generation_tokens_total 8", page);

        // Every series must declare its HELP/TYPE.
        Assert.Contains("# TYPE aidotnet_serving_requests_total counter", page);
        Assert.Contains("# TYPE aidotnet_serving_request_duration_seconds histogram", page);
    }

    [Fact]
    public void RenderPrometheus_Histogram_BucketsAreCumulativeAndCountMatches()
    {
        ServingMetrics.Reset();
        // Durations 0.2s and 0.1s: both <= the 0.25 and +Inf buckets; the 0.05 bucket should hold 0.
        ServingMetrics.RecordRequest(success: true, promptTokens: 1, generationTokens: 1, durationSeconds: 0.2);
        ServingMetrics.RecordRequest(success: true, promptTokens: 1, generationTokens: 1, durationSeconds: 0.1);

        string page = ServingMetrics.RenderPrometheus();
        var lines = page.Split('\n');

        // le="0.05" is below both observations -> 0.
        Assert.Contains("aidotnet_serving_request_duration_seconds_bucket{le=\"0.05\"} 0", lines);
        // le="0.25" is above both -> cumulative 2.
        Assert.Contains("aidotnet_serving_request_duration_seconds_bucket{le=\"0.25\"} 2", lines);
        // +Inf and _count must equal the total number of observations.
        Assert.Contains("aidotnet_serving_request_duration_seconds_bucket{le=\"+Inf\"} 2", lines);
        Assert.Contains("aidotnet_serving_request_duration_seconds_count 2", lines);

        // _sum ~= 0.3 seconds.
        var sumLine = lines.First(l => l.StartsWith("aidotnet_serving_request_duration_seconds_sum "));
        double sum = double.Parse(sumLine.Split(' ')[1], System.Globalization.CultureInfo.InvariantCulture);
        Assert.InRange(sum, 0.29, 0.31);
    }

    [Fact]
    public void RenderPrometheus_StreamingRequest_RecordsTtftAndTpot()
    {
        ServingMetrics.Reset();
        ServingMetrics.RecordRequest(success: true, promptTokens: 4, generationTokens: 6,
            durationSeconds: 0.5, ttftSeconds: 0.05, tpotSeconds: 0.02);

        string page = ServingMetrics.RenderPrometheus();

        Assert.Contains("aidotnet_serving_time_to_first_token_seconds_count 1", page);
        Assert.Contains("aidotnet_serving_time_per_output_token_seconds_count 1", page);
    }

    [Fact]
    public void RenderPrometheus_BatcherGauges_AreFoldedInFromSnapshot()
    {
        ServingMetrics.Reset();
        var snapshot = new Dictionary<string, object>
        {
            ["throughputRequestsPerSecond"] = 42.5,
            ["averageBatchSize"] = 8.0,
            ["averageQueueDepth"] = 3.0,
            ["batchUtilizationPercent"] = 91.25,
            ["latencyP95Ms"] = 120.0,
            ["uptimeSeconds"] = 60.0
        };

        string page = ServingMetrics.RenderPrometheus(snapshot);

        Assert.Contains("# TYPE aidotnet_serving_throughput_requests_per_second gauge", page);
        Assert.Contains("aidotnet_serving_throughput_requests_per_second 42.5", page);
        Assert.Contains("aidotnet_serving_batch_utilization_percent 91.25", page);
        Assert.Contains("aidotnet_serving_latency_p95_ms 120", page);
    }
}
