using AiDotNet.Serving.Observability;
using AiDotNet.Serving.Services;
using AiDotNet.Validation;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AiDotNet.Serving.Controllers;

/// <summary>
/// Prometheus metrics scrape endpoint (<c>/metrics</c>), matching how vLLM and TGI expose serving telemetry.
/// Returns the Prometheus text-exposition format so a standard Prometheus/Grafana stack works out of the box.
/// </summary>
/// <remarks>
/// Anonymous by design: Prometheus scrapers typically poll without application credentials, and this is the
/// industry-standard behavior for vLLM/TGI. Restrict access at the network layer (scrape from within the
/// cluster / behind a firewall) rather than requiring an API key.
/// </remarks>
[ApiController]
[AllowAnonymous]
public sealed class MetricsController : ControllerBase
{
    private readonly IRequestBatcher _requestBatcher;

    public MetricsController(IRequestBatcher requestBatcher)
    {
        Guard.NotNull(requestBatcher);
        _requestBatcher = requestBatcher;
    }

    /// <summary>Renders the current serving metrics in Prometheus text-exposition format (version 0.0.4).</summary>
    [HttpGet("/metrics")]
    [Produces("text/plain")]
    public IActionResult Scrape()
    {
        // Fold in the running batcher's live performance snapshot (throughput, queue depth, batch
        // utilization, latency percentiles) alongside the process-wide request counters/histograms.
        var batcherMetrics = _requestBatcher.GetPerformanceMetrics();
        string body = ServingMetrics.RenderPrometheus(batcherMetrics);
        return Content(body, "text/plain; version=0.0.4; charset=utf-8");
    }
}
