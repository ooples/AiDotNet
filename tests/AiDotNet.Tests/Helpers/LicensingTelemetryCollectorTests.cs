using System.Net;
using System.Net.Http;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests for <see cref="LicensingTelemetryCollector"/> covering event queuing,
/// enable/disable behavior, flush mechanics, and HTTP sending.
/// </summary>
public class LicensingTelemetryCollectorTests : IDisposable
{
    private readonly LicensingTelemetryCollector _collector;
    private readonly FakeHttpMessageHandler _handler;

    public LicensingTelemetryCollectorTests()
    {
        _handler = new FakeHttpMessageHandler();
        var httpClient = new HttpClient(_handler);
        _collector = new LicensingTelemetryCollector(httpClient, flushIntervalMs: 60_000, enabled: true);
        _collector.EndpointUrl = "https://example.com/telemetry";
        _collector.AnonKey = "test-anon-key";
    }

    public void Dispose()
    {
        _collector.Dispose();
    }

    [Fact]
    public void IsEnabled_DefaultTrue_WhenConstructedEnabled()
    {
        Assert.True(_collector.IsEnabled);
    }

    [Fact]
    public void Disable_StopsQueuing()
    {
        _collector.Disable();
        _collector.RecordTrialOperation(1, 9, 0);

        Assert.Equal(0, _collector.QueueCount);
    }

    [Fact]
    public void Enable_ResumesQueuing()
    {
        _collector.Disable();
        _collector.RecordTrialOperation(1, 9, 0);
        Assert.Equal(0, _collector.QueueCount);

        _collector.Enable();
        _collector.RecordTrialOperation(2, 8, 0);
        Assert.Equal(1, _collector.QueueCount);
    }

    [Fact]
    public void RecordTrialOperation_QueuesEvent()
    {
        _collector.RecordTrialOperation(3, 7, 1);
        Assert.Equal(1, _collector.QueueCount);
    }

    [Fact]
    public void RecordTrialExpired_QueuesEvent()
    {
        _collector.RecordTrialExpired("OperationLimitReached", 10, 5);
        Assert.Equal(1, _collector.QueueCount);
    }

    [Fact]
    public void RecordLicensedOperation_QueuesEvent()
    {
        _collector.RecordLicensedOperation("save");
        Assert.Equal(1, _collector.QueueCount);
    }

    [Fact]
    public void RecordLicensingError_QueuesEvent()
    {
        _collector.RecordLicensingError("tampered_file");
        Assert.Equal(1, _collector.QueueCount);
    }

    [Fact]
    public async Task FlushAsync_SendsEvents_ClearsQueue()
    {
        _collector.RecordTrialOperation(1, 9, 0);
        _collector.RecordTrialOperation(2, 8, 0);
        Assert.Equal(2, _collector.QueueCount);

        int flushed = await _collector.FlushAsync();

        Assert.Equal(2, flushed);
        Assert.Equal(0, _collector.QueueCount);
        Assert.Equal(1, _handler.RequestCount);
    }

    [Fact]
    public async Task FlushAsync_EmptyQueue_Returns0()
    {
        int flushed = await _collector.FlushAsync();
        Assert.Equal(0, flushed);
        Assert.Equal(0, _handler.RequestCount);
    }

    [Fact]
    public async Task FlushAsync_HttpFailure_Returns0_DoesNotThrow()
    {
        _handler.ResponseStatusCode = HttpStatusCode.InternalServerError;
        _handler.ThrowOnSend = true;

        _collector.RecordTrialOperation(1, 9, 0);

        // Should not throw even when HTTP fails
        int flushed = await _collector.FlushAsync();
        Assert.Equal(0, flushed);
    }

    [Fact]
    public async Task FlushAsync_SendsCorrectHeaders()
    {
        _collector.RecordTrialOperation(1, 9, 0);
        await _collector.FlushAsync();

        Assert.NotNull(_handler.LastRequest);
        Assert.True(_handler.LastRequest.Headers.Contains("apikey"));
        Assert.Contains("test-anon-key", _handler.LastRequest.Headers.GetValues("apikey"));
        Assert.True(_handler.LastRequest.Headers.Contains("Prefer"));
    }

    [Fact]
    public async Task FlushAsync_SendsJsonBody()
    {
        _collector.RecordTrialOperation(1, 9, 0);
        await _collector.FlushAsync();

        Assert.NotNull(_handler.LastRequestBody);
        Assert.Contains("trial_operation", _handler.LastRequestBody);
        Assert.Contains("operation_count", _handler.LastRequestBody);
    }

    [Fact]
    public void Flush_Synchronous_Works()
    {
        _collector.RecordTrialOperation(1, 9, 0);
        int flushed = _collector.Flush();

        Assert.Equal(1, flushed);
        Assert.Equal(0, _collector.QueueCount);
    }

    [Fact]
    public void MultipleEventTypes_AllQueued()
    {
        _collector.RecordTrialOperation(1, 9, 0);
        _collector.RecordTrialExpired("TimeExpired", 5, 31);
        _collector.RecordLicensedOperation("load");
        _collector.RecordLicensingError("corrupted");

        Assert.Equal(4, _collector.QueueCount);
    }

    [Fact]
    public async Task FlushAsync_NoEndpoint_DoesNotThrow()
    {
        _collector.EndpointUrl = "";
        _collector.RecordTrialOperation(1, 9, 0);

        int flushed = await _collector.FlushAsync();
        // Events are dequeued but not sent (no endpoint)
        Assert.Equal(0, _collector.QueueCount);
    }

    [Fact]
    public async Task FlushAsync_NoAnonKey_DoesNotThrow()
    {
        _collector.AnonKey = "";
        _collector.RecordTrialOperation(1, 9, 0);

        int flushed = await _collector.FlushAsync();
        Assert.Equal(0, _collector.QueueCount);
    }

    [Fact]
    public void Disposed_DoesNotQueueEvents()
    {
        _collector.Dispose();
        _collector.RecordTrialOperation(1, 9, 0);
        Assert.Equal(0, _collector.QueueCount);
    }

    /// <summary>
    /// A fake HTTP handler for testing without real network calls.
    /// </summary>
    private sealed class FakeHttpMessageHandler : HttpMessageHandler
    {
        public int RequestCount { get; private set; }
        public HttpRequestMessage? LastRequest { get; private set; }
        public string? LastRequestBody { get; private set; }
        public HttpStatusCode ResponseStatusCode { get; set; } = HttpStatusCode.Created;
        public bool ThrowOnSend { get; set; }

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (ThrowOnSend)
            {
                throw new HttpRequestException("Simulated network failure");
            }

            RequestCount++;
            LastRequest = request;
            if (request.Content != null)
            {
                LastRequestBody = await request.Content.ReadAsStringAsync().ConfigureAwait(false);
            }

            return new HttpResponseMessage(ResponseStatusCode);
        }
    }
}
