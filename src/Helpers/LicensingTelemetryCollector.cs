using System.Collections.Concurrent;
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.Helpers;

/// <summary>
/// Collects anonymous licensing telemetry events and periodically flushes them
/// to the AiDotNet telemetry endpoint.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class helps AiDotNet understand how the library is used
/// by collecting anonymous metrics like trial usage, model save/load counts, and error rates.
/// No personal data, model weights, or predictions are ever collected.</para>
///
/// <para><b>Privacy:</b> Telemetry is opt-in. Set <c>AIDOTNET_TELEMETRY=false</c> environment
/// variable or call <see cref="Disable"/> to prevent any data from being sent. Events are
/// batched and flushed periodically to minimize network overhead.</para>
///
/// <para><b>Design:</b> This is a singleton collector accessed via <see cref="Instance"/>.
/// Events are queued in a thread-safe concurrent queue and flushed every 60 seconds
/// (configurable) or when the queue reaches the batch size limit.</para>
/// </remarks>
internal sealed class LicensingTelemetryCollector : IDisposable
{
    /// <summary>
    /// The default flush interval in seconds.
    /// </summary>
    internal const int DefaultFlushIntervalSeconds = 60;

    /// <summary>
    /// Maximum events to queue before triggering an early flush.
    /// </summary>
    internal const int MaxQueueSize = 100;

    /// <summary>
    /// Maximum events to send in a single HTTP request.
    /// </summary>
    internal const int MaxBatchSize = 50;

    private static readonly Lazy<LicensingTelemetryCollector> LazyInstance =
        new(() => new LicensingTelemetryCollector(), LazyThreadSafetyMode.ExecutionAndPublication);

    /// <summary>
    /// Gets the singleton instance of the telemetry collector.
    /// </summary>
    internal static LicensingTelemetryCollector Instance => LazyInstance.Value;

    private readonly ConcurrentQueue<TelemetryEvent> _queue = new();
    private readonly Timer _flushTimer;
    private readonly HttpClient _httpClient;
    private readonly string _machineId;
    private volatile bool _enabled;
    private volatile bool _disposed;
    private int _flushIntervalMs;

    /// <summary>
    /// The telemetry endpoint URL. Can be overridden for testing.
    /// </summary>
    internal string EndpointUrl { get; set; } = "https://yfkqwpgjahoamlgckjib.supabase.co/rest/v1/telemetry_events";

    /// <summary>
    /// The Supabase anon key for telemetry insertion. This is a public key
    /// that only has INSERT permission on the telemetry_events table via RLS.
    /// </summary>
    internal string AnonKey { get; set; } = string.Empty;

    private LicensingTelemetryCollector()
    {
        _machineId = MachineFingerprint.GetMachineId();
        _enabled = ResolveTelemetryEnabled();
        _flushIntervalMs = DefaultFlushIntervalSeconds * 1000;

        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("Accept", "application/json");

        // Timer fires at the flush interval; the callback drains the queue
        _flushTimer = new Timer(FlushCallback, null, _flushIntervalMs, _flushIntervalMs);
    }

    /// <summary>
    /// Creates a new instance with custom settings. For testing only.
    /// </summary>
    internal LicensingTelemetryCollector(HttpClient httpClient, int flushIntervalMs, bool enabled)
    {
        _machineId = MachineFingerprint.GetMachineId();
        _enabled = enabled;
        _flushIntervalMs = flushIntervalMs;
        _httpClient = httpClient;

        _flushTimer = new Timer(FlushCallback, null, _flushIntervalMs, _flushIntervalMs);
    }

    /// <summary>
    /// Gets whether telemetry collection is enabled.
    /// </summary>
    internal bool IsEnabled => _enabled;

    /// <summary>
    /// Gets the number of queued events waiting to be flushed.
    /// </summary>
    internal int QueueCount => _queue.Count;

    /// <summary>
    /// Disables telemetry collection. No further events will be queued or sent.
    /// </summary>
    internal void Disable()
    {
        _enabled = false;
    }

    /// <summary>
    /// Enables telemetry collection.
    /// </summary>
    internal void Enable()
    {
        _enabled = true;
    }

    /// <summary>
    /// Records a trial operation event.
    /// </summary>
    /// <param name="operationCount">Current operation count after this operation.</param>
    /// <param name="operationsRemaining">Remaining operations in the trial.</param>
    /// <param name="daysElapsed">Days since trial started.</param>
    internal void RecordTrialOperation(int operationCount, int operationsRemaining, int daysElapsed)
    {
        Enqueue(new TelemetryEvent
        {
            EventType = "trial_operation",
            MachineIdHash = HashMachineId(),
            Properties = new Dictionary<string, object>
            {
                ["operation_count"] = operationCount,
                ["operations_remaining"] = operationsRemaining,
                ["days_elapsed"] = daysElapsed
            }
        });
    }

    /// <summary>
    /// Records a trial expiration event.
    /// </summary>
    /// <param name="reason">The reason the trial expired.</param>
    /// <param name="operationsPerformed">Total operations performed.</param>
    /// <param name="daysElapsed">Days since trial started.</param>
    internal void RecordTrialExpired(string reason, int operationsPerformed, int daysElapsed)
    {
        Enqueue(new TelemetryEvent
        {
            EventType = "trial_expired",
            MachineIdHash = HashMachineId(),
            Properties = new Dictionary<string, object>
            {
                ["reason"] = reason,
                ["operations_performed"] = operationsPerformed,
                ["days_elapsed"] = daysElapsed
            }
        });
    }

    /// <summary>
    /// Records a licensed operation event.
    /// </summary>
    /// <param name="operationType">The type of operation (save, load, serialize, deserialize).</param>
    internal void RecordLicensedOperation(string operationType)
    {
        Enqueue(new TelemetryEvent
        {
            EventType = "licensed_operation",
            MachineIdHash = HashMachineId(),
            Properties = new Dictionary<string, object>
            {
                ["operation_type"] = operationType
            }
        });
    }

    /// <summary>
    /// Records a licensing error event.
    /// </summary>
    /// <param name="errorType">The type of error that occurred.</param>
    internal void RecordLicensingError(string errorType)
    {
        Enqueue(new TelemetryEvent
        {
            EventType = "licensing_error",
            MachineIdHash = HashMachineId(),
            Properties = new Dictionary<string, object>
            {
                ["error_type"] = errorType
            }
        });
    }

    /// <summary>
    /// Immediately flushes all queued events. Returns the number of events flushed.
    /// </summary>
    internal async Task<int> FlushAsync()
    {
        if (_disposed || _queue.IsEmpty)
        {
            return 0;
        }

        var batch = DequeueBatch(MaxBatchSize);
        if (batch.Count == 0)
        {
            return 0;
        }

        try
        {
            await SendBatchAsync(batch).ConfigureAwait(false);
            return batch.Count;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or OperationCanceledException)
        {
            // Silently drop on network failures — telemetry is best-effort
            System.Diagnostics.Trace.TraceWarning(
                "LicensingTelemetry: failed to send batch ({0} events): {1}", batch.Count, ex.Message);
            return 0;
        }
    }

    /// <summary>
    /// Synchronous flush for use in the timer callback.
    /// </summary>
    internal int Flush()
    {
        try
        {
            return FlushAsync().ConfigureAwait(false).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicensingTelemetryCollector: flush failed: " + ex.GetType().Name + ": " + ex.Message);
            return 0;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _flushTimer.Dispose();

        // Best-effort final flush
        try
        {
            Flush();
        }
        catch
        {
            // Swallow — we're shutting down
        }

        _httpClient.Dispose();
    }

    private void Enqueue(TelemetryEvent telemetryEvent)
    {
        if (!_enabled || _disposed) return;

        telemetryEvent.TimestampUtc = DateTimeOffset.UtcNow;
        telemetryEvent.LibraryVersion = GetLibraryVersion();
        _queue.Enqueue(telemetryEvent);

        // Trigger early flush if queue is getting large
        if (_queue.Count >= MaxQueueSize)
        {
            ThreadPool.QueueUserWorkItem(_ => Flush());
        }
    }

    private List<TelemetryEvent> DequeueBatch(int maxCount)
    {
        var batch = new List<TelemetryEvent>(Math.Min(maxCount, _queue.Count));
        while (batch.Count < maxCount && _queue.TryDequeue(out var ev))
        {
            batch.Add(ev);
        }
        return batch;
    }

    private async Task SendBatchAsync(List<TelemetryEvent> batch)
    {
        if (string.IsNullOrWhiteSpace(AnonKey) || string.IsNullOrWhiteSpace(EndpointUrl))
        {
            return;
        }

        var payload = batch.Select(e => new
        {
            event_type = e.EventType,
            machine_id_hash = e.MachineIdHash,
            library_version = e.LibraryVersion,
            timestamp_utc = e.TimestampUtc.ToString("O"),
            properties = e.Properties
        });

        string json = JsonConvert.SerializeObject(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var request = new HttpRequestMessage(HttpMethod.Post, EndpointUrl);
        request.Content = content;
        request.Headers.Add("apikey", AnonKey);
        request.Headers.Add("Prefer", "return=minimal");

        using var response = await _httpClient.SendAsync(request).ConfigureAwait(false);
        // We don't check response — telemetry is fire-and-forget
    }

    private void FlushCallback(object? state)
    {
        if (_disposed || _queue.IsEmpty) return;

        try
        {
            Flush();
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                "LicensingTelemetryCollector: flush failed: " + ex.GetType().Name + ": " + ex.Message);
        }
    }

    private string HashMachineId()
    {
        // One-way hash of machine ID for anonymity
        using var sha = System.Security.Cryptography.SHA256.Create();
        byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes("telemetry:" + _machineId));
        return Convert.ToBase64String(hash, 0, 16); // First 16 bytes = 128-bit hash
    }

    private static string GetLibraryVersion()
    {
        try
        {
            var assembly = typeof(LicensingTelemetryCollector).Assembly;
            var version = assembly.GetName().Version;
            return version?.ToString(3) ?? "0.0.0";
        }
        catch
        {
            return "0.0.0";
        }
    }

    private static bool ResolveTelemetryEnabled()
    {
        // Check environment variable first
        string? envValue = Environment.GetEnvironmentVariable("AIDOTNET_TELEMETRY");
        if (envValue is not null)
        {
            return envValue.Trim().Equals("true", StringComparison.OrdinalIgnoreCase)
                || envValue.Trim() == "1";
        }

        // Default: disabled (opt-in)
        return false;
    }

    /// <summary>
    /// Represents an anonymous telemetry event.
    /// </summary>
    internal sealed class TelemetryEvent
    {
        /// <summary>Event type (e.g., "trial_operation", "trial_expired", "licensed_operation").</summary>
        public string EventType { get; set; } = string.Empty;

        /// <summary>One-way hash of the machine fingerprint for deduplication (not PII).</summary>
        public string MachineIdHash { get; set; } = string.Empty;

        /// <summary>Library version string.</summary>
        public string LibraryVersion { get; set; } = string.Empty;

        /// <summary>UTC timestamp of the event.</summary>
        public DateTimeOffset TimestampUtc { get; set; }

        /// <summary>Event-specific properties.</summary>
        public Dictionary<string, object> Properties { get; set; } = new();
    }
}
