using System.Collections.Specialized;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using AiDotNet.Interfaces;
using Newtonsoft.Json;

namespace AiDotNet.Dashboard.Dashboard;

/// <summary>
/// The opt-in local web view of <see cref="TrainingMonitorDashboard{T}"/>.
/// </summary>
public partial class TrainingMonitorDashboard<T>
{
    /// <summary>Environment variable read by <see cref="StartWebServerFromEnvironment"/>: the port to serve on
    /// (<c>0</c> picks a free port).</summary>
    public const string PortEnvironmentVariable = "AIDOTNET_DASHBOARD_PORT";

    /// <summary>Environment variable read by <see cref="StartWebServerFromEnvironment"/>: the host to bind
    /// (defaults to <c>127.0.0.1</c>).</summary>
    public const string HostEnvironmentVariable = "AIDOTNET_DASHBOARD_HOST";

    private const int MaxMessagesPerResponse = 50;
    private const int MaxAlertsPerResponse = 20;
    // A metric's newest points in one response: a first poll of a long run copies at most this many under the
    // training lock (the page keeps the same bound).
    private const int MaxPointsPerMetricResponse = 4096;
    // Requests served at once; more are answered 503 instead of queueing without bound on the thread pool.
    private const int MaxConcurrentRequests = 8;
    private readonly SemaphoreSlim _requestSlots = new(MaxConcurrentRequests, MaxConcurrentRequests);
    // The host the server was bound to; a request's Host header must name it, a loopback name, an IP literal or
    // this machine (see IsAllowedHost).
    private volatile string _boundHost = "127.0.0.1";

    private static readonly JsonSerializerSettings WebJsonSettings = new()
    {
        // NaN and infinities are valid metric values (a diverged loss); JSON has no literal for them.
        FloatFormatHandling = FloatFormatHandling.String,
        NullValueHandling = NullValueHandling.Include,
    };

    private readonly object _webLock = new();
    private HttpListener? _listener;
    private Thread? _listenerThread;
    private string? _webServerUrl;

    /// <summary>
    /// Gets the URL of the running web page, or <c>null</c> when the web server is not running.
    /// </summary>
    public string? WebServerUrl
    {
        get
        {
            lock (_webLock) return _webServerUrl;
        }
    }

    /// <summary>
    /// Starts a local web server that shows every session of this monitor as a live page.
    /// </summary>
    /// <param name="port">The TCP port to listen on; <c>0</c> (the default) picks a free port.</param>
    /// <param name="host">The host to bind. The default <c>127.0.0.1</c> serves this machine only.</param>
    /// <returns>The URL of the page, for example <c>http://127.0.0.1:53817/</c>.</returns>
    /// <remarks>
    /// <para>Endpoints:</para>
    /// <list type="bullet">
    /// <item><description><c>/</c>: a self-contained HTML page (no external assets) with one chart per metric that
    /// refreshes every 2 seconds, the current values, elapsed time, step progress, recent messages and alerts.</description></item>
    /// <item><description><c>/api/sessions</c>: every session as JSON.</description></item>
    /// <item><description><c>/api/metrics?session=&lt;id&gt;&amp;since=&lt;step&gt;</c>: the session's metric points
    /// with a step greater than <c>since</c> (all when omitted). <c>seq=&lt;n&gt;</c> instead returns the points
    /// logged after the response that reported sequence <c>n</c>, whatever their step.</description></item>
    /// </list>
    /// <para>Calling it again while running returns the current URL. The server has no authentication: binding a
    /// host other than loopback exposes the training metrics to the network (and on Windows needs a URL
    /// reservation, <c>netsh http add urlacl</c>).</para>
    /// <para>Against DNS rebinding (a web page that points its own host name at this machine to read the metrics), a
    /// request is served only when its address names a loopback host, an IP address, the bound host itself or this
    /// machine's own name. A server bound to every address ("+", "*", "0.0.0.0") is therefore opened by IP address,
    /// by <c>localhost</c> or by the machine's name, not by another DNS alias (an FQDN or mDNS name gets 403).</para>
    /// <para><b>For Beginners:</b> call this once, open the returned address in a browser, and watch training live
    /// even when the process runs in the background without a console.</para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="port"/> is outside 0..65535.</exception>
    /// <exception cref="ObjectDisposedException">The dashboard is disposed.</exception>
    /// <exception cref="PlatformNotSupportedException">The platform has no <see cref="HttpListener"/>.</exception>
    public string StartWebServer(int port = 0, string host = "127.0.0.1")
    {
        if (port < 0 || port > 65535)
            throw new ArgumentOutOfRangeException(nameof(port), "Port must be between 0 and 65535.");
        if (string.IsNullOrWhiteSpace(host))
            host = "127.0.0.1";
        if (!HttpListener.IsSupported)
            throw new PlatformNotSupportedException("HttpListener is not supported on this platform.");

        lock (_webLock)
        {
            // Under the lock: Dispose sets the flag before it stops the server, so a start racing it either sees
            // the flag or is stopped by it.
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(TrainingMonitorDashboard<T>));
            if (_listener is not null && _webServerUrl is not null)
                return _webServerUrl;

            // HttpListener cannot bind port 0, so a free port is found first; another process can take it in
            // between, hence the retries.
            int attempts = port == 0 ? 5 : 1;
            Exception? lastError = null;
            for (int attempt = 0; attempt < attempts; attempt++)
            {
                int boundPort = port == 0 ? FindFreeLoopbackPort() : port;
                foreach (var prefixHost in PrefixHostsFor(host))
                {
                    var prefix = $"http://{prefixHost}:{boundPort.ToString(CultureInfo.InvariantCulture)}/";
                    var listener = new HttpListener();
                    listener.Prefixes.Add(prefix);
                    try
                    {
                        listener.Start();
                    }
                    catch (HttpListenerException ex)
                    {
                        lastError = ex;
                        listener.Close();
                        continue;
                    }

                    _listener = listener;
                    _webServerUrl = prefix;
                    _boundHost = host;
                    _listenerThread = new Thread(() => ListenLoop(listener))
                    {
                        IsBackground = true,
                        Name = "AiDotNet training dashboard web server"
                    };
                    _listenerThread.Start();
                    return prefix;
                }
            }

            throw new InvalidOperationException(
                $"Could not start the dashboard web server on {host}:{port}: {lastError?.Message}", lastError);
        }
    }

    /// <summary>
    /// Stops the web server started by <see cref="StartWebServer"/> and frees its port. Does nothing when it is not
    /// running.
    /// </summary>
    public void StopWebServer()
    {
        HttpListener? listener;
        Thread? thread;
        lock (_webLock)
        {
            listener = _listener;
            thread = _listenerThread;
            _listener = null;
            _listenerThread = null;
            _webServerUrl = null;
        }

        if (listener is null)
            return;

        try
        {
            listener.Stop();
        }
        catch (ObjectDisposedException)
        {
        }
        finally
        {
            listener.Close();
        }

        if (thread is not null && thread != Thread.CurrentThread)
            thread.Join(TimeSpan.FromSeconds(5));
    }

    /// <summary>
    /// Starts the web server when the <c>AIDOTNET_DASHBOARD_PORT</c> environment variable is set, so a process can
    /// enable the live page without code changes. <c>AIDOTNET_DASHBOARD_HOST</c> optionally sets the host.
    /// </summary>
    /// <returns>The page URL, or <c>null</c> when the variable is not set.</returns>
    /// <exception cref="FormatException">The variable is set but is not a port number.</exception>
    public string? TryStartWebServerFromEnvironment()
    {
        var portText = Environment.GetEnvironmentVariable(PortEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(portText))
            return null;
        if (!int.TryParse(portText.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int port)
            || port < 0 || port > 65535)
        {
            throw new FormatException(
                $"{PortEnvironmentVariable} must be a port number between 0 and 65535, got '{portText}'.");
        }

        var host = Environment.GetEnvironmentVariable(HostEnvironmentVariable);
        return StartWebServer(port, string.IsNullOrWhiteSpace(host) ? "127.0.0.1" : host!.Trim());
    }

    /// <summary>
    /// Creates a dashboard and starts its web server when the <c>AIDOTNET_DASHBOARD_PORT</c> environment variable is
    /// set (see <see cref="TryStartWebServerFromEnvironment"/>).
    /// </summary>
    /// <param name="defaultName">Default name for training sessions.</param>
    /// <returns>A dashboard serving its page, or <c>null</c> when the variable is not set.</returns>
    public static TrainingMonitorDashboard<T>? StartWebServerFromEnvironment(string defaultName = "Training")
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(PortEnvironmentVariable)))
            return null;

        var dashboard = new TrainingMonitorDashboard<T>(defaultName);
        try
        {
            dashboard.TryStartWebServerFromEnvironment();
            return dashboard;
        }
        catch
        {
            dashboard.Dispose();
            throw;
        }
    }

    private static IEnumerable<string> PrefixHostsFor(string host)
    {
        yield return host.Contains(":") && !host.StartsWith("[", StringComparison.Ordinal) ? $"[{host}]" : host;
        // On Windows a non-administrator may register http://localhost:port/ but not http://127.0.0.1:port/
        // (http.sys URL ACLs); "localhost" binds loopback only, so it is an equivalent fallback.
        if (host == "127.0.0.1" || host == "::1" || host == "[::1]")
            yield return "localhost";
    }

    /// <summary>
    /// Whether a request whose Host header is <paramref name="hostHeader"/> may be served by a server bound to
    /// <paramref name="boundHost"/>. DNS rebinding points an attacker's own DNS name at this machine, so its requests
    /// carry that name as Host: only names that cannot be rebound are accepted, namely loopback names, IP literals, the
    /// bound host itself and this machine's own host name (exactly: a name that merely starts with it, such as
    /// "machine.attacker.example", is a name the attacker controls). That holds for a wildcard bind ("+", "*", "0.0.0.0") too,
    /// which a listener serves on every address including loopback. A request without a Host header is refused.
    /// </summary>
    private static bool IsAllowedHost(string? hostHeader, string boundHost)
    {
        string? name = HostNameOf(hostHeader);
        if (string.IsNullOrEmpty(name)) return false;
        string bare = name!.Trim().TrimStart('[').TrimEnd(']');
        // A fully qualified name's trailing dot names the same host ("localhost." is localhost).
        if (bare.EndsWith(".", StringComparison.Ordinal)) bare = bare.Substring(0, bare.Length - 1);
        if (bare.Length == 0) return false;
        if (IsLoopbackName(bare) || IPAddress.TryParse(bare, out _)) return true;
        string boundBare = (boundHost ?? string.Empty).Trim().TrimStart('[').TrimEnd(']');
        if (string.Equals(bare, boundBare, StringComparison.OrdinalIgnoreCase)) return true;
        foreach (var machine in s_machineNames.Value)
            if (string.Equals(bare, machine, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    // This machine's own names (looked up once): the DNS host name and the NetBIOS name.
    private static readonly Lazy<string[]> s_machineNames = new(() =>
    {
        var names = new List<string> { Environment.MachineName };
        try { names.Add(Dns.GetHostName()); }
        catch (System.Net.Sockets.SocketException) { }
        return names.ToArray();
    });

    private static bool IsLoopbackName(string? host)
    {
        if (string.IsNullOrEmpty(host)) return false;
        string h = host!.Trim().TrimStart('[').TrimEnd(']');
        return string.Equals(h, "localhost", StringComparison.OrdinalIgnoreCase)
            || (IPAddress.TryParse(h, out var address) && IPAddress.IsLoopback(address));
    }

    // The host part of a Host header ("name:port", "[v6]:port" or "name").
    private static string? HostNameOf(string? hostHeader)
    {
        if (string.IsNullOrWhiteSpace(hostHeader)) return null;
        string h = hostHeader!.Trim();
        if (h.StartsWith("[", StringComparison.Ordinal))
        {
            int close = h.IndexOf(']');
            return close > 0 ? h.Substring(0, close + 1) : h;
        }
        int colon = h.LastIndexOf(':');
        return colon > 0 && h.IndexOf(':') == colon ? h.Substring(0, colon) : h;
    }

    private static int FindFreeLoopbackPort()
    {
        var probe = new TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        try
        {
            return ((IPEndPoint)probe.LocalEndpoint).Port;
        }
        finally
        {
            probe.Stop();
        }
    }

    private void ListenLoop(HttpListener listener)
    {
        while (listener.IsListening)
        {
            HttpListenerContext context;
            try
            {
                context = listener.GetContext();
            }
            catch (HttpListenerException)
            {
                return;
            }
            catch (ObjectDisposedException)
            {
                return;
            }
            catch (InvalidOperationException)
            {
                return;
            }

            if (!_requestSlots.Wait(0))
            {
                try { Respond(context, 503, "text/plain; charset=utf-8", "Busy"); }
                catch (Exception) { /* the client is gone */ }
                continue;
            }
            ThreadPool.QueueUserWorkItem(_ =>
            {
                try { HandleRequest(context); }
                finally { _requestSlots.Release(); }
            });
        }
    }

    private void HandleRequest(HttpListenerContext context)
    {
        try
        {
            var request = context.Request;
            // A page on another origin that re-points its own host name at this machine (DNS rebinding) sends its own
            // name as Host and is refused, whatever address the server is bound to.
            if (!IsAllowedHost(request.Headers["Host"], _boundHost))
            {
                Respond(context, 403, "text/plain; charset=utf-8", "Forbidden");
                return;
            }
            var path = request.Url?.AbsolutePath ?? "/";
            if (!string.Equals(request.HttpMethod, "GET", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(request.HttpMethod, "HEAD", StringComparison.OrdinalIgnoreCase))
            {
                Respond(context, 405, "text/plain; charset=utf-8", "Method not allowed");
                return;
            }

            switch (path.TrimEnd('/').ToLowerInvariant())
            {
                case "":
                case "/index.html":
                    Respond(context, 200, "text/html; charset=utf-8", TrainingDashboardWebPage.Html);
                    break;
                case "/api/sessions":
                    Respond(context, 200, "application/json; charset=utf-8",
                        JsonConvert.SerializeObject(GetSessionsSnapshot(), WebJsonSettings));
                    break;
                case "/api/metrics":
                    HandleMetrics(context, request.QueryString);
                    break;
                default:
                    Respond(context, 404, "text/plain; charset=utf-8", "Not found");
                    break;
            }
        }
        catch (Exception ex)
        {
            try
            {
                System.Diagnostics.Trace.TraceWarning($"Training dashboard request failed: {ex}");
                Respond(context, 500, "text/plain; charset=utf-8", "Internal error");
            }
            catch (Exception)
            {
                // The client is gone or the listener stopped; nothing to report to.
            }
        }
    }

    private void HandleMetrics(HttpListenerContext context, NameValueCollection query)
    {
        var sessionId = query["session"];
        if (string.IsNullOrWhiteSpace(sessionId))
        {
            Respond(context, 400, "text/plain; charset=utf-8", "Missing 'session' query parameter.");
            return;
        }

        int? sinceStep = null;
        if (!string.IsNullOrWhiteSpace(query["since"]))
        {
            if (!int.TryParse(query["since"], NumberStyles.Integer, CultureInfo.InvariantCulture, out int since))
            {
                Respond(context, 400, "text/plain; charset=utf-8", "'since' must be an integer step.");
                return;
            }
            sinceStep = since;
        }

        long? afterSequence = null;
        if (!string.IsNullOrWhiteSpace(query["seq"]))
        {
            if (!long.TryParse(query["seq"], NumberStyles.Integer, CultureInfo.InvariantCulture, out long seq))
            {
                Respond(context, 400, "text/plain; charset=utf-8", "'seq' must be an integer.");
                return;
            }
            afterSequence = seq;
        }

        var snapshot = GetMetricsSnapshot(sessionId!, sinceStep, afterSequence);
        if (snapshot is null)
        {
            Respond(context, 404, "application/json; charset=utf-8",
                JsonConvert.SerializeObject(new { error = "session not found", session = sessionId }, WebJsonSettings));
            return;
        }

        Respond(context, 200, "application/json; charset=utf-8", JsonConvert.SerializeObject(snapshot, WebJsonSettings));
    }

    /// <summary>
    /// How long one response may take to write. A client that stops reading would otherwise hold its request slot
    /// (or, for a 503, the listener thread) for as long as the socket send blocks, which has no timeout: past this
    /// deadline the response is aborted, which closes the connection and unblocks the write. Defaults to 10 seconds.
    /// </summary>
    internal TimeSpan ResponseWriteTimeout { get; set; } = TimeSpan.FromSeconds(10);

    private const int ResponseWriting = 0;
    private const int ResponseCompleted = 1;
    private const int ResponseAborted = 2;

    private void Respond(HttpListenerContext context, int status, string contentType, string body)
    {
        var response = context.Response;
        // Exactly one of the writer (completing) and the deadline (aborting) ends the response.
        int state = ResponseWriting;
        using var deadline = new Timer(_ =>
        {
            if (Interlocked.CompareExchange(ref state, ResponseAborted, ResponseWriting) != ResponseWriting) return;
            try { response.Abort(); }
            catch (Exception) { /* the connection is already gone */ }
        }, null, Timeout.Infinite, Timeout.Infinite);
        try
        {
            var bytes = Encoding.UTF8.GetBytes(body);
            response.StatusCode = status;
            response.ContentType = contentType;
            response.Headers["Cache-Control"] = "no-store";
            response.Headers["X-Content-Type-Options"] = "nosniff";
            response.ContentLength64 = bytes.Length;
            deadline.Change(ResponseWriteTimeout, Timeout.InfiniteTimeSpan);
            if (!string.Equals(context.Request.HttpMethod, "HEAD", StringComparison.OrdinalIgnoreCase))
                response.OutputStream.Write(bytes, 0, bytes.Length);
        }
        finally
        {
            // Close can block too (it sends the headers of a HEAD or an empty body), so the deadline stays armed
            // across it; an abort that lands during or after Close is harmless.
            try
            {
                if (Volatile.Read(ref state) == ResponseWriting) response.Close();
            }
            finally
            {
                Interlocked.CompareExchange(ref state, ResponseCompleted, ResponseWriting);
            }
        }
    }

    // Snapshot copies taken under the training lock, so the HTTP threads never enumerate a collection the training
    // thread is mutating, and serialization happens outside the lock.
    private List<Dictionary<string, object?>> GetSessionsSnapshot()
    {
        lock (_lock)
        {
            return _sessions.Values
                .OrderBy(s => s.StartTime)
                .Select(SessionSummary)
                .ToList();
        }
    }

    private Dictionary<string, object?>? GetMetricsSnapshot(string sessionId, int? sinceStep, long? afterSequence)
    {
        lock (_lock)
        {
            if (!_sessions.TryGetValue(sessionId, out var session))
                return null;

            var metrics = new Dictionary<string, List<object?[]>>();
            foreach (var kvp in session.MetricHistory)
            {
                session.MetricSequence.TryGetValue(kvp.Key, out var sequence);
                var points = new List<object?[]>();
                var history = kvp.Value;
                for (int i = Math.Max(0, history.Count - MaxPointsPerMetricResponse); i < history.Count; i++)
                {
                    long seq = sequence is not null && i < sequence.Count ? sequence[i] : 0L;
                    if (afterSequence is long after && seq <= after) continue;
                    if (sinceStep is int since && history[i].Step <= since) continue;
                    points.Add(new object?[] { history[i].Step, ToDouble(history[i].Value), ToUnixMilliseconds(history[i].Timestamp) });
                }

                if (points.Count > 0 || afterSequence is null)
                    metrics[kvp.Key] = points;
            }

            var resource = session.LastResourceUsage;
            var speed = GetSpeedStats(sessionId);
            var messages = session.Messages
                .Skip(Math.Max(0, session.Messages.Count - MaxMessagesPerResponse))
                .Select(m => new Dictionary<string, object?>
                {
                    ["time"] = ToUnixMilliseconds(m.Time),
                    ["level"] = m.Level.ToString(),
                    ["message"] = m.Message
                })
                .ToList();
            var alerts = session.AlertMessages
                .Skip(Math.Max(0, session.AlertMessages.Count - MaxAlertsPerResponse))
                .ToList();

            return new Dictionary<string, object?>
            {
                ["session"] = SessionSummary(session),
                ["seq"] = session.LastSequence,
                ["current"] = new Dictionary<string, double>(session.CurrentMetricsDouble),
                ["metrics"] = metrics,
                ["messages"] = messages,
                ["alerts"] = alerts,
                ["issues"] = CheckForIssues(sessionId),
                ["resource"] = resource is null ? null : new Dictionary<string, object?>
                {
                    ["cpuPercent"] = resource.CpuUsagePercent,
                    ["memoryMB"] = resource.MemoryUsageMB,
                    ["gpuPercent"] = resource.GpuUsagePercent,
                    ["gpuMemoryMB"] = resource.GpuMemoryUsageMB,
                    ["time"] = ToUnixMilliseconds(resource.Timestamp)
                },
                ["speed"] = new Dictionary<string, object?>
                {
                    ["iterationsPerSecond"] = speed.IterationsPerSecond,
                    ["estimatedRemainingSeconds"] = speed.EstimatedTimeRemaining.TotalSeconds,
                    ["progressPercent"] = speed.ProgressPercentage
                }
            };
        }
    }

    private Dictionary<string, object?> SessionSummary(SessionData session)
    {
        var end = session.EndTime ?? DateTime.UtcNow;
        return new Dictionary<string, object?>
        {
            ["id"] = session.SessionId,
            ["name"] = session.SessionName,
            ["startTime"] = ToUnixMilliseconds(session.StartTime),
            ["endTime"] = session.EndTime is DateTime e ? ToUnixMilliseconds(e) : null,
            ["elapsedSeconds"] = (end - session.StartTime).TotalSeconds,
            ["active"] = session.EndTime is null,
            ["currentStep"] = session.CurrentStep,
            ["totalSteps"] = session.TotalSteps,
            ["currentEpoch"] = session.CurrentEpoch,
            ["totalEpochs"] = session.TotalEpochs,
            ["metricNames"] = session.MetricHistory.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList(),
            ["metadata"] = session.Metadata?.ToDictionary(
                kvp => kvp.Key,
                kvp => Convert.ToString(kvp.Value, CultureInfo.InvariantCulture))
        };
    }

    private static double? ToDouble(T value) =>
        value is IConvertible convertible ? convertible.ToDouble(CultureInfo.InvariantCulture) : null;

    private static long ToUnixMilliseconds(DateTime time)
    {
        var utc = time.Kind == DateTimeKind.Local ? time.ToUniversalTime() : time;
        return (long)(utc - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalMilliseconds;
    }
}
