using System.Collections.Concurrent;
using System.Net;
using System.Text;
using Newtonsoft.Json;

namespace AiDotNet.TrainingMonitoring.Dashboard;

/// <summary>
/// Provides a real-time training dashboard via embedded web server.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> LiveDashboard is like running TensorBoard - it starts
/// a local web server that you can open in your browser to see real-time
/// training progress. The charts update automatically as training progresses.
///
/// Example usage:
/// <code>
/// using var dashboard = new LiveDashboard(port: 6006);
/// dashboard.Start();
/// Console.WriteLine($"Dashboard running at: {dashboard.Url}");
///
/// // During training
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     var (loss, accuracy) = TrainEpoch();
///     dashboard.LogScalar("loss", epoch, loss);
///     dashboard.LogScalar("accuracy", epoch, accuracy);
///     // Dashboard updates in real-time!
/// }
///
/// Console.WriteLine("Press Enter to stop...");
/// Console.ReadLine();
/// dashboard.Stop();
/// </code>
/// </remarks>
public class LiveDashboard : ITrainingDashboard
{
    private readonly HtmlDashboard _htmlDashboard;
    private readonly HttpListener _listener;
    private readonly CancellationTokenSource _cts;
    private Task? _listenerTask;
    private bool _disposed;

    /// <inheritdoc />
    public string Name => _htmlDashboard.Name;

    /// <inheritdoc />
    public string LogDirectory => _htmlDashboard.LogDirectory;

    /// <inheritdoc />
    public bool IsRunning { get; private set; }

    /// <summary>
    /// Gets the port the server is listening on.
    /// </summary>
    public int Port { get; }

    /// <summary>
    /// Gets the URL to access the dashboard.
    /// </summary>
    public string Url => $"http://localhost:{Port}/";

    /// <summary>
    /// Gets or sets the refresh interval in milliseconds for auto-updating clients.
    /// </summary>
    public int RefreshIntervalMs { get; set; } = 2000;

    /// <summary>
    /// Gets or sets whether to allow external connections.
    /// </summary>
    /// <remarks>
    /// When false (default), only connections from localhost are accepted.
    /// Set to true to allow connections from other machines on the network.
    /// Note: Binding to all interfaces may require administrator privileges.
    /// </remarks>
    public bool AllowExternalConnections { get; set; }

    /// <summary>
    /// Creates a new live dashboard.
    /// </summary>
    /// <param name="logDirectory">Directory to save logs.</param>
    /// <param name="name">Name of this training run.</param>
    /// <param name="port">Port to listen on (default: 6006, same as TensorBoard).</param>
    /// <param name="allowExternalConnections">Whether to allow connections from other machines (default: false for security).</param>
    /// <remarks>
    /// <b>Security Note:</b> This dashboard uses HTTP (not HTTPS) for simplicity in local development.
    /// By default, it only accepts connections from localhost. If you enable external connections,
    /// be aware that data is transmitted unencrypted. For production use or sensitive data,
    /// consider running behind a reverse proxy with TLS.
    /// </remarks>
    public LiveDashboard(string? logDirectory = null, string? name = null, int port = 6006, bool allowExternalConnections = false)
    {
        Port = port;
        AllowExternalConnections = allowExternalConnections;
        _htmlDashboard = new HtmlDashboard(logDirectory ?? "./logs", name);
        _cts = new CancellationTokenSource();
        _listener = new HttpListener();
        // Default to localhost-only for security; use + for all interfaces only when explicitly requested
        var prefix = allowExternalConnections ? $"http://+:{port}/" : $"http://localhost:{port}/";
        _listener.Prefixes.Add(prefix);
    }

    /// <inheritdoc />
    public void Start()
    {
        if (IsRunning) return;

        try
        {
            _listener.Start();
        }
        catch (HttpListenerException)
        {
            // Fall back to localhost-only if external binding fails (requires admin on Windows)
            if (AllowExternalConnections)
            {
                _listener.Prefixes.Clear();
                _listener.Prefixes.Add($"http://localhost:{Port}/");
                _listener.Start();
                Console.WriteLine($"[LiveDashboard] Note: External connections unavailable, bound to localhost only");
            }
            else
            {
                throw; // Re-throw if localhost-only binding fails
            }
        }

        IsRunning = true;
        _listenerTask = Task.Run(() => ListenAsync(_cts.Token));

        Console.WriteLine($"[LiveDashboard] Running at {Url}");
    }

    /// <inheritdoc />
    public void Stop()
    {
        if (!IsRunning) return;

        IsRunning = false;
        _cts.Cancel();
        _listener.Stop();

        try
        {
            _listenerTask?.Wait(TimeSpan.FromSeconds(5));
        }
        catch (AggregateException)
        {
            // Expected when cancelling
        }

        Console.WriteLine("[LiveDashboard] Stopped");
    }

    private async Task ListenAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested && _listener.IsListening)
        {
            try
            {
                var context = await _listener.GetContextAsync();
                _ = Task.Run(() => HandleRequest(context), cancellationToken);
            }
            catch (HttpListenerException)
            {
                // Listener was stopped
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
        }
    }

    private void HandleRequest(HttpListenerContext context)
    {
        try
        {
            var path = context.Request.Url?.AbsolutePath ?? "/";
            var response = context.Response;

            switch (path)
            {
                case "/":
                case "/index.html":
                    ServeHtml(response);
                    break;
                case "/api/scalars":
                    ServeJson(response, _htmlDashboard.GetScalarData());
                    break;
                case "/api/histograms":
                    ServeJson(response, _htmlDashboard.GetHistogramData());
                    break;
                case "/api/summary":
                    ServeJson(response, GetSummary());
                    break;
                default:
                    response.StatusCode = 404;
                    ServeText(response, "Not Found");
                    break;
            }
        }
        catch (Exception ex)
        {
            try
            {
                context.Response.StatusCode = 500;
                // Don't expose exception details to HTTP clients for security
                // Log the actual error for debugging
                System.Diagnostics.Debug.WriteLine($"[LiveDashboard] Error processing request: {ex}");
                ServeText(context.Response, "Internal Server Error");
            }
            catch
            {
                // Ignore errors when sending error response
            }
        }
        finally
        {
            try
            {
                context.Response.Close();
            }
            catch
            {
                // Ignore close errors
            }
        }
    }

    private object GetSummary()
    {
        var scalars = _htmlDashboard.GetScalarData();
        var summary = new Dictionary<string, object>
        {
            ["name"] = Name,
            ["metricsCount"] = scalars.Count,
            ["dataPoints"] = scalars.Values.Sum(s => s.Count),
            ["latestValues"] = scalars.ToDictionary(
                kvp => kvp.Key,
                kvp => kvp.Value.LastOrDefault()?.Value ?? 0
            )
        };

        return summary;
    }

    private void ServeHtml(HttpListenerResponse response)
    {
        var html = GenerateLiveHtml();
        var buffer = Encoding.UTF8.GetBytes(html);

        response.ContentType = "text/html; charset=utf-8";
        response.ContentLength64 = buffer.Length;
        response.OutputStream.Write(buffer, 0, buffer.Length);
    }

    private void ServeJson(HttpListenerResponse response, object data)
    {
        var json = JsonConvert.SerializeObject(data);
        var buffer = Encoding.UTF8.GetBytes(json);

        response.ContentType = "application/json; charset=utf-8";
        response.ContentLength64 = buffer.Length;
        response.OutputStream.Write(buffer, 0, buffer.Length);
    }

    private void ServeText(HttpListenerResponse response, string text)
    {
        var buffer = Encoding.UTF8.GetBytes(text);

        response.ContentType = "text/plain; charset=utf-8";
        response.ContentLength64 = buffer.Length;
        response.OutputStream.Write(buffer, 0, buffer.Length);
    }

    private string GenerateLiveHtml()
    {
        return $@"<!DOCTYPE html>
<html lang=""en"">
<head>
    <meta charset=""UTF-8"">
    <meta name=""viewport"" content=""width=device-width, initial-scale=1.0"">
    <title>Live Training Dashboard - {System.Net.WebUtility.HtmlEncode(Name)}</title>
    <script src=""https://cdn.jsdelivr.net/npm/chart.js""></script>
    <style>
        :root {{
            --primary: #3b82f6;
            --bg: #0f172a;
            --bg-card: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
            --success: #22c55e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }}
        header {{
            background: linear-gradient(135deg, #2563eb, var(--primary));
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        header h1 {{ font-size: 1.5rem; }}
        .status {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success);
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .summary-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .summary-card .label {{ color: var(--text-muted); font-size: 0.875rem; }}
        .summary-card .value {{ font-size: 2rem; font-weight: bold; color: var(--primary); }}
        #charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1rem;
        }}
        .chart-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .chart-card h3 {{
            color: var(--text-muted);
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }}
        .chart-container {{
            position: relative;
            height: 250px;
        }}
        .no-data {{
            text-align: center;
            color: var(--text-muted);
            padding: 3rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Live Training: {System.Net.WebUtility.HtmlEncode(Name)}</h1>
        <div class=""status"">
            <div class=""status-dot""></div>
            <span>Live</span>
        </div>
    </header>
    <main>
        <div class=""summary"" id=""summary"">
            <div class=""summary-card"">
                <div class=""label"">Metrics</div>
                <div class=""value"" id=""metrics-count"">0</div>
            </div>
            <div class=""summary-card"">
                <div class=""label"">Data Points</div>
                <div class=""value"" id=""data-points"">0</div>
            </div>
            <div class=""summary-card"">
                <div class=""label"">Last Update</div>
                <div class=""value"" id=""last-update"" style=""font-size: 1rem;"">-</div>
            </div>
        </div>
        <div id=""charts-container"">
            <div class=""no-data"" id=""no-data"">
                <h2>Waiting for data...</h2>
                <p>Log some metrics to see charts appear here.</p>
            </div>
        </div>
    </main>
    <script>
        const charts = {{}};
        const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6'];
        let colorIndex = 0;

        function getColor() {{
            const color = colors[colorIndex % colors.length];
            colorIndex++;
            return color;
        }}

        async function fetchData() {{
            try {{
                const [scalarsRes, summaryRes] = await Promise.all([
                    fetch('/api/scalars'),
                    fetch('/api/summary')
                ]);
                const scalars = await scalarsRes.json();
                const summary = await summaryRes.json();

                document.getElementById('metrics-count').textContent = summary.metricsCount;
                document.getElementById('data-points').textContent = summary.dataPoints.toLocaleString();
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();

                updateCharts(scalars);
            }} catch (err) {{
                console.error('Failed to fetch data:', err);
            }}
        }}

        function updateCharts(scalars) {{
            const container = document.getElementById('charts-container');
            const noData = document.getElementById('no-data');

            if (Object.keys(scalars).length === 0) {{
                noData.style.display = 'block';
                return;
            }}

            noData.style.display = 'none';

            for (const [name, data] of Object.entries(scalars)) {{
                if (!charts[name]) {{
                    // Create new chart
                    const chartCard = document.createElement('div');
                    chartCard.className = 'chart-card';
                    chartCard.innerHTML = `
                        <h3>${{name}}</h3>
                        <div class=""chart-container""><canvas id=""chart-${{name.replace(/[^a-z0-9]/gi, '_')}}""></canvas></div>
                    `;
                    container.appendChild(chartCard);

                    const ctx = chartCard.querySelector('canvas').getContext('2d');
                    const color = getColor();
                    charts[name] = new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: [],
                            datasets: [{{
                                label: name,
                                data: [],
                                borderColor: color,
                                backgroundColor: color + '20',
                                borderWidth: 2,
                                fill: true,
                                tension: 0.1,
                                pointRadius: 0
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            animation: {{ duration: 0 }},
                            scales: {{
                                x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }},
                                y: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: '#334155' }} }}
                            }},
                            plugins: {{ legend: {{ display: false }} }}
                        }}
                    }});
                }}

                // Update data
                const chart = charts[name];
                chart.data.labels = data.map(d => d.Step);
                chart.data.datasets[0].data = data.map(d => d.Value);
                chart.update('none');
            }}
        }}

        // Initial fetch
        fetchData();

        // Refresh periodically
        setInterval(fetchData, {RefreshIntervalMs});
    </script>
</body>
</html>";
    }

    #region ITrainingDashboard Implementation (delegates to HtmlDashboard)

    /// <inheritdoc />
    public void LogScalar(string name, long step, double value, DateTime? wallTime = null)
        => _htmlDashboard.LogScalar(name, step, value, wallTime);

    /// <inheritdoc />
    public void LogScalars(Dictionary<string, double> scalars, long step, DateTime? wallTime = null)
        => _htmlDashboard.LogScalars(scalars, step, wallTime);

    /// <inheritdoc />
    public void LogHistogram(string name, long step, double[] values, DateTime? wallTime = null)
        => _htmlDashboard.LogHistogram(name, step, values, wallTime);

    /// <inheritdoc />
    public void LogImage(string name, long step, byte[] imageData, int width, int height, DateTime? wallTime = null)
        => _htmlDashboard.LogImage(name, step, imageData, width, height, wallTime);

    /// <inheritdoc />
    public void LogText(string name, long step, string text, DateTime? wallTime = null)
        => _htmlDashboard.LogText(name, step, text, wallTime);

    /// <inheritdoc />
    public void LogHyperparameters(Dictionary<string, object> hyperparams, Dictionary<string, double>? metrics = null)
        => _htmlDashboard.LogHyperparameters(hyperparams, metrics);

    /// <inheritdoc />
    public void LogConfusionMatrix(string name, long step, int[,] matrix, string[] labels, DateTime? wallTime = null)
        => _htmlDashboard.LogConfusionMatrix(name, step, matrix, labels, wallTime);

    /// <inheritdoc />
    public void LogPRCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null)
        => _htmlDashboard.LogPRCurve(name, step, predictions, labels, wallTime);

    /// <inheritdoc />
    public void LogROCCurve(string name, long step, double[] predictions, int[] labels, DateTime? wallTime = null)
        => _htmlDashboard.LogROCCurve(name, step, predictions, labels, wallTime);

    /// <inheritdoc />
    public void LogModelGraph(string modelDescription)
        => _htmlDashboard.LogModelGraph(modelDescription);

    /// <inheritdoc />
    public string GenerateReport(string? outputPath = null)
        => _htmlDashboard.GenerateReport(outputPath);

    /// <inheritdoc />
    public void ExportTensorBoardFormat(string outputDirectory)
        => _htmlDashboard.ExportTensorBoardFormat(outputDirectory);

    /// <inheritdoc />
    public Dictionary<string, List<ScalarDataPoint>> GetScalarData()
        => _htmlDashboard.GetScalarData();

    /// <inheritdoc />
    public Dictionary<string, List<HistogramDataPoint>> GetHistogramData()
        => _htmlDashboard.GetHistogramData();

    /// <inheritdoc />
    public void Clear()
        => _htmlDashboard.Clear();

    /// <inheritdoc />
    public void Flush()
        => _htmlDashboard.Flush();

    #endregion

    /// <summary>
    /// Disposes the live dashboard.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Stop();
        _cts.Dispose();
        _listener.Close();
        _htmlDashboard.Dispose();
    }
}
