using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Dashboard.Dashboard;
using AiDotNet.Interfaces;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Dashboard;

/// <summary>
/// The opt-in local web page of <see cref="TrainingMonitorDashboard{T}"/>: it binds a free port, serves the page and
/// the JSON endpoints with the logged values, reads safely while training logs, and frees its port when stopped.
/// </summary>
public class TrainingMonitorDashboardWebServerTests
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(20) };

    private static async Task<JToken> GetJson(string url)
    {
        using var response = await Http.GetAsync(url);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.StartsWith("application/json", response.Content.Headers.ContentType?.MediaType ?? string.Empty);
        return JToken.Parse(await response.Content.ReadAsStringAsync());
    }

    [Fact(Timeout = 60000)]
    public async Task StartWebServer_OnPortZero_ServesPageSessionsAndMetrics()
    {
        using var dashboard = new TrainingMonitorDashboard<double>("web-test");
        var session = dashboard.StartSession("run-a", new Dictionary<string, object> { ["model"] = "tiny" });
        dashboard.LogMetric(session, "train_loss", 2.5, 10);
        dashboard.LogMetric(session, "train_loss", 2.0, 20);
        dashboard.LogMetric(session, "learning_rate", 0.001, 20);
        dashboard.LogMetric(session, "diverged", double.NaN, 20);
        dashboard.UpdateProgress(session, 20, 100, 1, 1);
        dashboard.LogMessage(session, LogLevel.Warning, "watch this");
        dashboard.LogResourceUsage(session, 12.5, 512.0);

        var url = dashboard.StartWebServer();
        Assert.Equal(url, dashboard.WebServerUrl);
        var uri = new Uri(url);
        Assert.True(uri.IsLoopback);
        Assert.True(uri.Port > 0);
        Assert.Equal(url, dashboard.StartWebServer()); // idempotent while running

        using (var page = await Http.GetAsync(url))
        {
            Assert.Equal(HttpStatusCode.OK, page.StatusCode);
            Assert.Equal("text/html", page.Content.Headers.ContentType?.MediaType);
            var html = await page.Content.ReadAsStringAsync();
            Assert.Contains("<script>", html);           // inline script
            Assert.Contains("getContext(\"2d\")", html); // canvas charts
            Assert.Contains("api/metrics", html);
            Assert.DoesNotContain("http://", html);       // no external assets
            Assert.DoesNotContain("https://", html);
            Assert.DoesNotContain("<script src", html);
        }

        var sessions = (JArray)await GetJson(url + "api/sessions");
        var s = Assert.Single(sessions);
        Assert.Equal(session, (string?)s["id"]);
        Assert.Equal("run-a", (string?)s["name"]);
        Assert.Equal(20, (int)s["currentStep"]!);
        Assert.Equal(100, (int)s["totalSteps"]!);
        Assert.True((bool)s["active"]!);
        Assert.Equal("tiny", (string?)s["metadata"]!["model"]);

        var all = await GetJson(url + "api/metrics?session=" + session);
        var loss = (JArray)all["metrics"]!["train_loss"]!;
        Assert.Equal(new[] { 10, 20 }, loss.Select(p => (int)p[0]!).ToArray());
        Assert.Equal(new[] { 2.5, 2.0 }, loss.Select(p => (double)p[1]!).ToArray());
        Assert.Equal(0.001, (double)all["current"]!["learning_rate"]!, 12);
        Assert.Equal("NaN", (string?)all["current"]!["diverged"]);
        Assert.Equal(512.0, (double)all["resource"]!["memoryMB"]!);
        Assert.Contains(all["messages"]!, m => (string?)m["message"] == "watch this" && (string?)m["level"] == "Warning");

        // Incremental by step: only points after step 10.
        var since = await GetJson(url + "api/metrics?session=" + session + "&since=10");
        Assert.Equal(new[] { 20 }, ((JArray)since["metrics"]!["train_loss"]!).Select(p => (int)p[0]!).ToArray());

        // Incremental by sequence: nothing new, then exactly the new point.
        long seq = (long)all["seq"]!;
        var none = await GetJson(url + "api/metrics?session=" + session + "&seq=" + seq);
        Assert.Empty((JObject)none["metrics"]!);
        dashboard.LogMetric(session, "train_loss", 1.5, 30);
        var next = await GetJson(url + "api/metrics?session=" + session + "&seq=" + seq);
        Assert.Equal(new[] { 30 }, ((JArray)next["metrics"]!["train_loss"]!).Select(p => (int)p[0]!).ToArray());

        using (var missing = await Http.GetAsync(url + "api/metrics?session=nope"))
            Assert.Equal(HttpStatusCode.NotFound, missing.StatusCode);
        using (var bad = await Http.GetAsync(url + "api/metrics"))
            Assert.Equal(HttpStatusCode.BadRequest, bad.StatusCode);
        using (var unknown = await Http.GetAsync(url + "nothing-here"))
            Assert.Equal(HttpStatusCode.NotFound, unknown.StatusCode);
    }

    [Fact(Timeout = 60000)]
    public async Task StopAndDispose_FreeThePort()
    {
        int port;
        using (var dashboard = new TrainingMonitorDashboard<double>())
        {
            var url = dashboard.StartWebServer();
            port = new Uri(url).Port;
            using (var ok = await Http.GetAsync(url + "api/sessions"))
                Assert.Equal(HttpStatusCode.OK, ok.StatusCode);

            dashboard.StopWebServer();
            Assert.Null(dashboard.WebServerUrl);
            await AssertNotServing(url);

            // The same port can be bound again, by the same dashboard.
            var again = dashboard.StartWebServer(port);
            Assert.Equal(port, new Uri(again).Port);
            using (var ok = await Http.GetAsync(again + "api/sessions"))
                Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
        }

        // Dispose stopped it: a second dashboard can take the same port (retried briefly: a platform may hold a
        // just-closed listening socket for a moment).
        using var other = new TrainingMonitorDashboard<double>();
        string? reused = null;
        for (int attempt = 0; reused is null; attempt++)
        {
            try { reused = other.StartWebServer(port); }
            catch (InvalidOperationException) when (attempt < 20) { await Task.Delay(250); }
        }
        Assert.Equal(port, new Uri(reused).Port);
        other.StopWebServer();
        other.StopWebServer(); // stopping twice is a no-op
    }

    // A loopback server answers only requests addressed to a loopback name. A page on another origin can point its own
    // host name at 127.0.0.1 (DNS rebinding) and read the metrics; its requests carry its own name as Host.
    [Fact(Timeout = 60000)]
    public async Task A_loopback_server_refuses_a_request_addressed_to_another_host_name()
    {
        using var dashboard = new TrainingMonitorDashboard<double>();
        var url = dashboard.StartWebServer();
        int port = new Uri(url).Port;
        using (var rebound = new HttpRequestMessage(HttpMethod.Get, url + "api/sessions"))
        {
            rebound.Headers.Host = "attacker.example:" + port.ToString(System.Globalization.CultureInfo.InvariantCulture);
            using var response = await Http.SendAsync(rebound);
            // Refused by the server (403) or, on Windows, by http.sys before it (400 invalid host name).
            Assert.Contains(response.StatusCode, new[] { HttpStatusCode.Forbidden, HttpStatusCode.BadRequest });
        }
        using (var local = await Http.GetAsync(url + "api/sessions"))
            Assert.Equal(HttpStatusCode.OK, local.StatusCode);

    }

    // The check holds whatever the server is bound to: a wildcard bind serves loopback too, where http.sys does not
    // filter the Host name, so the dashboard's own check is what refuses a rebound name.
    [Theory]
    [InlineData("127.0.0.1:8080", "+", true)]
    [InlineData("localhost:8080", "+", true)]
    [InlineData("[::1]:8080", "*", true)]
    [InlineData("10.1.2.3:8080", "0.0.0.0", true)]
    [InlineData("dash.example:8080", "dash.example", true)]
    [InlineData("attacker.example:8080", "+", false)]
    [InlineData("attacker.example:8080", "127.0.0.1", false)]
    [InlineData("attacker.example", "dash.example", false)]
    [InlineData("LOCALHOST:8080", "+", true)]
    [InlineData("localhost.:8080", "+", true)]
    [InlineData("[fe80::1%4]:8080", "+", true)]
    [InlineData("127.0.0.1.attacker.example:8080", "+", false)]
    [InlineData("localhost.attacker.example", "127.0.0.1", false)]
    [InlineData("user@attacker.example:8080", "+", false)]
    [InlineData("attacker.example.:8080", "+", false)]
    [InlineData(".", "+", false)]
    [InlineData("", "+", false)]
    [InlineData(null, "127.0.0.1", false)]
    public void Host_check_accepts_only_names_that_cannot_be_rebound(string? hostHeader, string boundHost, bool allowed)
    {
        var method = typeof(TrainingMonitorDashboard<double>).GetMethod("IsAllowedHost",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        Assert.NotNull(method);
        Assert.Equal(allowed, (bool)method!.Invoke(null, new object?[] { hostHeader, boundHost })!);
        // This machine's own name is not an attacker's, but a name that only starts with it is.
        string machine = System.Net.Dns.GetHostName();
        Assert.True((bool)method.Invoke(null, new object?[] { machine + ":8080", "+" })!);
        Assert.False((bool)method.Invoke(null, new object?[] { machine + ".attacker.example:8080", "+" })!);
    }

    private static async Task AssertNotServing(string url)
    {
        try
        {
            using var response = await Http.GetAsync(url + "api/sessions");
            // http.sys may answer for a prefix nobody owns any more, but never with the dashboard's payload.
            Assert.NotEqual(HttpStatusCode.OK, response.StatusCode);
        }
        catch (HttpRequestException)
        {
            // Connection refused: nothing listens.
        }
    }

    [Fact(Timeout = 120000)]
    public async Task ConcurrentLoggingWhileServing_DoesNotThrow()
    {
        using var dashboard = new TrainingMonitorDashboard<float>();
        var url = dashboard.StartWebServer();
        var session = dashboard.StartSession("concurrent");
        var errors = new List<Exception>();
        using var stop = new CancellationTokenSource();

        var writer = Task.Run(() =>
        {
            try
            {
                for (int step = 1; step <= 20000; step++)
                {
                    dashboard.LogMetric(session, "train_loss", 1.0f / step, step);
                    dashboard.LogMetric(session, "metric_" + (step % 7), step, step);
                    if (step % 50 == 0)
                    {
                        dashboard.UpdateProgress(session, step, 20000, 1, 1);
                        dashboard.LogMessage(session, LogLevel.Info, "step " + step);
                        dashboard.LogResourceUsage(session, 10, 100);
                    }
                    if (step % 5000 == 0)
                        dashboard.StartSession("extra-" + step); // the session list changes too
                }
            }
            catch (Exception ex)
            {
                lock (errors) errors.Add(ex);
            }
        });

        var readers = Enumerable.Range(0, 3).Select(_ => Task.Run(async () =>
        {
            long seq = -1;
            while (!writer.IsCompleted)
            {
                try
                {
                    await GetJson(url + "api/sessions");
                    var data = await GetJson(url + "api/metrics?session=" + session + (seq >= 0 ? "&seq=" + seq : string.Empty));
                    seq = (long)data["seq"]!;
                }
                catch (Exception ex)
                {
                    lock (errors) errors.Add(ex);
                    return;
                }
            }
        })).ToArray();

        await writer;
        await Task.WhenAll(readers);
        Assert.Empty(errors);

        var final = await GetJson(url + "api/metrics?session=" + session);
        // A full read serves each metric's newest 4096 points (a long run's first poll stays bounded).
        var trainLoss = (JArray)final["metrics"]!["train_loss"]!;
        Assert.Equal(4096, trainLoss.Count);
        Assert.Equal(20000, (int)trainLoss[trainLoss.Count - 1]![0]!);
        Assert.Equal(20000 - 4096 + 1, (int)trainLoss[0]![0]!);
    }

    [Fact(Timeout = 60000)]
    public async Task CreateWebTrainingMonitor_StartsTheServer_AndDisposeStopsIt()
    {
        string url;
        using (var monitor = MetricsDashboard.CreateWebTrainingMonitor<double>("parity"))
        {
            url = monitor.WebServerUrl ?? throw new InvalidOperationException("server not started");
            using var response = await Http.GetAsync(url);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        }

        await AssertNotServing(url);
    }

    [Fact]
    public void StartWebServerFromEnvironment_WithoutTheVariable_ReturnsNull()
    {
        var previous = Environment.GetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable, null);
            Assert.Null(TrainingMonitorDashboard<double>.StartWebServerFromEnvironment());
            using var dashboard = new TrainingMonitorDashboard<double>();
            Assert.Null(dashboard.TryStartWebServerFromEnvironment());
        }
        finally
        {
            Environment.SetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable, previous);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task StartWebServerFromEnvironment_WithPortZero_Serves()
    {
        var previous = Environment.GetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable, "0");
            using var dashboard = TrainingMonitorDashboard<double>.StartWebServerFromEnvironment();
            Assert.NotNull(dashboard);
            using var response = await Http.GetAsync(dashboard!.WebServerUrl + "api/sessions");
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);

            Environment.SetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable, "not-a-port");
            using var other = new TrainingMonitorDashboard<double>();
            Assert.Throws<FormatException>(() => other.TryStartWebServerFromEnvironment());
        }
        finally
        {
            Environment.SetEnvironmentVariable(TrainingMonitorDashboard<double>.PortEnvironmentVariable, previous);
        }
    }
}
