using System.Net;
using System.Text;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Playground.Functions.Tests;

public sealed class ExecuteCodeProxyTests
{
    private const string Endpoint = "https://isolated-execution.example/api/execute";

    [Theory]
    [InlineData(200, "{\"success\":true,\"output\":\"42\"}")]
    [InlineData(400, "{\"error\":\"No code provided\"}")]
    [InlineData(503, "{\"error\":\"Unavailable\"}")]
    public async Task ForwardsOnlyBodyAndPreservesBackendResult(int status, string body)
    {
        const string code = "{\"code\":\"System.Console.WriteLine(42);\"}";
        var handler = new Handler(async request =>
        {
            Assert.Equal(Endpoint, request.RequestUri!.AbsoluteUri);
            Assert.Equal(HttpMethod.Post, request.Method);
            Assert.Empty(request.Headers);
            Assert.Equal("application/json", request.Content!.Headers.ContentType!.MediaType);
            Assert.Equal(code, await request.Content.ReadAsStringAsync());
            return new HttpResponseMessage((HttpStatusCode)status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        });
        using var client = new HttpClient(handler);
        var request = Request(code);
        request.Headers.Authorization = "Bearer must-not-forward";
        request.Headers.Cookie = "session=must-not-forward";
        request.Headers["Stripe-Signature"] = "must-not-forward";
        request.Headers["x-functions-key"] = "must-not-forward";
        var result = Assert.IsType<ContentResult>(await Proxy(client, Endpoint).Run(request));
        Assert.Equal(status, result.StatusCode);
        Assert.Equal(body, result.Content);
        Assert.Equal("application/json", result.ContentType);
        Assert.Equal("*", request.HttpContext.Response.Headers.AccessControlAllowOrigin);
        Assert.Equal(1, handler.Calls);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("http://isolated-execution.example/api/execute")]
    [InlineData("https://isolated-execution.example/api/stripe-webhook")]
    [InlineData("https://user:password@isolated-execution.example/api/execute")]
    [InlineData("https://isolated-execution.example/api/execute?code=secret")]
    public async Task MissingOrUnsafeEndpointFailsClosed(string? endpoint)
    {
        var handler = new Handler(_ => throw new Xunit.Sdk.XunitException("Must not send"));
        using var client = new HttpClient(handler);
        var result = Assert.IsType<ContentResult>(await Proxy(client, endpoint).Run(Request("{}")));
        Assert.Equal(503, result.StatusCode);
        Assert.Equal(0, handler.Calls);
    }

    [Fact]
    public async Task BackendFailureDoesNotFallBackToLocalExecution()
    {
        using var client = new HttpClient(new Handler(_ => throw new HttpRequestException("Backend failed")));
        var result = Assert.IsType<ContentResult>(await Proxy(client, Endpoint).Run(Request("{}")));
        Assert.Equal(502, result.StatusCode);
    }

    [Theory]
    [InlineData(200, 502)]
    [InlineData(503, 503)]
    public async Task NonJsonBackendResponseBecomesControlledError(int status, int expected)
    {
        using var client = new HttpClient(new Handler(_ => Task.FromResult(new HttpResponseMessage((HttpStatusCode)status)
        {
            Content = new StringContent("<html>Internal upstream failure</html>", Encoding.UTF8, "text/html")
        })));
        var result = Assert.IsType<ContentResult>(await Proxy(client, Endpoint).Run(Request("{}")));
        Assert.Equal(expected, result.StatusCode);
        Assert.Equal("application/json", result.ContentType);
        using var json = System.Text.Json.JsonDocument.Parse(result.Content!);
        Assert.False(json.RootElement.GetProperty("success").GetBoolean());
        Assert.Equal("Execution service unavailable", json.RootElement.GetProperty("error").GetString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task OversizedResponsesAreRejectedIncludingUnknownContentLength(bool unknownLength)
    {
        using var client = new HttpClient(new Handler(_ =>
        {
            HttpContent content = unknownLength ? new ChunkedContent() : new ByteArrayContent(new byte[ExecuteCodeProxy.MaxResponseBytes + 1]);
            content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        }));
        var result = Assert.IsType<ContentResult>(await Proxy(client, Endpoint).Run(Request("{}")));
        Assert.Equal(502, result.StatusCode);
        Assert.Equal("{\"success\":false,\"error\":\"Execution service unavailable\"}", result.Content);
    }

    [Fact]
    public async Task ResponseAtLimitIsAccepted()
    {
        var body = "\"" + new string('x', ExecuteCodeProxy.MaxResponseBytes - 2) + "\"";
        using var client = new HttpClient(new Handler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new StringContent(body, Encoding.UTF8, "application/json")
        })));
        var result = Assert.IsType<ContentResult>(await Proxy(client, Endpoint).Run(Request("{}")));
        Assert.Equal(200, result.StatusCode);
        Assert.Equal(body, result.Content);
    }

    [Fact]
    public async Task FunctionsActivationUsesNamedClientWithProductionLimits()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddSingleton(Config(Endpoint, "120"));
        ExecuteCodeProxy.RegisterClient(services);
        services.AddHttpClient(ExecuteCodeProxy.ClientName).ConfigurePrimaryHttpMessageHandler(() =>
            new Handler(_ => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(new string('x', ExecuteCodeProxy.MaxResponseBytes + 1), Encoding.UTF8, "application/json")
            })));
        using var provider = services.BuildServiceProvider();
        using var configured = provider.GetRequiredService<IHttpClientFactory>().CreateClient(ExecuteCodeProxy.ClientName);
        Assert.Equal(TimeSpan.FromSeconds(120), configured.Timeout);
        Assert.Equal(ExecuteCodeProxy.MaxResponseBytes, configured.MaxResponseContentBufferSize);
        // This is how the isolated Functions activator creates the function instance.
        var proxy = ActivatorUtilities.CreateInstance<ExecuteCodeProxy>(provider);
        var result = Assert.IsType<ContentResult>(await proxy.Run(Request("{}")));
        Assert.Equal(502, result.StatusCode);
        Assert.Equal("{\"success\":false,\"error\":\"Execution service unavailable\"}", result.Content);
    }

    [Theory]
    [InlineData(null, 60)]
    [InlineData("120", 120)]
    public void ClientUsesValidatedTimeoutAndResponseLimit(string? setting, int seconds)
    {
        using var client = new HttpClient();
        ExecuteCodeProxy.ConfigureClient(client, Config(Endpoint, setting));
        Assert.Equal(TimeSpan.FromSeconds(seconds), client.Timeout);
        Assert.Equal(ExecuteCodeProxy.MaxResponseBytes, client.MaxResponseContentBufferSize);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("181")]
    [InlineData("invalid")]
    public void InvalidTimeoutIsRejected(string setting)
    {
        using var client = new HttpClient();
        Assert.Throws<InvalidOperationException>(() => ExecuteCodeProxy.ConfigureClient(client, Config(Endpoint, setting)));
    }

    private static ExecuteCodeProxy Proxy(HttpClient client, string? endpoint)
    {
        var config = Config(endpoint);
        ExecuteCodeProxy.ConfigureClient(client, config);
        return new(new ClientFactory(client), config, NullLogger<ExecuteCodeProxy>.Instance);
    }

    private sealed class ClientFactory(HttpClient client) : IHttpClientFactory
    {
        public HttpClient CreateClient(string name)
        {
            Assert.Equal(ExecuteCodeProxy.ClientName, name);
            return client;
        }
    }

    private static IConfiguration Config(string? endpoint, string? timeout = null) =>
        new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["PLAYGROUND_EXECUTION_URL"] = endpoint,
            ["PLAYGROUND_EXECUTION_TIMEOUT_SECONDS"] = timeout
        }).Build();

    private sealed class ChunkedContent : HttpContent
    {
        protected override bool TryComputeLength(out long length) { length = 0; return false; }
        protected override async Task SerializeToStreamAsync(Stream stream, TransportContext? context)
        {
            var chunk = new byte[4096];
            for (var bytes = 0; bytes <= ExecuteCodeProxy.MaxResponseBytes; bytes += chunk.Length)
                await stream.WriteAsync(chunk);
        }
    }

    private static HttpRequest Request(string body)
    {
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(body));
        return context.Request;
    }

    private sealed class Handler(Func<HttpRequestMessage, Task<HttpResponseMessage>> send) : HttpMessageHandler
    {
        public int Calls { get; private set; }
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Calls++;
            return send(request);
        }
    }
}
