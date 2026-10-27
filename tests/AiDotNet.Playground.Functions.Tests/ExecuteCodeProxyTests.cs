using System.Net;
using System.Text;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
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
            return new HttpResponseMessage((HttpStatusCode)status) { Content = new StringContent(body) };
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
        var result = Assert.IsType<StatusCodeResult>(await Proxy(client, endpoint).Run(Request("{}")));
        Assert.Equal(503, result.StatusCode);
        Assert.Equal(0, handler.Calls);
    }

    [Fact]
    public async Task BackendFailureDoesNotFallBackToLocalExecution()
    {
        using var client = new HttpClient(new Handler(_ => throw new HttpRequestException("Backend failed")));
        var result = Assert.IsType<StatusCodeResult>(await Proxy(client, Endpoint).Run(Request("{}")));
        Assert.Equal(502, result.StatusCode);
    }

    private static ExecuteCodeProxy Proxy(HttpClient client, string? endpoint) => new(client,
        new ConfigurationBuilder().AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["PLAYGROUND_EXECUTION_URL"] = endpoint
        }).Build(), NullLogger<ExecuteCodeProxy>.Instance);

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
