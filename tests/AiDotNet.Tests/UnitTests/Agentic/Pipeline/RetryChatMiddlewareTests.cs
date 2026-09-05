using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Configuration;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline;

public sealed class RetryChatMiddlewareTests
{
    private static readonly ChatMessage[] Prompt = { ChatMessage.User("hello") };
    private static readonly TimeSpan Instant = TimeSpan.FromMilliseconds(1);

    [Fact]
    public async Task AttemptsAreExactlyMaxRetriesPlusOneAndNeverSquared()
    {
        // OpenEvolve configures retries in its own loop AND passes the same count
        // to the provider SDK, so three retries can become sixteen HTTP attempts.
        var inner = StubChatClient.AlwaysThrows("m", () => Transient());
        IChatClient<double> client = Wrap(inner, maxRetries: 3);

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.GetResponseAsync(Prompt));
        Assert.Equal(4, inner.Calls);
    }

    [Fact]
    public async Task RetriesStopAsSoonAsAnAttemptSucceeds()
    {
        var inner = StubChatClient.ThrowsThenText("m", failures: 2, Transient, "recovered");
        IChatClient<double> client = Wrap(inner, maxRetries: 5);

        ChatResponse response = await client.GetResponseAsync(Prompt);

        Assert.Equal("recovered", response.Text);
        Assert.Equal(3, inner.Calls);
    }

    [Fact]
    public async Task ZeroRetriesMeansExactlyOneAttempt()
    {
        var inner = StubChatClient.AlwaysThrows("m", () => Transient());
        var middleware = new RetryChatMiddleware(maxRetries: 0, baseDelay: Instant, maxDelay: Instant);
        IChatClient<double> client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { middleware });

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.GetResponseAsync(Prompt));
        Assert.Equal(1, inner.Calls);
        Assert.Equal(1, middleware.TotalAttempts);
        Assert.Equal(0, middleware.TotalRetries);
    }

    [Fact]
    public async Task PermanentFailuresAreNotRetriedAndSurfaceUnchanged()
    {
        var inner = StubChatClient.AlwaysThrows(
            "m", () => new HttpResponseException(HttpStatusCode.Unauthorized, "bad key"));
        IChatClient<double> client = Wrap(inner, maxRetries: 4);

        await Assert.ThrowsAsync<HttpResponseException>(() => client.GetResponseAsync(Prompt));
        Assert.Equal(1, inner.Calls);
    }

    [Fact]
    public async Task ANonProviderExceptionIsNotRetriedByDefault()
    {
        var inner = StubChatClient.AlwaysThrows("m", () => new ArgumentException("bad request shape"));
        IChatClient<double> client = Wrap(inner, maxRetries: 4);

        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(Prompt));
        Assert.Equal(1, inner.Calls);
    }

    [Fact]
    public async Task AnExtraPredicateCanWidenWhatCountsAsTransient()
    {
        var inner = StubChatClient.ThrowsThenText("m", 1, () => new ArgumentException("flaky"), "ok");
        var middleware = new RetryChatMiddleware(
            maxRetries: 2, baseDelay: Instant, maxDelay: Instant, shouldRetry: error => error is ArgumentException);
        IChatClient<double> client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { middleware });

        Assert.Equal("ok", (await client.GetResponseAsync(Prompt)).Text);
        Assert.Equal(2, inner.Calls);
    }

    [Theory]
    [InlineData(HttpStatusCode.RequestTimeout, true)]
    // 429 by number rather than by name: HttpStatusCode.TooManyRequests does not exist on the oldest target
    // framework, and this file has to compile on all three for the suite to run there at all.
    [InlineData((HttpStatusCode)429, true)]
    [InlineData(HttpStatusCode.InternalServerError, true)]
    [InlineData(HttpStatusCode.BadGateway, true)]
    [InlineData(HttpStatusCode.BadRequest, false)]
    [InlineData(HttpStatusCode.Unauthorized, false)]
    [InlineData(HttpStatusCode.NotFound, false)]
    public void HttpStatusClassificationFollowsTheTransientCodes(HttpStatusCode status, bool expected)
    {
        var middleware = new RetryChatMiddleware(maxRetries: 1, baseDelay: Instant, maxDelay: Instant);
        Assert.Equal(expected, middleware.IsTransient(new HttpResponseException(status, "message")));
    }

    [Fact]
    public void TransportFailuresAndTimeoutsAreTransient()
    {
        var middleware = new RetryChatMiddleware(maxRetries: 1, baseDelay: Instant, maxDelay: Instant);
        Assert.True(middleware.IsTransient(new HttpRequestException("connection reset")));
        Assert.True(middleware.IsTransient(new TimeoutException()));
        Assert.True(middleware.IsTransient(new TaskCanceledException()));
        Assert.True(middleware.IsTransient(new System.IO.IOException("socket closed")));
    }

    [Fact]
    public async Task AnAttemptThatOverrunsTheCallTimeoutIsTreatedAsATimeout()
    {
        var inner = StubChatClient.Delays("m", TimeSpan.FromSeconds(30), "never");
        var middleware = new RetryChatMiddleware(
            maxRetries: 1,
            baseDelay: Instant,
            maxDelay: Instant,
            callTimeout: TimeSpan.FromMilliseconds(40));
        IChatClient<double> client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { middleware });

        InvalidOperationException error =
            await Assert.ThrowsAsync<InvalidOperationException>(() => client.GetResponseAsync(Prompt));

        Assert.IsType<TimeoutException>(error.InnerException);
        Assert.Equal(2, inner.Calls);
        Assert.Equal(2, middleware.TotalTimeouts);
    }

    [Fact]
    public async Task CallerCancellationIsNotRetriedAndIsNotWrapped()
    {
        var inner = StubChatClient.Delays("m", TimeSpan.FromSeconds(30), "never");
        var middleware = new RetryChatMiddleware(maxRetries: 3, baseDelay: Instant, maxDelay: Instant);
        IChatClient<double> client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { middleware });

        using var source = new CancellationTokenSource();
        source.CancelAfter(TimeSpan.FromMilliseconds(30));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => client.GetResponseAsync(Prompt, null, source.Token));
        Assert.Equal(1, inner.Calls);
    }

    [Fact]
    public async Task DelaysGrowWithEachRetryAndStayUnderTheCeiling()
    {
        var inner = StubChatClient.AlwaysThrows("m", () => Transient());
        var middleware = new RetryChatMiddleware(
            maxRetries: 3,
            baseDelay: TimeSpan.FromMilliseconds(20),
            maxDelay: TimeSpan.FromMilliseconds(50),
            jitterFactor: 0);
        IChatClient<double> client = new MiddlewareChatClient<double>(inner, new IChatMiddleware[] { middleware });

        var watch = System.Diagnostics.Stopwatch.StartNew();
        await Assert.ThrowsAsync<InvalidOperationException>(() => client.GetResponseAsync(Prompt));
        watch.Stop();

        // Waits are 20 ms, 40 ms, and 50 ms (capped): at least 110 ms in total, and
        // three uncapped doublings would have been 20 + 40 + 80.
        Assert.True(watch.ElapsedMilliseconds >= 100, $"Elapsed {watch.ElapsedMilliseconds} ms was too short.");
        Assert.Equal(4, inner.Calls);
        Assert.Equal(3, middleware.TotalRetries);
    }

    [Fact]
    public void JitterIsReproducibleForAGivenSeed()
    {
        var first = new RetryChatMiddleware(maxRetries: 3, baseDelay: Instant, maxDelay: Instant, seed: 77UL);
        var second = new RetryChatMiddleware(maxRetries: 3, baseDelay: Instant, maxDelay: Instant, seed: 77UL);
        Assert.Equal(first.BaseDelay, second.BaseDelay);
        Assert.Equal(first.JitterFactor, second.JitterFactor);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(RetryChatMiddleware.MaxSupportedRetries + 1)]
    public void OutOfRangeRetryCountsAreRejected(int maxRetries)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RetryChatMiddleware(maxRetries));
    }

    [Fact]
    public void AMaximumDelaySmallerThanTheBaseDelayIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RetryChatMiddleware(
            maxRetries: 1, baseDelay: TimeSpan.FromSeconds(5), maxDelay: TimeSpan.FromSeconds(1)));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.1)]
    [InlineData(double.NaN)]
    public void OutOfRangeJitterIsRejected(double jitter)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RetryChatMiddleware(
            maxRetries: 1, baseDelay: Instant, maxDelay: Instant, jitterFactor: jitter));
    }

    [Fact]
    public void ANonPositiveCallTimeoutIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RetryChatMiddleware(
            maxRetries: 1, baseDelay: Instant, maxDelay: Instant, callTimeout: TimeSpan.Zero));
    }

    [Fact]
    public async Task ThePipelineFactoryAppliesExactlyOneRetryLayer()
    {
        var inner = StubChatClient.AlwaysThrows("m", () => Transient());
        var log = new List<string>();
        var options = new ChatClientOptions
        {
            MaxRetries = 2,
            RetryBaseDelay = Instant,
            RetryMaxDelay = Instant,
            Middleware = new List<IChatMiddleware> { new RecordingChatMiddleware(log, "user") }
        };

        IChatClient<double> client = ChatClientPipelineFactory.Create(inner, options);
        await Assert.ThrowsAsync<InvalidOperationException>(() => client.GetResponseAsync(Prompt));

        Assert.Equal(3, inner.Calls);
        Assert.Equal(3, log.FindAll(entry => string.Equals(entry, "user-before", StringComparison.Ordinal)).Count);
    }

    [Fact]
    public async Task RetryAppliesToAnyClientNotJustHttpConnectors()
    {
        // The whole point of moving retry into the pipeline: an in-process client
        // that never touches HTTP still gets the policy.
        var inner = StubChatClient.ThrowsThenText("local-engine", 1, () => new TimeoutException(), "done");
        IChatClient<double> client = ChatClientPipelineFactory.Create(
            inner,
            new ChatClientOptions { MaxRetries = 2, RetryBaseDelay = Instant, RetryMaxDelay = Instant });

        Assert.Equal("done", (await client.GetResponseAsync(Prompt)).Text);
        Assert.Equal(2, inner.Calls);
    }

    [Fact]
    public async Task DefaultChatOptionsFillInBeneathThePerCallSettings()
    {
        var inner = StubChatClient.Text("m", "ok");
        IChatClient<double> client = ChatClientPipelineFactory.Create(
            inner,
            new ChatClientOptions
            {
                MaxRetries = 0,
                DefaultChatOptions = new ChatOptions { Temperature = 0.3, MaxOutputTokens = 512 }
            });

        await client.GetResponseAsync(Prompt, new ChatOptions { MaxOutputTokens = 64 });

        ChatOptions? observed = inner.LastOptions;
        Assert.NotNull(observed);
        Assert.Equal<double?>(0.3, observed?.Temperature);
        Assert.Equal<int?>(64, observed?.MaxOutputTokens);
    }

    [Fact]
    public void APipelineWithNoOptionsReturnsTheOriginalClient()
    {
        var inner = StubChatClient.Text("m", "ok");
        Assert.Same(inner, ChatClientPipelineFactory.Create<double>(inner, null));
    }

    private static IChatClient<double> Wrap(IChatClient<double> inner, int maxRetries) =>
        new MiddlewareChatClient<double>(
            inner,
            new IChatMiddleware[] { new RetryChatMiddleware(maxRetries, Instant, Instant) });

    private static Exception Transient() =>
        new HttpResponseException(HttpStatusCode.ServiceUnavailable, "temporarily unavailable");
}
