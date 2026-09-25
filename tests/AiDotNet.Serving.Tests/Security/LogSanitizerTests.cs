using System.Collections.Concurrent;
using System.Net;
using AiDotNet.Serving.Security;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using Xunit;

namespace AiDotNet.Serving.Tests.Security;

public class LogSanitizerTests
{
    [Theory]
    [InlineData("model\r\n[CRIT] forged", @"model\r\n[CRIT] forged")]
    [InlineData("a\nb", @"a\nb")]
    [InlineData("a\rb", @"a\rb")]
    [InlineData("a\tb", @"a\tb")]
    [InlineData("a\u0085b", @"a\u0085b")] // NEL
    [InlineData("a\u2028b\u2029c", @"a\u2028b\u2029c")] // line / paragraph separator
    [InlineData("a\u001B[31mb", @"a\u001B[31mb")] // ANSI escape sequence
    [InlineData("a\0b", @"a\u0000b")]
    [InlineData("admin\u202Egnp.exe", @"admin\u202Egnp.exe")] // right-to-left override
    public void Sanitize_EscapesLineBreaksAndControlCharacters(string raw, string expected)
    {
        var sanitized = LogSanitizer.Sanitize(raw);

        Assert.Equal(expected, sanitized);
        Assert.DoesNotContain('\r', sanitized!);
        Assert.DoesNotContain('\n', sanitized!);
    }

    [Theory]
    [InlineData("resnet-50")]
    [InlineData("models/sub dir/model_v2.aidn")]
    [InlineData("modèle-日本語")]
    [InlineData("")]
    public void Sanitize_LeavesPrintableTextUnchanged(string raw)
    {
        var sanitized = LogSanitizer.Sanitize(raw);

        Assert.Same(raw, sanitized);
    }

    [Fact]
    public void Sanitize_PassesNullThrough()
    {
        Assert.Null(LogSanitizer.Sanitize(null));
    }

    [Fact(Timeout = 60000)]
    public async Task ModelNotFoundLog_WithCrLfInRouteValue_StaysOnOneLine()
    {
        var sink = new CapturingLoggerProvider();
        using var factory = new ServingTestWebApplicationFactory()
            .WithWebHostBuilder(builder => builder.ConfigureLogging(logging => logging.AddProvider(sink)));
        using var client = factory.CreateClient();

        const string forged = "ghost\r\n[CRIT] admin password reset for root";
        var response = await client.GetAsync("/api/models/" + Uri.EscapeDataString(forged));

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);

        var notFound = sink.Messages.Where(m => m.Contains("not found", StringComparison.Ordinal)
                                                && m.Contains("ghost", StringComparison.Ordinal)).ToList();
        Assert.NotEmpty(notFound);
        foreach (var message in notFound)
        {
            Assert.DoesNotContain('\r', message);
            Assert.DoesNotContain('\n', message);
            Assert.Contains(@"ghost\r\n[CRIT] admin password reset for root", message, StringComparison.Ordinal);
        }
    }

    private sealed class CapturingLoggerProvider : ILoggerProvider
    {
        private readonly ConcurrentQueue<string> _messages = new();

        public IReadOnlyCollection<string> Messages => _messages.ToArray();

        public ILogger CreateLogger(string categoryName) => new CapturingLogger(_messages);

        public void Dispose()
        {
        }

        private sealed class CapturingLogger(ConcurrentQueue<string> messages) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter) => messages.Enqueue(formatter(state, exception));
        }
    }
}
