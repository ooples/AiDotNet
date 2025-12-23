using AiDotNet.Serving.Sandboxing.Docker;
using Xunit;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class DockerRunnerValidationTests
{
    [Fact]
    public async Task RunAsync_ValidatesArguments()
    {
        var runner = new DockerRunner();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            runner.RunAsync(null!, stdIn: null, timeout: TimeSpan.FromSeconds(1), maxStdOutChars: 1, maxStdErrChars: 1, cancellationToken: default));

        await Assert.ThrowsAsync<ArgumentException>(() =>
            runner.RunAsync("   ", stdIn: null, timeout: TimeSpan.FromSeconds(1), maxStdOutChars: 1, maxStdErrChars: 1, cancellationToken: default));

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            runner.RunAsync("version", stdIn: null, timeout: TimeSpan.Zero, maxStdOutChars: 1, maxStdErrChars: 1, cancellationToken: default));

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            runner.RunAsync("version", stdIn: null, timeout: TimeSpan.FromSeconds(1), maxStdOutChars: -1, maxStdErrChars: 1, cancellationToken: default));

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            runner.RunAsync("version", stdIn: null, timeout: TimeSpan.FromSeconds(1), maxStdOutChars: 1, maxStdErrChars: -1, cancellationToken: default));
    }
}

