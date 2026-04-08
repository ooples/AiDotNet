using System.Reflection;
using System.Text;
using AiDotNet.Serving.Sandboxing.Docker;
using Xunit;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class DockerRunnerReadWithLimitAsyncTests
{
    [Fact]
    public async Task ReadWithLimitAsync_ReturnsEmptyAndTruncated_WhenMaxCharsIsZeroAndStreamHasData()
    {
        var (output, truncated) = await InvokeReadWithLimitAsync("hello", maxChars: 0);

        Assert.Equal(string.Empty, output);
        Assert.True(truncated);
    }

    [Fact]
    public async Task ReadWithLimitAsync_Truncates_WhenOutputExceedsLimit()
    {
        var (output, truncated) = await InvokeReadWithLimitAsync("hello world", maxChars: 5);

        Assert.Equal("hello", output);
        Assert.True(truncated);
    }

    [Fact]
    public async Task ReadWithLimitAsync_DoesNotTruncate_WhenOutputWithinLimit()
    {
        var (output, truncated) = await InvokeReadWithLimitAsync("hi", maxChars: 10);

        Assert.Equal("hi", output);
        Assert.False(truncated);
    }

    private static async Task<(string Output, bool Truncated)> InvokeReadWithLimitAsync(string value, int maxChars)
    {
        var method = typeof(DockerRunner).GetMethod(
            "ReadWithLimitAsync",
            BindingFlags.NonPublic | BindingFlags.Static);

        Assert.NotNull(method);

        await using var stream = new MemoryStream(Encoding.UTF8.GetBytes(value));
        using var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 1024, leaveOpen: true);

        var task = (Task)method!.Invoke(null, new object[] { reader, maxChars, CancellationToken.None })!;
        await task.ConfigureAwait(false);

        var resultProperty = task.GetType().GetProperty("Result")!;
        return ((string Output, bool Truncated))resultProperty.GetValue(task)!;
    }
}

