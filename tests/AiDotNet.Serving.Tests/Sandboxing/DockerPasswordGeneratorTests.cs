using AiDotNet.Serving.Sandboxing;
using Xunit;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class DockerPasswordGeneratorTests
{
    [Fact]
    public void Generate_Default_ReturnsBase64UrlString()
    {
        var password = DockerPasswordGenerator.Generate();

        Assert.False(string.IsNullOrWhiteSpace(password));
        Assert.DoesNotContain("=", password);
        Assert.DoesNotContain("+", password);
        Assert.DoesNotContain("/", password);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Generate_InvalidLength_Throws(int bytes)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DockerPasswordGenerator.Generate(bytes));
    }
}

