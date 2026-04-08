using AiDotNet.Serving.Sandboxing;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Serving.Tests.Sandboxing;

public sealed class DockerPasswordGeneratorTests
{
    [Fact(Timeout = 60000)]
    public async Task Generate_Default_ReturnsBase64UrlString()
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

