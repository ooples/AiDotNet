using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class PlatformDetectorTests
{
    [Fact]
    public void Capabilities_IsPopulated_AndDoesNotThrow()
    {
        var caps = PlatformDetector.Capabilities;

        Assert.True(caps.ProcessorCount > 0);
        Assert.NotNull(caps.OSDescription);
        Assert.NotNull(caps.FrameworkDescription);
        Assert.True(caps.L1CacheSize > 0);
        Assert.True(caps.L2CacheSize > 0);
        Assert.True(caps.L3CacheSize > 0);
    }
}

