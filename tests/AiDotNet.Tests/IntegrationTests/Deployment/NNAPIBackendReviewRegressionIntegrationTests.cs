#nullable disable
using AiDotNet.Deployment.Mobile.Android;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Deployment;

public class NNAPIBackendReviewRegressionIntegrationTests
{
    [Fact]
    public void Execute_WithoutNativeGraphAndWithoutCpuExecutor_Throws()
    {
        using var backend = new NNAPIBackend<float>(new NNAPIConfiguration { AllowCpuFallback = true });
        backend.Initialize();

        var ex = Assert.Throws<InvalidOperationException>(() => backend.Execute([1f, 2f]));
        Assert.Contains("CpuExecutor", ex.Message);
    }

    [Fact]
    public void Execute_UsesConfiguredCpuExecutor_WhenNativeGraphIsUnavailable()
    {
        using var backend = new NNAPIBackend<float>(new NNAPIConfiguration { AllowCpuFallback = true });
        backend.Initialize();
        backend.CpuExecutor = input => [input[0] + 1f, input[1] + 2f];

        var output = backend.Execute([3f, 5f]);

        Assert.Equal([4f, 7f], output);
    }

    [Fact]
    public void Execute_AfterDispose_ThrowsObjectDisposedException()
    {
        var backend = new NNAPIBackend<float>(new NNAPIConfiguration { AllowCpuFallback = true });
        backend.Initialize();
        backend.Dispose();

        Assert.Throws<ObjectDisposedException>(() => backend.Execute([1f]));
    }
}
