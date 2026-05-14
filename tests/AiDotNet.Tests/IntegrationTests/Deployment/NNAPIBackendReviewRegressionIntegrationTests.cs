#nullable disable
using AiDotNet.Deployment.Mobile.Android;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Deployment;

public class NNAPIBackendReviewRegressionIntegrationTests
{
    [Fact(Timeout = 120000)]
    public async Task Execute_WithoutNativeGraphAndWithoutCpuExecutor_Throws()
    {
        await Task.Yield();
        using var backend = new NNAPIBackend<float>(new NNAPIConfiguration { AllowCpuFallback = true });
        backend.Initialize();

        var ex = Assert.Throws<InvalidOperationException>(() => backend.Execute([1f, 2f]));
        Assert.Contains("CpuExecutor", ex.Message);
    }

    [Fact(Timeout = 120000)]
    public async Task Execute_UsesConfiguredCpuExecutor_WhenNativeGraphIsUnavailable()
    {
        await Task.Yield();
        using var backend = new NNAPIBackend<float>(new NNAPIConfiguration { AllowCpuFallback = true });
        backend.Initialize();
        backend.CpuExecutor = input => [input[0] + 1f, input[1] + 2f];

        var output = backend.Execute([3f, 5f]);

        Assert.Equal([4f, 7f], output);
    }

    [Fact(Timeout = 120000)]
    public async Task Execute_AfterDispose_ThrowsObjectDisposedException()
    {
        await Task.Yield();
        var backend = new NNAPIBackend<float>(new NNAPIConfiguration { AllowCpuFallback = true });
        backend.Initialize();
        backend.Dispose();

        Assert.Throws<ObjectDisposedException>(() => backend.Execute([1f]));
    }
}
