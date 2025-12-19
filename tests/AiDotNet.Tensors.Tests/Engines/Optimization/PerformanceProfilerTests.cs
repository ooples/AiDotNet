using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

public class PerformanceProfilerTests
{
    [Fact]
    public void Profile_WhenEnabled_RecordsStats()
    {
        var profiler = PerformanceProfiler.Instance;
        profiler.Clear();
        profiler.Enabled = true;

        using (profiler.Profile("op"))
        {
            _ = 1 + 1;
        }

        var stats = profiler.GetStats("op");
        Assert.NotNull(stats);
        Assert.Equal("op", stats!.OperationName);
        Assert.True(stats.CallCount >= 1);
        Assert.True(stats.TotalTicks > 0);
    }

    [Fact]
    public void Profile_WhenDisabled_ReturnsEmptyDisposable()
    {
        var profiler = PerformanceProfiler.Instance;
        profiler.Clear();
        profiler.Enabled = false;

        using (profiler.Profile("op-disabled"))
        {
            _ = 1 + 1;
        }

        Assert.Null(profiler.GetStats("op-disabled"));
    }
}

