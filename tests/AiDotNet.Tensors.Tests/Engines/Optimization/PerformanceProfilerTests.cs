using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

public class PerformanceProfilerTests
{
    [Fact]
    public void Profile_WhenEnabled_RecordsStats()
    {
        var profiler = PerformanceProfiler.Instance;
        string operationName = $"op-{Guid.NewGuid():N}";
        bool wasEnabled = profiler.Enabled;

        profiler.Clear();
        profiler.Enabled = true;

        try
        {
            using (profiler.Profile(operationName))
            {
                _ = 1 + 1;
            }

            var stats = profiler.GetStats(operationName);
            Assert.NotNull(stats);
            Assert.Equal(operationName, stats!.OperationName);
            Assert.Equal(1, stats.CallCount);
            Assert.True(stats.TotalTicks > 0);
        }
        finally
        {
            profiler.Enabled = wasEnabled;
            profiler.Clear();
        }
    }

    [Fact]
    public void Profile_WhenDisabled_ReturnsEmptyDisposable()
    {
        var profiler = PerformanceProfiler.Instance;
        string operationName = $"op-disabled-{Guid.NewGuid():N}";
        bool wasEnabled = profiler.Enabled;

        profiler.Clear();
        profiler.Enabled = false;

        try
        {
            using (profiler.Profile(operationName))
            {
                _ = 1 + 1;
            }

            Assert.Null(profiler.GetStats(operationName));
        }
        finally
        {
            profiler.Enabled = wasEnabled;
            profiler.Clear();
        }
    }
}
