using System.Linq;
using System.Threading.Tasks;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;

namespace AiDotNet.CompetitiveBenchmarks;

/// <summary>
/// Entry point for the cross-framework competitive benchmarks.
/// <list type="bullet">
/// <item><c>dotnet run -c Release -- --filter *SemanticKernel*</c> — BenchmarkDotNet head-to-head vs Semantic Kernel (.NET).</item>
/// <item><c>dotnet run -c Release -- --langgraph [iterations]</c> — cross-runtime comparison vs LangGraph (Python sidecar).</item>
/// </list>
/// </summary>
public static class Program
{
    public static async Task Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--langgraph")
        {
            await LangGraphSidecarRunner.RunAsync(args.Skip(1).ToArray());
            return;
        }

        // Use the in-process toolchain so BenchmarkDotNet does not regenerate + build a standalone harness
        // project. That regeneration otherwise collides with the repo's central package management (the root
        // Directory.Packages.props forces CPM on, while this project pins Semantic Kernel inline), failing the
        // generated build. In-process runs the benchmarks in this process — correct for an orchestration
        // micro-benchmark where there is no JIT-warmup-sensitive native interop to isolate.
        var config = DefaultConfig.Instance.AddJob(Job.Default.WithToolchain(InProcessEmitToolchain.Instance));
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
    }
}
