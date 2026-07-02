using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// xUnit collection for the GPU execution-graph diffusion tests. These test classes mutate the
/// process-global <c>AiDotNetEngine.Current</c> and shared diffusion diagnostics counters, so they must
/// NOT run in parallel with each other or with any other test that touches the global engine. Marking the
/// collection <c>DisableParallelization = true</c> serializes every class assigned to it and keeps it out
/// of the parallel test pool, preventing cross-test engine/diagnostics interference.
/// </summary>
[CollectionDefinition("DiffusionGpuCuda", DisableParallelization = true)]
public sealed class DiffusionGpuExecutionGraphCollection
{
}
