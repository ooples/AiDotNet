using System;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// Dedicated-core execution for foundation-scale model tests (Tier-4 / AiDotNet#1622).
/// <para>
/// Foundation-scale ModelFamily tests (large VLMs, large diffusion models) run a forward whose
/// managed <c>BlasManaged</c> GEMMs parallelize over <see cref="CpuParallelSettings.MaxDegreeOfParallelism"/>
/// — which defaults to <see cref="Environment.ProcessorCount"/>. With the default xUnit
/// <c>maxParallelThreads</c> (all logical cores) the runner runs that many test classes at once,
/// so a single heavy forward's managed GEMM contends with every other class's GEMM for the cores
/// (N classes × N managed threads ≫ cores). On the 8-core CI runner that contention is what pushes
/// the 2-forward Clone/Deterministic/ScaledInput checks past their <c>[Fact(Timeout)]</c>.
/// </para>
/// <para>
/// This collection sets <c>DisableParallelization = true</c>, so its member classes run one at a
/// time and never concurrently with any other collection — i.e. each foundation-scale forward gets
/// the whole machine, exactly the "dedicated cores" the plan calls for. (Native BLAS is already
/// pinned to one thread per worker in <c>TestModuleInitializer</c> for the small-test path; the
/// managed engine is what these heavy forwards use, and serialization is what un-contends it.)
/// </para>
/// </summary>
public sealed class FoundationScaleCpuFixture : IDisposable
{
    public FoundationScaleCpuFixture()
    {
        TestModuleInitializer.EnsureInitialized();
        // Serialized → no concurrent test contends, so let the heavy forward use every core.
        // (Defensive: restore the all-cores default in case an earlier test capped it.)
        if (CpuParallelSettings.MaxDegreeOfParallelism < Environment.ProcessorCount)
            CpuParallelSettings.MaxDegreeOfParallelism = Environment.ProcessorCount;
    }

    public void Dispose() { }
}

/// <summary>
/// Collection that serializes foundation-scale model tests so each gets dedicated cores. Apply
/// <c>[Collection("FoundationScaleSerial")]</c> to a ModelFamily test class whose model is
/// foundation-scale (it appears on the AiDotNet#1622 inventory).
/// </summary>
[CollectionDefinition("FoundationScaleSerial", DisableParallelization = true)]
public class FoundationScaleSerialCollection : ICollectionFixture<FoundationScaleCpuFixture>
{
    // No code; the attribute is the point. Member classes run serially with the whole machine.
}
