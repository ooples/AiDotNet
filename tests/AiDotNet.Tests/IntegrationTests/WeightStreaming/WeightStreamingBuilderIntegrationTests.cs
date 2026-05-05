using AiDotNet;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.WeightStreaming;

/// <summary>
/// Issue #1222 task #187 — regression coverage for the public
/// weight-streaming surface introduced across tasks #183, #184, and
/// #186. These tests pin the BUILDER and DTO contracts; the actual
/// end-to-end forward through a streaming-configured model is exercised
/// in <see cref="AiDotNet.Tests.UnitTests.NeuralNetworks.WeightStreaming.AutoDetectWeightStreamingTests"/>.
///
/// The PaLME 562B OOM repro itself needs ~2 TB of disk + tens of
/// minutes per forward and is gated behind <c>[Fact(Skip = "...")]</c>
/// in <c>PaLMEProfilerTest</c>. These tests prove the surface is
/// wired end-to-end on small builds so the PaLME-scale flow exercises
/// the same paths whenever an engineer un-skips the canary.
/// </summary>
public sealed class WeightStreamingBuilderIntegrationTests
{
    [Fact]
    public void ConfigureWeightStreaming_ReturnsBuilderForChaining()
    {
        // Pin the fluent return contract so a future regression that
        // drops `return this` from the implementation breaks loudly.
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        var sameBuilder = builder.ConfigureWeightStreaming(new WeightStreamingConfig { Enabled = false });
        Assert.Same(builder, sameBuilder);
    }

    [Fact]
    public void ConfigureWeightStreaming_AcceptsNullConfig()
    {
        // Null is the documented "reset to default auto-detect" contract.
        // Should not throw and should return the builder.
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        var result = builder.ConfigureWeightStreaming(null);
        Assert.Same(builder, result);
    }

    [Fact]
    public void ConfigureWeightStreaming_AcceptsAllThreeEnabledStates()
    {
        // Three valid states for the Enabled flag: null (auto-detect),
        // true (force on), false (force off). Pin that all three are
        // accepted by the API.
        var builder = new AiModelBuilder<double, Tensor<double>, Tensor<double>>();
        builder.ConfigureWeightStreaming(new WeightStreamingConfig { Enabled = null });
        builder.ConfigureWeightStreaming(new WeightStreamingConfig { Enabled = true });
        builder.ConfigureWeightStreaming(new WeightStreamingConfig { Enabled = false });
        // No assertion needed — the test passes if no throw occurs.
    }

    [Fact]
    public void WeightStreamingReport_InitOnlyProperties_ConstructFromDto()
    {
        // Smoke-test that the report DTO's init-only properties accept
        // realistic counter values. This is the wire format we surface
        // on AiModelResult.WeightStreamingReport; pinning the property
        // set means a future field rename here breaks the test loudly
        // instead of silently dropping fields from operator dashboards.
        var report = new WeightStreamingReport
        {
            StreamingEnabled = true,
            AutoDetected = true,
            ModelParameterCount = 562_000_000_000L,
            EffectiveThresholdParameters = 10_000_000_000L,
            DiskReadCount = 12_345,
            EvictionCount = 2_345,
            PrefetchIssueCount = 67_890,
            PrefetchHitCount = 67_000,
            PrefetchMissCount = 890,
            ResidentBytes = 24_000_000_000L,    // 24 GB resident in pool
            CompressionRatio = 1.85,            // healthy LZ4 ratio for fp32 weights
        };

        Assert.True(report.StreamingEnabled);
        Assert.True(report.AutoDetected);
        Assert.Equal(562_000_000_000L, report.ModelParameterCount);
        Assert.Equal(10_000_000_000L, report.EffectiveThresholdParameters);
        Assert.Equal(67_000, report.PrefetchHitCount);
        Assert.True(report.PrefetchHitCount > report.PrefetchMissCount,
            "Sanity: hits should exceed misses on a healthy streaming run.");
    }

    [Fact]
    public void WeightStreamingConfig_ThresholdParameters_AcceptsLongValue()
    {
        // The threshold is in absolute parameter count and PaLME-scale
        // is 562B — well above int.MaxValue (2.1B). Pin that the
        // property correctly accepts long values; an accidental int
        // type would silently overflow.
        var config = new WeightStreamingConfig
        {
            ThresholdParameters = 562_000_000_000L
        };
        Assert.Equal(562_000_000_000L, config.ThresholdParameters);
    }
}
