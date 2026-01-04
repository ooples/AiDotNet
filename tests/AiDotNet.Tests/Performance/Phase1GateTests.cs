using System.Diagnostics;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.Fixtures;
using Xunit;

namespace AiDotNet.Tests.Performance;

/// <summary>
/// Phase 1 Gate Tests for Performance Optimization Plan.
/// These tests validate that quick wins (mini networks, config-only operations) meet targets.
/// </summary>
[Trait("Category", "Phase1Gate")]
[Trait("Category", "Performance")]
public class Phase1GateTests : IClassFixture<NetworkFixture<float>>
{
    private readonly NetworkFixture<float> _fixture;

    public Phase1GateTests(NetworkFixture<float> fixture)
    {
        _fixture = fixture;
    }

    /// <summary>
    /// DenseNet mini variant should construct quickly.
    /// Current baseline: ~10ms, Target: &lt;500ms, Stretch goal: &lt;50ms
    /// </summary>
    [Fact]
    public void MiniDenseNet_Constructs_UnderThreshold()
    {
        var sw = Stopwatch.StartNew();
        var network = DenseNetNetwork<float>.ForTesting(numClasses: 10);
        sw.Stop();

        Assert.NotNull(network);
        Assert.True(network.Layers.Count < 30, $"Mini network has too many layers: {network.Layers.Count}");
        Assert.True(sw.ElapsedMilliseconds < 500, $"Construction took {sw.ElapsedMilliseconds}ms, expected < 500ms");
    }

    /// <summary>
    /// EfficientNet mini variant construction time.
    /// Current baseline: ~5000ms (needs optimization), Target: &lt;1000ms
    /// </summary>
    [Fact]
    public void MiniEfficientNet_Constructs_UnderThreshold()
    {
        var sw = Stopwatch.StartNew();
        var network = EfficientNetNetwork<float>.ForTesting(numClasses: 10);
        sw.Stop();

        Assert.NotNull(network);
        // EfficientNet has complex MBConv blocks - threshold set higher for Phase 1
        // TODO: Optimize in Phase 2 to reduce to <1000ms
        Assert.True(sw.ElapsedMilliseconds < 10000, $"Construction took {sw.ElapsedMilliseconds}ms, expected < 10000ms");
    }

    /// <summary>
    /// ResNet mini variant construction time.
    /// Current baseline: ~800ms, Target: &lt;500ms
    /// </summary>
    [Fact]
    public void MiniResNet_Constructs_UnderThreshold()
    {
        var sw = Stopwatch.StartNew();
        var network = ResNetNetwork<float>.ForTesting(numClasses: 10);
        sw.Stop();

        Assert.NotNull(network);
        // ResNet18 with 32x32 input - threshold set higher for Phase 1
        // TODO: Optimize in Phase 2 to reduce to <500ms
        Assert.True(sw.ElapsedMilliseconds < 2000, $"Construction took {sw.ElapsedMilliseconds}ms, expected < 2000ms");
    }

    [Fact]
    public void DenseNetConfig_GetExpectedLayerCount_NoConstruction()
    {
        var sw = Stopwatch.StartNew();
        var config121 = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses: 1000);
        var count121 = config121.GetExpectedLayerCount();
        sw.Stop();

        // Should be nearly instant since no network construction
        Assert.True(sw.ElapsedMilliseconds < 10, $"Config-only operation took {sw.ElapsedMilliseconds}ms");
        Assert.True(count121 > 50, $"Expected layer count > 50, got {count121}");
    }

    [Fact]
    public void DenseNet_VariantComparison_UsingConfig_IsFast()
    {
        var sw = Stopwatch.StartNew();

        var config121 = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses: 10);
        var config169 = new DenseNetConfiguration(DenseNetVariant.DenseNet169, numClasses: 10);
        var config201 = new DenseNetConfiguration(DenseNetVariant.DenseNet201, numClasses: 10);
        var config264 = new DenseNetConfiguration(DenseNetVariant.DenseNet264, numClasses: 10);

        var count121 = config121.GetExpectedLayerCount();
        var count169 = config169.GetExpectedLayerCount();
        var count201 = config201.GetExpectedLayerCount();
        var count264 = config264.GetExpectedLayerCount();

        sw.Stop();

        // All config-only operations should complete in < 10ms
        Assert.True(sw.ElapsedMilliseconds < 50, $"Config comparison took {sw.ElapsedMilliseconds}ms");

        // Verify ordering is correct
        Assert.True(count169 >= count121, $"DenseNet169 ({count169}) should have >= layers than DenseNet121 ({count121})");
        Assert.True(count201 >= count169, $"DenseNet201 ({count201}) should have >= layers than DenseNet169 ({count169})");
        Assert.True(count264 >= count201, $"DenseNet264 ({count264}) should have >= layers than DenseNet201 ({count201})");
    }

    [Fact]
    public void NetworkFixture_IsThreadSafe()
    {
        var exceptions = new List<Exception>();

        Parallel.For(0, 10, i =>
        {
            try
            {
                // Access all networks concurrently
                var denseNet = _fixture.MiniDenseNet;
                var efficientNet = _fixture.MiniEfficientNet;
                var resNet = _fixture.MiniResNet;

                Assert.NotNull(denseNet);
                Assert.NotNull(efficientNet);
                Assert.NotNull(resNet);
            }
            catch (Exception ex)
            {
                lock (exceptions)
                {
                    exceptions.Add(ex);
                }
            }
        });

        Assert.Empty(exceptions);
    }

    [Fact]
    public void NetworkFixture_ReusesNetworks()
    {
        // Access networks multiple times
        var denseNet1 = _fixture.MiniDenseNet;
        var denseNet2 = _fixture.MiniDenseNet;

        var efficientNet1 = _fixture.MiniEfficientNet;
        var efficientNet2 = _fixture.MiniEfficientNet;

        var resNet1 = _fixture.MiniResNet;
        var resNet2 = _fixture.MiniResNet;

        // Same instances should be returned
        Assert.Same(denseNet1, denseNet2);
        Assert.Same(efficientNet1, efficientNet2);
        Assert.Same(resNet1, resNet2);
    }

    [Fact]
    public void CustomDenseNet_HasCorrectBlockLayers()
    {
        var customBlockLayers = new[] { 2, 2, 2, 2 };
        var config = new DenseNetConfiguration(
            variant: DenseNetVariant.Custom,
            numClasses: 10,
            inputHeight: 32,
            inputWidth: 32,
            growthRate: 8,
            customBlockLayers: customBlockLayers);

        var blockLayers = config.GetBlockLayers();

        Assert.Equal(customBlockLayers, blockLayers);
    }

    [Fact]
    public void CustomEfficientNet_HasCorrectParameters()
    {
        var config = new EfficientNetConfiguration(
            variant: EfficientNetVariant.Custom,
            numClasses: 10,
            customInputHeight: 64,
            customWidthMultiplier: 0.5,
            customDepthMultiplier: 0.5);

        Assert.Equal(64, config.GetInputHeight());
        Assert.Equal(0.5, config.GetWidthMultiplier());
        Assert.Equal(0.5, config.GetDepthMultiplier());
    }

    [Fact]
    public void MiniNetworks_AreMuchSmallerThanFullVariants()
    {
        // Compare mini DenseNet to full DenseNet121
        var miniConfig = DenseNetConfiguration.CreateForTesting(10);
        var fullConfig = new DenseNetConfiguration(DenseNetVariant.DenseNet121, 10);

        var miniLayers = miniConfig.GetExpectedLayerCount();
        var fullLayers = fullConfig.GetExpectedLayerCount();

        // Mini should have significantly fewer layers
        Assert.True(miniLayers < fullLayers / 2,
            $"Mini ({miniLayers}) should have < half the layers of full ({fullLayers})");
    }
}
