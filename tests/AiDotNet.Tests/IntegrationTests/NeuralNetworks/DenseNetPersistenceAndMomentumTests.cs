using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// DenseNet's batch-norm momentum convention and the strict reading of persisted layer behaviour flags.
/// </summary>
public class DenseNetPersistenceAndMomentumTests
{
    [Fact]
    public void DenseBlockLayer_PyTorchMomentum_KeepsNinetyPercentOfRunningStatistics()
    {
        // torchvision DenseNet: momentum 0.1 weights the new batch by 0.1, so the running statistics keep 0.9.
        // BatchNormalizationLayer takes the kept weight, so the layer must receive 1 - 0.1.
        var layer = new DenseBlockLayer<float>(growthRate: 4, bnMomentum: 0.1);

        foreach (string name in new[] { "_bn1", "_bn2" })
        {
            var field = typeof(DenseBlockLayer<float>).GetField(name, BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(field);
            var batchNorm = Assert.IsType<BatchNormalizationLayer<float>>(field?.GetValue(layer));
            Assert.Equal(0.9f, batchNorm.GetMomentum(), 6);
        }
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.5)]
    [InlineData(double.NaN)]
    public void DenseBlockLayer_MomentumOutsideUnitInterval_Throws(double momentum)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseBlockLayer<float>(growthRate: 4, bnMomentum: momentum));
    }

    [Fact]
    public void DenseBlock_MalformedConvolutionBias_IsRejected()
    {
        var metadata = new Dictionary<string, object>
        {
            ["NumLayers"] = 2,
            ["GrowthRate"] = 4,
            ["ConvolutionBias"] = "invalid",
        };

        var ex = Assert.Throws<InvalidOperationException>(() => DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(DenseBlock<>).Name,
            inputShape: new[] { 1, 8, 4, 4 },
            outputShape: new[] { 1, 16, 4, 4 },
            additionalParams: metadata));
        Assert.Contains("ConvolutionBias", ex.Message);
    }

    [Fact]
    public void DenseBlock_AbsentConvolutionBias_KeepsLegacyBiasedLayout()
    {
        var metadata = new Dictionary<string, object> { ["NumLayers"] = 2, ["GrowthRate"] = 4 };

        var layer = DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(DenseBlock<>).Name,
            inputShape: new[] { 1, 8, 4, 4 },
            outputShape: new[] { 1, 16, 4, 4 },
            additionalParams: metadata);

        Assert.IsType<DenseBlock<float>>(layer);
    }
}
