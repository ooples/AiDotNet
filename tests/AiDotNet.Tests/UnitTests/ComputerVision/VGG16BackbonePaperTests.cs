using System;
using System.Linq;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

public class VGG16BackbonePaperTests
{
    [Fact]
    public void ConfigurationD_ProducesTheFourDetectionPyramidLevels()
    {
        var options = new VGG16BackboneOptions { Seed = 19, MaxGradNorm = 0.25 };
        using var model = new VGG16Backbone<double>(options);
        Assert.Same(options, model.GetOptions());
        // Simonyan and Zisserman, Table 1, configuration D: 13 convolutions, five 2x2 pools.
        Assert.Equal(14_714_688, model.GetParameters().Length);
        model.LoadTorchWeights((name, shape) => Enumerable.Repeat(
            name.EndsWith(".bias", StringComparison.Ordinal) ? 1.0 : 0.0,
            shape.Aggregate(1, (count, extent) => count * extent)).ToArray());
        var levels = model.ExtractFeatures(new Tensor<double>(new[] { 1, 3, 32, 32 }));
        var channels = new[] { 128, 256, 512, 512 };
        var widths = new[] { 8, 4, 2, 1 };
        Assert.Equal(4, levels.Count);
        for (int i = 0; i < levels.Count; i++)
        {
            Assert.Equal(new[] { 1, channels[i], widths[i], widths[i] }, levels[i].Shape.ToArray());
            // Zero kernels and unit biases give ReLU(1)=1, preserved by every max pool.
            Assert.All(levels[i].ToArray(), value => Assert.Equal(1.0, value));
        }
    }
}
