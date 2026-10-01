using System;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// RBFLayer's initial widths follow the P-nearest-neighbour heuristic of Moody and Darken (1989).
/// </summary>
/// <remarks>
/// Widths drawn from U(0, 1) could start near zero, which turns a Gaussian unit into a spike that is zero
/// for every input: 25 of 50,000 freshly initialized generated fixtures mapped every probed input to one
/// output. Widths set from the spacing of the centers cannot collapse that way.
/// </remarks>
public class RBFLayerWidthInitializationTests
{
    private static double[] Field(RBFLayer<double> layer, string name)
        => ((Tensor<double>)typeof(RBFLayer<double>)
                .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!
                .GetValue(layer)!)
            .ToArray();

    [Theory]
    [InlineData(4, 4)]
    [InlineData(3, 16)]
    [InlineData(8, 2)]
    public void EachWidthIsTheRmsDistanceToItsTwoNearestCenters(int inputSize, int centers)
    {
        var layer = new RBFLayer<double>(inputSize, centers);
        double[] c = Field(layer, "_centers");
        double[] widths = Field(layer, "_widths");

        Assert.Equal(centers, widths.Length);
        for (int j = 0; j < centers; j++)
        {
            var nearest = Enumerable.Range(0, centers)
                .Where(i => i != j)
                .Select(i => Enumerable.Range(0, inputSize)
                    .Sum(d => Math.Pow(c[j * inputSize + d] - c[i * inputSize + d], 2)))
                .OrderBy(d2 => d2)
                .Take(2)
                .ToArray();
            double expected = Math.Sqrt(nearest.Average());

            Assert.Equal(expected, widths[j], 12);
            Assert.True(widths[j] > 0, $"width {j} is {widths[j]}");
        }
    }

    [Fact]
    public void ASingleCenterFallsBackToUnitWidth()
    {
        var layer = new RBFLayer<double>(4, 1);

        Assert.Equal(new[] { 1.0 }, Field(layer, "_widths"));
    }
}
