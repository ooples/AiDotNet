using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The population convergence check is computed in bounded coordinate blocks; the result must still equal the
/// definition (1 / (count * dimension)) * sum_k sum_i |x_k,i - mean_i| when the coordinates span several blocks
/// and the last block is partial.
/// </summary>
public class PopulationMeanAbsoluteDeviationTests
{
    [Theory]
    [InlineData(4, 3)]
    [InlineData(4, (1 << 17) + 5)]
    [InlineData(3, (1 << 19) + 7)]
    public void MatchesTheDefinitionAcrossCoordinateBlocks(int count, int dimension)
    {
        var points = new List<Vector<double>>();
        for (int k = 0; k < count; k++)
        {
            var point = new Vector<double>(dimension);
            for (int i = 0; i < dimension; i++)
            {
                point[i] = ((k + 1) * (i % 11)) % 7 - 0.5 * k;
            }

            points.Add(point);
        }

        double expected = 0.0;
        for (int i = 0; i < dimension; i++)
        {
            double mean = 0.0;
            for (int k = 0; k < count; k++) mean += points[k][i];
            mean /= count;
            for (int k = 0; k < count; k++) expected += Math.Abs(points[k][i] - mean);
        }

        expected /= (double)count * dimension;

        var optimizer = new DifferentialEvolutionOptimizer<double, Matrix<double>, Vector<double>>(null);
        var method = typeof(OptimizerBase<double, Matrix<double>, Vector<double>>).GetMethod(
            "PopulationMeanAbsoluteDeviation", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(method);

        double actual = Assert.IsType<double>(method.Invoke(optimizer, new object[] { points, dimension }));

        Assert.Equal(expected, actual, 9);
    }
}
