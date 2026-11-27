using System;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// <see cref="TensorReductionHelper.StandardizeRescaled{T}"/> against the direct standardization it replaces when the
/// direct statistics overflow (and always in graph mode).
/// </summary>
public class StandardizeRescaledTests
{
    // A constant Half slice with max |x| >= 255: (max |x| + 1)^2 overflowed Half, so the scaled epsilon became 0 and a
    // rounding residue in the unit mean set the standard deviation and standardized the values to +-1. The direct
    // form gives 0 and sqrt(epsilon).
    [Theory]
    [InlineData(1000.0)]
    [InlineData(-30000.0)]
    [InlineData(0.1)]
    public void ConstantHalfSlice_StandardizesToZero_WithTheEpsilonFloor(double value)
    {
        const double epsilon = 1e-5;
        var x = new Tensor<Half>(Enumerable.Repeat((Half)value, 6).ToArray(), new[] { 2, 3 });

        var (standardized, mean, std) = TensorReductionHelper.StandardizeRescaled(
            AiDotNetEngine.Current, x, new[] { 1 }, epsilon, 3);

        Assert.All(standardized.ToArray(), v => Assert.Equal(0.0, (double)v));
        Assert.All(mean.ToArray(), v => Assert.Equal((double)(Half)value, (double)v));
        Assert.All(std.ToArray(), v => Assert.Equal(Math.Sqrt(epsilon), (double)v, 4));
    }

    [Fact]
    public void SpreadDoubleSlice_MatchesTheDirectForm()
    {
        const double epsilon = 1e-5;
        var values = new[] { 3.0, -1.5, 7.25, 1000.0, 1001.0, 999.5 };
        var x = new Tensor<double>(values, new[] { 2, 3 });

        var (standardized, mean, std) = TensorReductionHelper.StandardizeRescaled(
            AiDotNetEngine.Current, x, new[] { 1 }, epsilon, 3);

        for (int row = 0; row < 2; row++)
        {
            var slice = values.Skip(row * 3).Take(3).ToArray();
            double m = slice.Average();
            double s = Math.Sqrt(slice.Sum(v => (v - m) * (v - m)) / 3 + epsilon);
            Assert.Equal(m, mean[row, 0], 10);
            Assert.Equal(s, std[row, 0], 10);
            for (int i = 0; i < 3; i++)
                Assert.Equal((slice[i] - m) / s, standardized[row, i], 10);
        }
    }
}
