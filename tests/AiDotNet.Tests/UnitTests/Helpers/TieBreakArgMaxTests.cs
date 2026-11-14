using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// Stable ordering under NumOps.Compare (CompareTo): NaN ranks below every number, negative infinity included.
/// </summary>
public class TieBreakArgMaxTests
{
    // Columns:                        0    1           2                         3    4           5
    private static readonly double[] Row = { 2.0, double.NaN, double.NegativeInfinity, 1.0, double.NaN, double.NegativeInfinity };

    [Fact]
    public void StableSmallestK_PutsNaNBeforeNegativeInfinity_TiesInColumnOrder()
    {
        // Ascending under CompareTo: NaN (cols 1, 4), then -inf (cols 2, 5), then 1 (col 3), then 2 (col 0).
        var values = new Tensor<double>((double[])Row.Clone(), new[] { 1, Row.Length });

        var order = TieBreakArgMax.StableSmallestK(values, Row.Length).ToArray();

        Assert.Equal(new[] { 1, 4, 2, 5, 3, 0 }, order);
    }

    [Fact]
    public void StableSmallestK_TakesTheFirstKOfThatOrder()
    {
        var values = new Tensor<double>((double[])Row.Clone(), new[] { 1, Row.Length });

        var order = TieBreakArgMax.StableSmallestK(values, 3).ToArray();

        Assert.Equal(new[] { 1, 4, 2 }, order);
    }

    [Fact]
    public void StableDescendingOrder1D_PutsNegativeInfinityBeforeNaN()
    {
        // Descending under CompareTo: 2 (col 0), 1 (col 3), -inf (cols 2, 5), NaN (cols 1, 4).
        var values = new Tensor<double>((double[])Row.Clone(), new[] { Row.Length });

        var order = TieBreakArgMax.StableDescendingOrder1D(values).ToArray();

        Assert.Equal(new[] { 0, 3, 2, 5, 1, 4 }, order);
    }

    [Fact]
    public void StableSmallestK_KeepsRowsIndependent()
    {
        // Row 0: [3, 3, 1] -> 2, 0, 1. Row 1: [-inf, NaN, -inf] -> 1, 0, 2.
        var values = new Tensor<double>(
            new[] { 3.0, 3.0, 1.0, double.NegativeInfinity, double.NaN, double.NegativeInfinity }, new[] { 2, 3 });

        var order = TieBreakArgMax.StableSmallestK(values, 3).ToArray();

        Assert.Equal(new[] { 2, 0, 1, 1, 0, 2 }, order);
    }
}
