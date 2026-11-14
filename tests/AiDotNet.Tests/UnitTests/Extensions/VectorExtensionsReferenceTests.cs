using System;
using AiDotNet.Extensions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Extensions;

/// <summary>
/// Reference values for the <see cref="VectorExtensions"/> members whose engine-op conversion (#2490) also
/// fixed a wrong result.
/// </summary>
public class VectorExtensionsReferenceTests
{
    [Fact]
    public void PointwiseSqrt_Half_ReturnsSquareRoots()
    {
        // Half is not IConvertible, so the former Convert.ToDouble route threw InvalidCastException.
        var vector = new Vector<Half>(new[] { (Half)4, (Half)9, (Half)16 });

        var roots = vector.PointwiseSqrt();

        Assert.Equal(3, roots.Length);
        Assert.Equal((Half)2, roots[0]);
        Assert.Equal((Half)3, roots[1]);
        Assert.Equal((Half)4, roots[2]);
    }

    [Fact]
    public void Argsort_WithNaN_SortsTheFiniteValuesAndPutsNaNLast()
    {
        // numpy.argsort([2, nan, 1, nan], kind="stable") = [2, 0, 1, 3]: finite values ascending, then the NaNs in
        // their original order. Ranking the values by a binary search over a sequence that contains NaN gave the
        // NaNs (or everything) the same rank, so the finite values came back out of order.
        var vector = new Vector<double>(new[] { 2.0, double.NaN, 1.0, double.NaN });

        Assert.Equal(new[] { 2, 0, 1, 3 }, vector.Argsort());
    }

    [Fact]
    public void Argsort_Ties_KeepTheirOriginalOrder()
    {
        var vector = new Vector<double>(new[] { 3.0, 1.0, 3.0, 1.0, 2.0 });

        Assert.Equal(new[] { 1, 3, 4, 0, 2 }, vector.Argsort());
    }

    [Fact]
    public void EuclideanDistance_Half_MoreElementsThanHalfMax_IsFinite()
    {
        // 90000 differences of 1: the distance is sqrt(90000) = 300, but the sum of squares, 90000, is above the
        // largest Half (65504), and pre-scaling the differences by their largest magnitude (1) does not bound it.
        const int n = 90000;
        var ones = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            ones[i] = (Half)1;
        }

        Assert.Equal(300.0, (double)ones.EuclideanDistance(new Vector<Half>(n)), 0.5);
    }

    [Fact]
    public void StandardDeviation_Half_MoreElementsThanHalfMax_IsFinite()
    {
        // 90000 values alternating -1 and 1: mean 0 and sample standard deviation sqrt(90000 / 89999) = 1.0000056.
        // Both the sum of squares and n - 1 are above the largest Half, which made the result NaN.
        const int n = 90000;
        var values = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            values[i] = i % 2 == 0 ? (Half)(-1) : (Half)1;
        }

        Assert.Equal(1.0, (double)values.StandardDeviation(), 1e-2);
    }
}
