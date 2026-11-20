using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// <see cref="TensorReductionHelper.SumOfSquares{T}"/> against a plain loop, including the double SIMD
/// reduction regression fixed upstream in AiDotNet.Tensors #1733.
/// </summary>
public class TensorSumOfSquaresDoubleTests
{
    public static IEnumerable<object[]> Lengths() =>
        Enumerable.Range(1, 40).Concat(new[] { 63, 64, 65, 127, 128, 1000, 100000 }).Select(n => new object[] { n });

    private static IEnumerable<IEngine> Engines()
    {
        yield return new CpuEngine();
        yield return AiDotNetEngine.Current;
    }

    // Mixed sign and magnitude, and different at even and odd indices (the defect cancels when they match).
    private static double[] Values(int n, int seed)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = Math.Sin(0.9 * i + seed) * (1 + (i * 7 + seed) % 5) - 0.25;
        }

        return values;
    }

    private static double LoopSumOfSquares(IEnumerable<double> values)
    {
        double sum = 0;
        foreach (double v in values) sum += v * v;
        return sum;
    }

    private static void AssertClose(double expected, double actual, double relativeTolerance = 1e-12)
    {
        Assert.True(
            Math.Abs(actual - expected) <= relativeTolerance * Math.Max(Math.Abs(expected), double.Epsilon),
            $"expected {expected:R}, got {actual:R}");
    }

    /// <summary>Every SIMD lane must contribute once to the double sum of squares.</summary>
    [Fact]
    public void EngineTensorSumOfSquares_Double_AllSixteenElementsContribute()
    {
        var oneToSixteen = Enumerable.Range(1, 16).Select(i => (double)i).ToArray();
        double engineValue = new CpuEngine().TensorSumOfSquares(new Tensor<double>(oneToSixteen, new[] { 16 }));

        // The old AVX2/FMA horizontal reduction dropped odd-index squares and returned 1360.
        Assert.Equal(1496.0, engineValue);
    }

    [Fact]
    public void SumOfSquares_SparseDouble_MatchesTheDenseLoop()
    {
        // A 4 x 8 COO tensor whose stored values sit at odd flat indices (odd columns), plus a duplicate coordinate
        // that must accumulate before the sum of squares is taken.
        var rows = new List<int>();
        var columns = new List<int>();
        var values = new List<double>();
        var dense = new double[32];
        for (int r = 0; r < 4; r++)
        {
            for (int c = 1; c < 8; c += 2)
            {
                double v = 1.5 + r * 4 + c;
                rows.Add(r);
                columns.Add(c);
                values.Add(v);
                dense[r * 8 + c] += v;
            }
        }

        rows.Add(0);
        columns.Add(1);
        values.Add(-0.75);
        dense[1] += -0.75;

        var sparse = new SparseTensor<double>(4, 8, rows.ToArray(), columns.ToArray(), values.ToArray());

        foreach (var engine in Engines())
        {
            AssertClose(LoopSumOfSquares(dense), TensorReductionHelper.SumOfSquares(engine, sparse));
            AssertClose(LoopSumOfSquares(dense), TensorReductionHelper.SumOfSquaresAsDouble(engine, sparse));
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(40)]
    [InlineData(1000)]
    public void SumOfSquaresAsDouble_Double_MatchesALoop(int n)
    {
        var values = Values(n, seed: 2);
        foreach (var engine in Engines())
        {
            AssertClose(LoopSumOfSquares(values),
                TensorReductionHelper.SumOfSquaresAsDouble(engine, new Tensor<double>((double[])values.Clone(), new[] { n })));
        }
    }

    [Fact]
    public void SumOfSquaresAsDouble_EmptyTensor_IsZero()
    {
        Assert.Equal(0.0, TensorReductionHelper.SumOfSquaresAsDouble(new CpuEngine(), new Tensor<float>(new[] { 0 })));
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData(40)]
    [InlineData(1000)]
    [InlineData(70000)]
    public void SumOfSquaresAsDouble_Half_StaysFinitePastHalfMaxValue(int n)
    {
        // Elements of a few hundred: every element and every square-root statistic fits Half, the sum of squares
        // (4e5 at n = 40) does not. SumOfSquares returns infinity there; the double sum must not.
        var values = Enumerable.Range(0, n).Select(i => (Half)(150.0 + 100.0 * Math.Sin(0.9 * i))).ToArray();
        double expected = LoopSumOfSquares(values.Select(v => (double)v));
        Assert.True(expected > (double)Half.MaxValue);

        double actual = TensorReductionHelper.SumOfSquaresAsDouble(new CpuEngine(), new Tensor<Half>((Half[])values.Clone(), new[] { n }));

        // The rescaled elements keep Half's 11 significant bits.
        AssertClose(expected, actual, 5e-3);
    }

    [Fact]
    public void SumOfSquaresAsDouble_Half_KeepsInfinityAndNaN()
    {
        var withInfinity = new[] { (Half)1.0, Half.PositiveInfinity, (Half)2.0 };
        var withNaN = new[] { (Half)1.0, Half.NaN, (Half)2.0 };

        Assert.True(double.IsPositiveInfinity(
            TensorReductionHelper.SumOfSquaresAsDouble(new CpuEngine(), new Tensor<Half>(withInfinity, new[] { 3 }))));
        Assert.True(double.IsNaN(
            TensorReductionHelper.SumOfSquaresAsDouble(new CpuEngine(), new Tensor<Half>(withNaN, new[] { 3 }))));
    }
#endif

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SumOfSquares_Double_MatchesALoop_AtEveryLength(int n)
    {
        var values = Values(n, seed: n);
        double expected = LoopSumOfSquares(values);

        foreach (var engine in Engines())
        {
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, new Tensor<double>((double[])values.Clone(), new[] { n })));
        }
    }

    [Fact]
    public void SumOfSquares_Double_CountsElementsAtOddIndices()
    {
        // Only odd indices are non-zero: the engine op returns 0 for this tensor.
        var values = new double[32];
        for (int i = 1; i < 32; i += 2) values[i] = i;
        double expected = LoopSumOfSquares(values);

        foreach (var engine in Engines())
        {
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, new Tensor<double>((double[])values.Clone(), new[] { 32 })));
        }
    }

    [Theory]
    [InlineData(16)]
    [InlineData(36)]
    [InlineData(40)]
    [InlineData(128)]
    [InlineData(1000)]
    public void SumOfSquares_Double_MatchesALoop_OnRank2AndViews(int n)
    {
        var values = Values(n + 7, seed: 3);
        var body = values.Skip(3).Take(n).ToArray();
        double expected = LoopSumOfSquares(body);
        int rows = 4, columns = n / 4;

        foreach (var engine in Engines())
        {
            Tensor<double> Rank2() => new((double[])body.Clone(), new[] { rows, columns });
            Tensor<double> Padded() => new((double[])values.Clone(), new[] { n + 7 });

            // A [n, 3] tensor whose middle column is the data: a strided, non-contiguous view once narrowed.
            var wide = new double[n * 3];
            for (int i = 0; i < n; i++)
            {
                wide[i * 3] = 7.0;
                wide[i * 3 + 1] = body[i];
                wide[i * 3 + 2] = -3.0;
            }

            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, Rank2()));
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, Rank2().Transpose()));
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, engine.TensorPermute(Rank2(), new[] { 1, 0 })));
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, engine.TensorNarrow(Padded(), 0, 3, n)));
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, engine.TensorSlice(Padded(), new[] { 3 }, new[] { n })));
            AssertClose(expected, TensorReductionHelper.SumOfSquares(
                engine, engine.TensorNarrow(new Tensor<double>(wide, new[] { n, 3 }), 1, 1, 1)));

            // The result of an engine op, as the call sites pass it (centred values).
            double centred = LoopSumOfSquares(body.Select(v => v - 0.25));
            AssertClose(centred, TensorReductionHelper.SumOfSquares(
                engine, engine.TensorSubtractScalar(new Tensor<double>((double[])body.Clone(), new[] { n }), 0.25)));
        }
    }

    [Fact]
    public void SumOfSquares_Double_EmptyTensor_IsZero()
    {
        Assert.Equal(0.0, TensorReductionHelper.SumOfSquares(new CpuEngine(), new Tensor<double>(new[] { 0 })));
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void SumOfSquares_Float_MatchesALoop(int n)
    {
        var values = Values(n, seed: n).Select(v => (float)v).ToArray();
        double expected = LoopSumOfSquares(values.Select(v => (double)v));

        foreach (var engine in Engines())
        {
            AssertClose(expected, TensorReductionHelper.SumOfSquares(engine, new Tensor<float>((float[])values.Clone(), new[] { n })), 2e-5);
        }
    }

    [Theory]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(40)]
    [InlineData(1000)]
    public void SumOfSquares_Decimal_MatchesALoop(int n)
    {
        var values = Values(n, seed: 5).Select(v => (decimal)v).ToArray();
        decimal expected = 0m;
        foreach (decimal v in values) expected += v * v;

        decimal actual = TensorReductionHelper.SumOfSquares(new CpuEngine(), new Tensor<decimal>((decimal[])values.Clone(), new[] { n }));

        AssertClose((double)expected, (double)actual);
    }

#if NET5_0_OR_GREATER
    [Theory]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(40)]
    [InlineData(1000)]
    public void SumOfSquares_Half_MatchesALoop(int n)
    {
        var values = Values(n, seed: 5).Select(v => (Half)v).ToArray();
        double expected = LoopSumOfSquares(values.Select(v => (double)v));

        double actual = (double)TensorReductionHelper.SumOfSquares(new CpuEngine(), new Tensor<Half>((Half[])values.Clone(), new[] { n }));

        // Half keeps 11 significant bits.
        AssertClose(expected, actual, 2e-3);
    }
#endif
}
