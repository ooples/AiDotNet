using System;
using System.Collections.Generic;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

/// <summary>
/// Per-step sample quantiles (used by Chronos and DeepAR) must rank NaN samples exactly like master's
/// double sort (List&lt;double&gt;.Sort / OrderBy), which places NaN before every number, on every sort path.
/// </summary>
public class SampleQuantilesNaNTests
{
    private const double NaN = double.NaN;
    private const double Inf = double.PositiveInfinity;

    // Five sampled paths over three steps. Ranked per step, NaN first:
    // step 0: NaN, NaN, 0.5, 1, 2   step 1: NaN, NaN, -3, 2, 4   step 2: NaN, -1, 3, 5, +inf
    private static readonly double[][] MultiStepSamples =
    {
        new[] { 1.0, NaN, 3.0 },
        new[] { NaN, 2.0, NaN },
        new[] { 0.5, NaN, -1.0 },
        new[] { 2.0, 4.0, Inf },
        new[] { NaN, -3.0, 5.0 },
    };

    private static readonly double[] MultiStepQuantiles = { 0.0, 0.2, 0.5, 0.7, 0.9, 1.0 };

    private static readonly double[][] MultiStepExpected =
    {
        new[] { NaN, NaN, NaN },    // q = 0.0 -> rank 0
        new[] { NaN, NaN, -1.0 },   // q = 0.2 -> rank 1
        new[] { 0.5, -3.0, 3.0 },   // q = 0.5 -> rank 2
        new[] { 1.0, 2.0, 5.0 },    // q = 0.7 -> rank 3
        new[] { 2.0, 4.0, Inf },    // q = 0.9 -> rank 4
        new[] { 2.0, 4.0, Inf },    // q = 1.0 -> rank 5 clamped to 4
    };

    // A single step takes the engine's fast float sort path (one sorted axis, inner size 1)
    private static readonly double[][] SingleStepSamples =
    {
        new[] { 3.0 }, new[] { NaN }, new[] { 1.0 }, new[] { NaN }, new[] { 2.0 },
    };

    private static readonly double[] SingleStepQuantiles = { 0.0, 0.3, 0.4, 0.6, 0.99 };

    // Ranked: NaN, NaN, 1, 2, 3
    private static readonly double[][] SingleStepExpected =
    {
        new[] { NaN }, new[] { NaN }, new[] { 1.0 }, new[] { 2.0 }, new[] { 3.0 },
    };

    [Fact]
    public void MultiStep_Double_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<double>(MultiStepSamples, MultiStepQuantiles, MultiStepExpected, v => v, v => v);
    }

    [Fact]
    public void MultiStep_Float_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<float>(MultiStepSamples, MultiStepQuantiles, MultiStepExpected, v => (float)v, v => v);
    }

    [Fact]
    public void SingleStep_Double_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<double>(SingleStepSamples, SingleStepQuantiles, SingleStepExpected, v => v, v => v);
    }

    [Fact]
    public void SingleStep_Float_FastSortPath_RanksNaNFirstLikeMaster()
    {
        AssertQuantiles<float>(SingleStepSamples, SingleStepQuantiles, SingleStepExpected, v => (float)v, v => v);
    }

    [Fact]
    public void Expectations_MatchMastersDoubleSort()
    {
        // The hard-coded expectations are what master's List<double>.Sort-based pick returns
        AssertMatchesReference(MultiStepSamples, MultiStepQuantiles, MultiStepExpected);
        AssertMatchesReference(SingleStepSamples, SingleStepQuantiles, SingleStepExpected);
    }

    private static void AssertQuantiles<T>(
        double[][] samples, double[] quantiles, double[][] expected, Func<double, T> fromDouble, Func<T, double> toDouble)
    {
        var paths = new List<Vector<T>>();
        foreach (var sample in samples)
        {
            paths.Add(new Vector<T>(Array.ConvertAll(sample, v => fromDouble(v))));
        }

        int horizon = samples[0].Length;
        var result = TimeSeriesEngineOps<T>.SampleQuantiles(paths, quantiles, horizon);

        Assert.Equal(quantiles.Length, result.Count);
        for (int k = 0; k < quantiles.Length; k++)
        {
            var actual = result[quantiles[k]];
            Assert.Equal(horizon, actual.Length);
            for (int h = 0; h < horizon; h++)
            {
                double value = toDouble(actual[h]);
                if (double.IsNaN(expected[k][h]))
                {
                    Assert.True(double.IsNaN(value), $"q={quantiles[k]}, step {h}: expected NaN, got {value}");
                }
                else
                {
                    Assert.Equal(expected[k][h], value);
                }
            }
        }
    }

    private static void AssertMatchesReference(double[][] samples, double[] quantiles, double[][] expected)
    {
        int horizon = samples[0].Length;
        for (int k = 0; k < quantiles.Length; k++)
        {
            for (int h = 0; h < horizon; h++)
            {
                var values = new List<double>(samples.Length);
                foreach (var sample in samples)
                {
                    values.Add(sample[h]);
                }

                values.Sort();
                int idx = (int)(quantiles[k] * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                double reference = values[idx];
                if (double.IsNaN(reference))
                {
                    Assert.True(double.IsNaN(expected[k][h]));
                }
                else
                {
                    Assert.Equal(reference, expected[k][h]);
                }
            }
        }
    }
}
