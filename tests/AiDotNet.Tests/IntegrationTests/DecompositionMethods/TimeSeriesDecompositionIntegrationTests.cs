using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.Enums;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.DecompositionMethods;

/// <summary>
/// Integration tests for time series decomposition classes.
/// </summary>
public class TimeSeriesDecompositionIntegrationTests
{
    /// <summary>
    /// Creates a synthetic time series with trend + seasonal + noise.
    /// Length 48 with period 12 (monthly data, 4 years).
    /// </summary>
    private static Vector<double> CreateSeasonalTimeSeries()
    {
        int n = 48;
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            double trend = 10.0 + 0.5 * i;
            double seasonal = 5.0 * Math.Sin(2.0 * Math.PI * i / 12.0);
            double noise = 0.1 * ((i * 7 + 3) % 11 - 5); // deterministic pseudo-noise
            data[i] = trend + seasonal + noise;
        }

        return new Vector<double>(data);
    }

    /// <summary>
    /// Creates a simple monotone increasing time series for basic tests.
    /// </summary>
    private static Vector<double> CreateSimpleTimeSeries()
    {
        int n = 36;
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            data[i] = 5.0 + 0.3 * i + 2.0 * Math.Sin(2.0 * Math.PI * i / 12.0);
        }

        return new Vector<double>(data);
    }

    /// <summary>
    /// Creates a strictly positive time series for multiplicative decomposition.
    /// </summary>
    private static Vector<double> CreatePositiveTimeSeries()
    {
        int n = 48;
        var data = new double[n];
        for (int i = 0; i < n; i++)
        {
            double trend = 50.0 + 0.5 * i;
            double seasonal = 1.0 + 0.3 * Math.Sin(2.0 * Math.PI * i / 12.0);
            data[i] = trend * seasonal;
        }

        return new Vector<double>(data);
    }

    #region AdditiveDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task Additive_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new AdditiveDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task Additive_GetComponents_ReturnsMultipleComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new AdditiveDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count >= 2, $"Should have at least 2 components (trend+residual), got {components.Count}");
    }

    [Fact(Timeout = 120000)]
    public async Task Additive_TimeSeries_PreservedAfterDecomposition()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new AdditiveDecomposition<double>(ts);
        Assert.Equal(ts.Length, decomp.TimeSeries.Length);
    }

    #endregion

    private static object? InvokeAdditivePrivate(AdditiveDecomposition<double> decomposition, string name, params object[] args)
    {
        var method = typeof(AdditiveDecomposition<double>).GetMethod(
            name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(method);
        return method.Invoke(decomposition, args);
    }

    [Fact]
    public void Additive_STL_SeriesShorterThanOnePeriod_HasNoNaN()
    {
        // n <= period gives one-row subseries (window 0); they must pass through unsmoothed
        // instead of scaling the distance by infinity (0 * Infinity = NaN).
        var data = new double[10];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = 3.0 + 0.5 * i + ((i * 5) % 3);
        }

        var decomp = new AdditiveDecomposition<double>(new Vector<double>(data), AdditiveDecompositionAlgorithmType.STL);

        foreach (var type in new[] { DecompositionComponentType.Trend, DecompositionComponentType.Seasonal, DecompositionComponentType.Residual })
        {
            var component = decomp.GetComponentAsVector(type);
            Assert.Equal(data.Length, component.Length);
            for (int i = 0; i < component.Length; i++)
            {
                Assert.False(double.IsNaN(component[i]), $"{type}[{i}] is NaN");
            }
        }
    }

    [Fact]
    public void Additive_SubseriesSmoothing_WindowIsMeasuredInSubseriesRows()
    {
        // Five rows with period 12: window = 3 rows, half window 1.5, so a neighbouring row (distance
        // 1 row) has tri-cube weight (1 - 1/1.5)^3 = 1/27. Measuring the distance in time steps (12)
        // gave every neighbour weight 0 and returned the input unchanged.
        var decomp = new AdditiveDecomposition<double>(CreateSeasonalTimeSeries());
        var column = new Matrix<double>(5, 1);
        column[0, 0] = 0; column[1, 0] = 0; column[2, 0] = 27; column[3, 0] = 0; column[4, 0] = 0;
        var grid = AiDotNet.Tensors.LinearAlgebra.Tensor<double>.FromMatrix(column);

        var smoothed = Assert.IsType<AiDotNet.Tensors.LinearAlgebra.Tensor<double>>(
            InvokeAdditivePrivate(decomp, "SmoothSubseriesColumns", grid, 1));

        double w = 1.0 / 27.0;
        Assert.Equal(27.0 / (1 + 2 * w), smoothed[2, 0], 1e-10);
        Assert.Equal(27.0 * w / (1 + 2 * w), smoothed[1, 0], 1e-10);
        Assert.Equal(27.0 * w / (1 + 2 * w), smoothed[3, 0], 1e-10);
        Assert.Equal(0.0, smoothed[0, 0], 1e-10);
        Assert.Equal(0.0, smoothed[4, 0], 1e-10);
    }

    [Fact]
    public void Additive_LoessSmoothing_IsLocalLinear_ReproducesARampAtTheBoundaries()
    {
        // Local-linear LOESS reproduces a straight line exactly, including the truncated windows at
        // both ends; a weighted mean is pulled inwards there (biased up at index 0).
        var decomp = new AdditiveDecomposition<double>(CreateSeasonalTimeSeries());
        var ramp = new double[30];
        for (int i = 0; i < ramp.Length; i++)
        {
            ramp[i] = 2.0 + 0.5 * i;
        }

        var smoothed = Assert.IsType<Vector<double>>(
            InvokeAdditivePrivate(decomp, "LoessSmoothing", new Vector<double>(ramp), 23));

        for (int i = 0; i < ramp.Length; i++)
        {
            Assert.Equal(ramp[i], smoothed[i], 1e-9);
        }
    }

    #region MultiplicativeDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task Multiplicative_Construction_DoesNotThrow()
    {
        var ts = CreatePositiveTimeSeries();
        var decomp = new MultiplicativeDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task Multiplicative_GetComponents_ReturnsComponents()
    {
        var ts = CreatePositiveTimeSeries();
        var decomp = new MultiplicativeDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "Should have at least one component");
    }

    [Fact(Timeout = 120000)]
    public async Task Multiplicative_WithSeasonalPeriod_DoesNotThrow()
    {
        var ts = CreatePositiveTimeSeries();
        var decomp = new MultiplicativeDecomposition<double>(ts, seasonalPeriod: 12);
        Assert.NotNull(decomp);
    }

    #endregion

    #region STLTimeSeriesDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task STL_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var options = new STLDecompositionOptions<double>();
        var decomp = new STLTimeSeriesDecomposition<double>(ts, options);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task STL_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var options = new STLDecompositionOptions<double>();
        var decomp = new STLTimeSeriesDecomposition<double>(ts, options);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "STL should produce components");
    }

    [Theory]
    [InlineData(OutlierDetectionMethod.ZScore)]
    [InlineData(OutlierDetectionMethod.IQR)]
    public void STL_OutlierMask_IgnoresNonFiniteValues(OutlierDetectionMethod method)
    {
        // One NaN must not make the mean / standard deviation (or the IQR order statistics) NaN,
        // which silently disabled outlier detection for the whole series.
        var options = new STLDecompositionOptions<double> { OutlierDetectionMethod = method };
        var decomp = new STLTimeSeriesDecomposition<double>(CreateSeasonalTimeSeries(), options);

        var data = new double[21];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = 1.0 + 0.01 * i;
        }

        data[5] = 100.0;
        data[12] = double.NaN;
        var series = new Vector<double>(data);
        var backingField = typeof(TimeSeriesDecompositionBase<double>).GetField(
            "<TimeSeries>k__BackingField",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(backingField);
        backingField.SetValue(decomp, series);

        string maskMethod = method == OutlierDetectionMethod.ZScore ? "ZScoreOutlierMask" : "IqrOutlierMask";
        var mask = typeof(STLTimeSeriesDecomposition<double>).GetMethod(
            maskMethod, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(mask);

        var result = Assert.IsType<AiDotNet.Tensors.LinearAlgebra.Tensor<double>>(
            mask.Invoke(decomp, new object[] { AiDotNet.Tensors.LinearAlgebra.Tensor<double>.FromVector(series) }));

        Assert.Equal(1.0, result[5]);
        Assert.Equal(0.0, result[0]);
        Assert.Equal(0.0, result[20]);
    }

    #endregion

    #region SSADecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task SSA_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SSADecomposition<double>(ts, windowSize: 12, numberOfComponents: 3);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task SSA_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SSADecomposition<double>(ts, windowSize: 12, numberOfComponents: 3);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "SSA should produce components");
    }

    #endregion

    #region EMDDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task EMD_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new EMDDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task EMD_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new EMDDecomposition<double>(ts, maxImf: 5);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "EMD should produce components");
    }

    #endregion

    #region WaveletDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task Wavelet_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new WaveletDecomposition<double>(ts, levels: 3);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task Wavelet_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new WaveletDecomposition<double>(ts, levels: 3);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "Wavelet should produce components");
    }

    #endregion

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(20)]
    public void Wavelet_CircularFilterPair_MatchesDefinitionForAnyDilation(int dilation)
    {
        // first[t] = sum_l a[l] v[(t - l * dilation) mod N]; large dilations (span beyond N, or a
        // multiple of N) are filtered in bounded tap groups and must still match the definition.
        var decomp = new WaveletDecomposition<double>(CreateSeasonalTimeSeries(), levels: 3);
        var method = typeof(WaveletDecomposition<double>).GetMethod(
            "CircularFilterPair", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(method);

        var v = new[] { 1.0, -2.0, 3.5, 0.25, 4.0, -1.5, 2.0 };
        var a = new[] { 1.0, 2.0, 3.0, 4.0 };
        var b = new[] { 0.5, -1.0, 2.0, 0.25 };
        int n = v.Length;

        var result = method.Invoke(decomp, new object[]
        {
            new Vector<double>(v), new Vector<double>(a), new Vector<double>(b), dilation
        });
        var (first, second) = Assert.IsType<(Vector<double>, Vector<double>)>(result);

        for (int t = 0; t < n; t++)
        {
            double expectedFirst = 0;
            double expectedSecond = 0;
            for (int l = 0; l < a.Length; l++)
            {
                int index = (int)(((t - (long)l * dilation) % n + n) % n);
                expectedFirst += a[l] * v[index];
                expectedSecond += b[l] * v[index];
            }

            Assert.Equal(expectedFirst, first[t], 1e-10);
            Assert.Equal(expectedSecond, second[t], 1e-10);
        }
    }

    #region HodrickPrescottDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task HodrickPrescott_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task HodrickPrescott_GetComponents_ContainsTrend()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.ContainsKey(DecompositionComponentType.Trend),
            "HP filter should produce a Trend component");
    }

    [Fact(Timeout = 120000)]
    public async Task HodrickPrescott_TrendComponent_HasCorrectLength()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600);
        var components = decomp.GetComponents();

        Assert.True(components.ContainsKey(DecompositionComponentType.Trend),
            "HP filter must produce a Trend component");
        var trend = (Vector<double>)components[DecompositionComponentType.Trend];
        Assert.Equal(ts.Length, trend.Length);
    }

    #endregion

    #region BeveridgeNelsonDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task BeveridgeNelson_Construction_DoesNotThrow()
    {
        var ts = CreateSimpleTimeSeries();
        var decomp = new BeveridgeNelsonDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task BeveridgeNelson_GetComponents_ReturnsComponents()
    {
        var ts = CreateSimpleTimeSeries();
        var decomp = new BeveridgeNelsonDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "BN should produce components");
    }

    [Theory]
    [InlineData(new[] { 5.0 })]
    [InlineData(new[] { 1.0, 4.0 })]
    [InlineData(new[] { 1.0, 3.0, 5.0, 7.0, 9.0 })]
    [InlineData(new[] { 2.0, 2.0, 2.0, 2.0 })]
    public void BeveridgeNelson_ShortOrConstantDifferenceSeries_TrendIsSeriesWithoutNaN(double[] values)
    {
        // A series with at most one point has no differences, and constant differences give
        // gamma_0 = 0; both have no transitory part, so trend = series and cycle = 0 (never NaN).
        var ts = new Vector<double>(values);

        var decomp = new BeveridgeNelsonDecomposition<double>(ts);
        var trend = decomp.GetComponentAsVector(DecompositionComponentType.Trend);
        var cycle = decomp.GetComponentAsVector(DecompositionComponentType.Cycle);

        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(values[i], trend[i], 1e-12);
            Assert.Equal(0.0, cycle[i], 1e-12);
        }
    }

    #endregion

    #region SEATSDecomposition Tests

    [Fact(Timeout = 120000)]
    public async Task SEATS_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SEATSDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task SEATS_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SEATSDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "SEATS should produce components");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public void SEATS_CanonicalSeasonalFilter_NoSeasonalFrequencies_IsZeroNotNaN(int seasonalPeriod)
    {
        // With s < 2 the sum over seasonal frequencies j = 1 .. s - 1 is empty, so the filter is
        // zero; scaling it by 1 / (s - 1) gave 0 * Infinity = NaN for s = 1.
        var decomp = new SEATSDecomposition<double>(CreateSeasonalTimeSeries());
        var model = new AiDotNet.TimeSeries.SARIMAModel<double>(
            new SARIMAOptions<double> { SeasonalPeriod = seasonalPeriod });
        var design = typeof(SEATSDecomposition<double>).GetMethod(
            "DesignCanonicalSeasonalFilter",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(design);

        var filter = Assert.IsType<Vector<double>>(design.Invoke(decomp, new object[] { model }));

        Assert.True(filter.Length > 0);
        for (int i = 0; i < filter.Length; i++)
        {
            Assert.Equal(0.0, filter[i]);
        }
    }

    [Fact]
    public void SEATS_ApplyFilter_EmptyFilter_ReturnsZeroSignalOfSameLength()
    {
        // A Burman seasonal filter has length 0 when the seasonal period is 0; filtering with it
        // must give a zero component of the signal length instead of a Conv1D shape error.
        var decomp = new SEATSDecomposition<double>(CreateSeasonalTimeSeries());
        var apply = typeof(SEATSDecomposition<double>).GetMethod(
            "ApplyFilter",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        Assert.NotNull(apply);
        var signal = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        var filtered = Assert.IsType<Vector<double>>(apply.Invoke(decomp, new object[] { signal, new Vector<double>(0) }));

        Assert.Equal(4, filtered.Length);
        for (int i = 0; i < filtered.Length; i++)
        {
            Assert.Equal(0.0, filtered[i]);
        }
    }

    #endregion

    #region X11Decomposition Tests

    [Theory]
    [InlineData(0)]
    [InlineData(11)]
    public void X11_SeriesShorterThanOneSeasonalPeriod_ThrowsArgumentException(int length)
    {
        // Seasonal normalisation needs one full cycle; a partial cycle must be rejected up front.
        var ts = new Vector<double>(Enumerable.Range(0, length).Select(i => 1.0 + i).ToArray());

        var ex = Assert.Throws<ArgumentException>(() => new X11Decomposition<double>(ts, seasonalPeriod: 12));
        Assert.Equal("timeSeries", ex.ParamName);
    }

    [Fact(Timeout = 120000)]
    public async Task X11_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new X11Decomposition<double>(ts, seasonalPeriod: 12);
        Assert.NotNull(decomp);
    }

    [Fact(Timeout = 120000)]
    public async Task X11_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new X11Decomposition<double>(ts, seasonalPeriod: 12);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "X11 should produce components");
    }

    #endregion

    #region Cross-Decomposition Tests

    [Fact(Timeout = 120000)]
    public async Task AllDecompositions_PreserveOriginalTimeSeries()
    {
        var ts = CreateSeasonalTimeSeries();
        var posTs = CreatePositiveTimeSeries();

        var decompositions = new TimeSeriesDecompositionBase<double>[]
        {
            new AdditiveDecomposition<double>(ts),
            new MultiplicativeDecomposition<double>(posTs),
            new HodrickPrescottDecomposition<double>(ts),
            new WaveletDecomposition<double>(ts, levels: 2),
            new EMDDecomposition<double>(ts, maxImf: 3),
            new X11Decomposition<double>(ts, seasonalPeriod: 12),
        };

        foreach (var decomp in decompositions)
        {
            Assert.NotNull(decomp.TimeSeries);
            Assert.True(decomp.TimeSeries.Length > 0,
                $"{decomp.GetType().Name} should preserve time series");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AllDecompositions_ProduceNonEmptyComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var posTs = CreatePositiveTimeSeries();

        var decompositions = new TimeSeriesDecompositionBase<double>[]
        {
            new AdditiveDecomposition<double>(ts),
            new MultiplicativeDecomposition<double>(posTs),
            new HodrickPrescottDecomposition<double>(ts),
        };

        foreach (var decomp in decompositions)
        {
            var components = decomp.GetComponents();
            Assert.NotNull(components);
            Assert.True(components.Count > 0,
                $"{decomp.GetType().Name} should produce at least one component");
        }
    }

    #endregion
}
