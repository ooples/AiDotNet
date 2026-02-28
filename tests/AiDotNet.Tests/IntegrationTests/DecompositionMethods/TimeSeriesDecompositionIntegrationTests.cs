using AiDotNet.DecompositionMethods.TimeSeriesDecomposition;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

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

    [Fact]
    public void Additive_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new AdditiveDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void Additive_GetComponents_ReturnsMultipleComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new AdditiveDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count >= 2, $"Should have at least 2 components (trend+residual), got {components.Count}");
    }

    [Fact]
    public void Additive_TimeSeries_PreservedAfterDecomposition()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new AdditiveDecomposition<double>(ts);
        Assert.Equal(ts.Length, decomp.TimeSeries.Length);
    }

    #endregion

    #region MultiplicativeDecomposition Tests

    [Fact]
    public void Multiplicative_Construction_DoesNotThrow()
    {
        var ts = CreatePositiveTimeSeries();
        var decomp = new MultiplicativeDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void Multiplicative_GetComponents_ReturnsComponents()
    {
        var ts = CreatePositiveTimeSeries();
        var decomp = new MultiplicativeDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "Should have at least one component");
    }

    [Fact]
    public void Multiplicative_WithSeasonalPeriod_DoesNotThrow()
    {
        var ts = CreatePositiveTimeSeries();
        var decomp = new MultiplicativeDecomposition<double>(ts, seasonalPeriod: 12);
        Assert.NotNull(decomp);
    }

    #endregion

    #region STLTimeSeriesDecomposition Tests

    [Fact]
    public void STL_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var options = new STLDecompositionOptions<double>();
        var decomp = new STLTimeSeriesDecomposition<double>(ts, options);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void STL_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var options = new STLDecompositionOptions<double>();
        var decomp = new STLTimeSeriesDecomposition<double>(ts, options);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "STL should produce components");
    }

    #endregion

    #region SSADecomposition Tests

    [Fact]
    public void SSA_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SSADecomposition<double>(ts, windowSize: 12, numberOfComponents: 3);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void SSA_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SSADecomposition<double>(ts, windowSize: 12, numberOfComponents: 3);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "SSA should produce components");
    }

    #endregion

    #region EMDDecomposition Tests

    [Fact]
    public void EMD_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new EMDDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void EMD_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new EMDDecomposition<double>(ts, maxImf: 5);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "EMD should produce components");
    }

    #endregion

    #region WaveletDecomposition Tests

    [Fact]
    public void Wavelet_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new WaveletDecomposition<double>(ts, levels: 3);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void Wavelet_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new WaveletDecomposition<double>(ts, levels: 3);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "Wavelet should produce components");
    }

    #endregion

    #region HodrickPrescottDecomposition Tests

    [Fact]
    public void HodrickPrescott_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void HodrickPrescott_GetComponents_ContainsTrend()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new HodrickPrescottDecomposition<double>(ts, lambda: 1600);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.ContainsKey(DecompositionComponentType.Trend),
            "HP filter should produce a Trend component");
    }

    [Fact]
    public void HodrickPrescott_TrendComponent_HasCorrectLength()
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

    [Fact]
    public void BeveridgeNelson_Construction_DoesNotThrow()
    {
        var ts = CreateSimpleTimeSeries();
        var decomp = new BeveridgeNelsonDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void BeveridgeNelson_GetComponents_ReturnsComponents()
    {
        var ts = CreateSimpleTimeSeries();
        var decomp = new BeveridgeNelsonDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "BN should produce components");
    }

    #endregion

    #region SEATSDecomposition Tests

    [Fact]
    public void SEATS_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SEATSDecomposition<double>(ts);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void SEATS_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new SEATSDecomposition<double>(ts);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "SEATS should produce components");
    }

    #endregion

    #region X11Decomposition Tests

    [Fact]
    public void X11_Construction_DoesNotThrow()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new X11Decomposition<double>(ts, seasonalPeriod: 12);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void X11_GetComponents_ReturnsComponents()
    {
        var ts = CreateSeasonalTimeSeries();
        var decomp = new X11Decomposition<double>(ts, seasonalPeriod: 12);
        var components = decomp.GetComponents();
        Assert.NotNull(components);
        Assert.True(components.Count > 0, "X11 should produce components");
    }

    #endregion

    #region Cross-Decomposition Tests

    [Fact]
    public void AllDecompositions_PreserveOriginalTimeSeries()
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

    [Fact]
    public void AllDecompositions_ProduceNonEmptyComponents()
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
