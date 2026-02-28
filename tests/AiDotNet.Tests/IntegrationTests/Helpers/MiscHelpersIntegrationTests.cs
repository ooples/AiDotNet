using System.Text.RegularExpressions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for miscellaneous helper classes:
/// EnumHelper, VectorHelper, RegexHelper, TensorCopyHelper,
/// TextProcessingHelper, DataAggregationHelper, TimeSeriesHelper,
/// WeightFunctionHelper.
/// </summary>
public class MiscHelpersIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region EnumHelper

    [Fact]
    public void EnumHelper_GetEnumValues_ReturnsAllValues()
    {
        var values = EnumHelper.GetEnumValues<ActivationFunction>();
        Assert.NotEmpty(values);
        Assert.Contains(ActivationFunction.ReLU, values);
        Assert.Contains(ActivationFunction.Sigmoid, values);
        Assert.Contains(ActivationFunction.Tanh, values);
    }

    [Fact]
    public void EnumHelper_GetEnumValues_WithIgnore_ExcludesValue()
    {
        var values = EnumHelper.GetEnumValues<ActivationFunction>("ReLU");
        Assert.DoesNotContain(ActivationFunction.ReLU, values);
        Assert.Contains(ActivationFunction.Sigmoid, values);
    }

    [Fact]
    public void EnumHelper_GetEnumValues_NullIgnore_ReturnsAll()
    {
        var values = EnumHelper.GetEnumValues<ActivationFunction>(null);
        Assert.NotEmpty(values);
    }

    #endregion

    #region VectorHelper

    [Fact]
    public void VectorHelper_CreateVector_CorrectSize()
    {
        var v = VectorHelper.CreateVector<double>(5);
        Assert.Equal(5, v.Length);
    }

    [Fact]
    public void VectorHelper_CreateVector_DefaultValues()
    {
        var v = VectorHelper.CreateVector<double>(3);
        Assert.Equal(0.0, v[0], Tolerance);
        Assert.Equal(0.0, v[1], Tolerance);
        Assert.Equal(0.0, v[2], Tolerance);
    }

    [Fact]
    public void VectorHelper_CreateVector_ZeroSize()
    {
        var v = VectorHelper.CreateVector<double>(0);
        Assert.Equal(0, v.Length);
    }

    #endregion

    #region RegexHelper

    [Fact]
    public void RegexHelper_DefaultTimeout_IsOneSecond()
    {
        Assert.Equal(TimeSpan.FromSeconds(1), RegexHelper.DefaultTimeout);
    }

    [Fact]
    public void RegexHelper_FastTimeout_Is100ms()
    {
        Assert.Equal(TimeSpan.FromMilliseconds(100), RegexHelper.FastTimeout);
    }

    [Fact]
    public void RegexHelper_Create_ReturnsRegex()
    {
        var regex = RegexHelper.Create(@"\d+");
        Assert.NotNull(regex);
        Assert.True(regex.IsMatch("abc123"));
    }

    [Fact]
    public void RegexHelper_IsMatch_FindsPattern()
    {
        Assert.True(RegexHelper.IsMatch("hello123", @"\d+"));
        Assert.False(RegexHelper.IsMatch("hello", @"\d+"));
    }

    [Fact]
    public void RegexHelper_Match_ReturnsMatch()
    {
        var match = RegexHelper.Match("test123end", @"\d+");
        Assert.True(match.Success);
        Assert.Equal("123", match.Value);
    }

    [Fact]
    public void RegexHelper_Matches_ReturnsAllMatches()
    {
        var matches = RegexHelper.Matches("a1b2c3", @"\d");
        Assert.Equal(3, matches.Count);
    }

    [Fact]
    public void RegexHelper_Replace_ReplacesPattern()
    {
        var result = RegexHelper.Replace("hello 123 world 456", @"\d+", "NUM");
        Assert.Equal("hello NUM world NUM", result);
    }

    [Fact]
    public void RegexHelper_Replace_WithEvaluator()
    {
        var result = RegexHelper.Replace("a1b2", @"\d", m => (int.Parse(m.Value) * 10).ToString());
        Assert.Equal("a10b20", result);
    }

    [Fact]
    public void RegexHelper_Split_SplitsOnPattern()
    {
        var parts = RegexHelper.Split("one,two,three", ",");
        Assert.Equal(3, parts.Length);
        Assert.Equal("one", parts[0]);
        Assert.Equal("three", parts[2]);
    }

    [Fact]
    public void RegexHelper_Escape_EscapesSpecialChars()
    {
        var escaped = RegexHelper.Escape("(hello)");
        Assert.Contains(@"\(", escaped);
        Assert.Contains(@"\)", escaped);
    }

    [Fact]
    public void RegexHelper_Create_WithOptions()
    {
        var regex = RegexHelper.Create("hello", RegexOptions.IgnoreCase);
        Assert.True(regex.IsMatch("HELLO"));
    }

    #endregion

    #region TensorCopyHelper

    [Fact]
    public void TensorCopyHelper_CopySample_1D()
    {
        var source = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 10, 20, 30 }));
        var dest = new Tensor<double>(new[] { 3 });

        TensorCopyHelper.CopySample(source, dest, 1, 2);
        Assert.Equal(20.0, dest[2], Tolerance);
    }

    [Fact]
    public void TensorCopyHelper_CopySample_2D()
    {
        var source = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var dest = new Tensor<double>(new[] { 2, 3 });

        TensorCopyHelper.CopySample(source, dest, 0, 1);
        // Source sample 0 = [1, 2, 3] copied to dest sample 1
        Assert.Equal(1.0, dest[1, 0], Tolerance);
        Assert.Equal(2.0, dest[1, 1], Tolerance);
        Assert.Equal(3.0, dest[1, 2], Tolerance);
    }

    [Fact]
    public void TensorCopyHelper_CopySample_DifferentRanks_Throws()
    {
        var source = new Tensor<double>(new[] { 3 });
        var dest = new Tensor<double>(new[] { 3, 2 });

        Assert.Throws<ArgumentException>(() =>
            TensorCopyHelper.CopySample(source, dest, 0, 0));
    }

    [Fact]
    public void TensorCopyHelper_CopySample_SourceOutOfRange_Throws()
    {
        var source = new Tensor<double>(new[] { 2, 3 });
        var dest = new Tensor<double>(new[] { 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorCopyHelper.CopySample(source, dest, 5, 0));
    }

    [Fact]
    public void TensorCopyHelper_CopySample_DestOutOfRange_Throws()
    {
        var source = new Tensor<double>(new[] { 2, 3 });
        var dest = new Tensor<double>(new[] { 2, 3 });

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TensorCopyHelper.CopySample(source, dest, 0, 5));
    }

    [Fact]
    public void TensorCopyHelper_CopySample_MismatchedShapes_Throws()
    {
        var source = new Tensor<double>(new[] { 2, 3 });
        var dest = new Tensor<double>(new[] { 2, 4 }); // different second dim

        Assert.Throws<ArgumentException>(() =>
            TensorCopyHelper.CopySample(source, dest, 0, 0));
    }

    #endregion

    #region DataAggregationHelper

    [Fact]
    public void DataAggregationHelper_AggregateVectors()
    {
        var v1 = new Vector<double>(new double[] { 1, 2, 3 });
        var v2 = new Vector<double>(new double[] { 4, 5, 6 });
        var list = new List<Vector<double>> { v1, v2 };

        var result = DataAggregationHelper.Aggregate<double, Vector<double>>(list, "test");
        Assert.Equal(6, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(4.0, result[3], Tolerance);
    }

    [Fact]
    public void DataAggregationHelper_AggregateMatrices()
    {
        var m1 = new Matrix<double>(2, 3);
        m1[0, 0] = 1.0;
        var m2 = new Matrix<double>(1, 3);
        m2[0, 0] = 7.0;
        var list = new List<Matrix<double>> { m1, m2 };

        var result = DataAggregationHelper.Aggregate<double, Matrix<double>>(list, "test");
        Assert.Equal(3, result.Rows);
        Assert.Equal(3, result.Columns);
    }

    [Fact]
    public void DataAggregationHelper_AggregateTensors()
    {
        var t1 = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6 }));
        var t2 = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new double[] { 7, 8, 9 }));
        var list = new List<Tensor<double>> { t1, t2 };

        var result = DataAggregationHelper.Aggregate<double, Tensor<double>>(list, "test");
        Assert.Equal(new[] { 3, 3 }, result.Shape);
    }

    [Fact]
    public void DataAggregationHelper_SingleItem_ReturnsSame()
    {
        var v = new Vector<double>(new double[] { 1, 2, 3 });
        var list = new List<Vector<double>> { v };

        var result = DataAggregationHelper.Aggregate<double, Vector<double>>(list, "test");
        Assert.Equal(3, result.Length);
    }

    [Fact]
    public void DataAggregationHelper_EmptyList_Throws()
    {
        var list = new List<Vector<double>>();
        Assert.Throws<InvalidOperationException>(() =>
            DataAggregationHelper.Aggregate<double, Vector<double>>(list, "test"));
    }

    [Fact]
    public void DataAggregationHelper_MismatchedMatrixColumns_Throws()
    {
        var m1 = new Matrix<double>(2, 3);
        var m2 = new Matrix<double>(2, 4); // different columns
        var list = new List<Matrix<double>> { m1, m2 };

        Assert.Throws<ArgumentException>(() =>
            DataAggregationHelper.Aggregate<double, Matrix<double>>(list, "test"));
    }

    #endregion

    #region TimeSeriesHelper

    [Fact]
    public void TimeSeriesHelper_DifferenceSeries_Order1()
    {
        var y = new Vector<double>(new double[] { 10.0, 13.0, 15.0, 19.0 });
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 1);
        Assert.Equal(3, result.Length);
        Assert.Equal(3.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(4.0, result[2], Tolerance);
    }

    [Fact]
    public void TimeSeriesHelper_DifferenceSeries_Order2()
    {
        var y = new Vector<double>(new double[] { 10.0, 13.0, 15.0, 19.0 });
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 2);
        // First diff: [3, 2, 4]
        // Second diff: [-1, 2]
        Assert.Equal(2, result.Length);
        Assert.Equal(-1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
    }

    [Fact]
    public void TimeSeriesHelper_DifferenceSeries_Order0_Unchanged()
    {
        var y = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 0);
        Assert.Equal(3, result.Length);
        Assert.Equal(10.0, result[0], Tolerance);
    }

    [Fact]
    public void TimeSeriesHelper_CalculateAutoCorrelation_Lag0Like()
    {
        // With known data, autocorrelation at lag 0 normalized by itself should be 1
        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var acf = TimeSeriesHelper<double>.CalculateMultipleAutoCorrelation(y, 2);
        Assert.Equal(1.0, acf[0], Tolerance); // lag 0 always 1
    }

    [Fact]
    public void TimeSeriesHelper_CalculateMultipleAutoCorrelation_Length()
    {
        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var acf = TimeSeriesHelper<double>.CalculateMultipleAutoCorrelation(y, 3);
        Assert.Equal(4, acf.Length); // maxLag + 1
    }

    [Fact]
    public void TimeSeriesHelper_EstimateMACoefficients_ReturnsCorrectLength()
    {
        var residuals = new Vector<double>(new double[] { 0.1, -0.2, 0.3, -0.1, 0.05, -0.15 });
        var maCoeffs = TimeSeriesHelper<double>.EstimateMACoefficients(residuals, 2);
        Assert.Equal(2, maCoeffs.Length);
    }

    [Fact]
    public void TimeSeriesHelper_CalculateARResiduals_CorrectLength()
    {
        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        var arCoeffs = new Vector<double>(new double[] { 0.5, 0.3 }); // AR(2)
        var residuals = TimeSeriesHelper<double>.CalculateARResiduals(y, arCoeffs);
        Assert.Equal(4, residuals.Length); // n - p = 6 - 2
    }

    #endregion

    #region WeightFunctionHelper

    [Fact]
    public void WeightFunctionHelper_Huber_SmallResiduals_WeightOne()
    {
        var residuals = new Vector<double>(new double[] { 0.1, -0.1, 0.05 });
        var weights = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Huber, 1.345);
        // All residuals < tuning constant, so weights should be 1
        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(1.0, weights[i], Tolerance);
        }
    }

    [Fact]
    public void WeightFunctionHelper_Huber_LargeResiduals_ReducedWeight()
    {
        var residuals = new Vector<double>(new double[] { 5.0, -10.0 });
        var weights = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Huber, 1.345);
        // Large residuals get reduced weight
        Assert.True(weights[0] < 1.0);
        Assert.True(weights[1] < 1.0);
        Assert.True(weights[0] > 0.0);
    }

    [Fact]
    public void WeightFunctionHelper_Bisquare_SmallResiduals_NonZero()
    {
        var residuals = new Vector<double>(new double[] { 0.5, -0.3 });
        var weights = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Bisquare, 4.685);
        Assert.True(weights[0] > 0.0);
        Assert.True(weights[1] > 0.0);
    }

    [Fact]
    public void WeightFunctionHelper_Bisquare_LargeResiduals_Zero()
    {
        var residuals = new Vector<double>(new double[] { 100.0 });
        var weights = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Bisquare, 4.685);
        Assert.Equal(0.0, weights[0], Tolerance);
    }

    [Fact]
    public void WeightFunctionHelper_Andrews_SmallResiduals_NonZero()
    {
        var residuals = new Vector<double>(new double[] { 0.1 });
        var weights = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Andrews, 1.339);
        Assert.True(weights[0] > 0.0);
    }

    [Fact]
    public void WeightFunctionHelper_Andrews_LargeResiduals_Zero()
    {
        var residuals = new Vector<double>(new double[] { 100.0 });
        var weights = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Andrews, 1.339);
        Assert.Equal(0.0, weights[0], Tolerance);
    }

    [Fact]
    public void WeightFunctionHelper_InvalidFunction_Throws()
    {
        var residuals = new Vector<double>(new double[] { 1.0 });
        Assert.Throws<ArgumentException>(() =>
            WeightFunctionHelper<double>.CalculateWeights(residuals, (WeightFunction)999, 1.0));
    }

    #endregion
}
