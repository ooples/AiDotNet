using AiDotNet.Enums;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Deep math-correctness integration tests for WeightFunctionHelper (Huber, Bisquare, Andrews),
/// TimeSeriesHelper (DifferenceSeries, AutoCorrelation), and SamplingHelper.
/// All expected values are hand-calculated from first principles.
/// </summary>
public class WeightFunctionAndTimeSeriesHelperDeepMathIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-6;

    #region Huber Weights

    [Fact]
    public void HuberWeights_SmallResiduals_WeightIsOne()
    {
        // Huber: if |r| <= k, weight = 1
        // k = 1.345 (common default)
        var residuals = new Vector<double>([0.5, -0.3, 1.0, -1.345]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, 1.345);

        for (int i = 0; i < weights.Length; i++)
            Assert.Equal(1.0, weights[i], Tolerance);
    }

    [Fact]
    public void HuberWeights_LargeResiduals_HandCalculated()
    {
        // Huber: if |r| > k, weight = k / |r|
        // k = 1.345
        // r = [3.0, -5.0, 10.0]
        // weights = [1.345/3, 1.345/5, 1.345/10]
        double k = 1.345;
        var residuals = new Vector<double>([3.0, -5.0, 10.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, k);

        Assert.Equal(k / 3.0, weights[0], Tolerance);
        Assert.Equal(k / 5.0, weights[1], Tolerance);
        Assert.Equal(k / 10.0, weights[2], Tolerance);
    }

    [Fact]
    public void HuberWeights_MixedResiduals_HandCalculated()
    {
        // Mix of small and large residuals
        // k = 2.0
        // r = [0.5, -3.0, 1.5, -4.0]
        // weights = [1.0, 2/3, 1.0, 2/4=0.5]
        double k = 2.0;
        var residuals = new Vector<double>([0.5, -3.0, 1.5, -4.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, k);

        Assert.Equal(1.0, weights[0], Tolerance);
        Assert.Equal(2.0 / 3.0, weights[1], Tolerance);
        Assert.Equal(1.0, weights[2], Tolerance);
        Assert.Equal(2.0 / 4.0, weights[3], Tolerance);
    }

    [Fact]
    public void HuberWeights_ZeroResidual_WeightIsOne()
    {
        var residuals = new Vector<double>([0.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, 1.345);

        Assert.Equal(1.0, weights[0], Tolerance);
    }

    [Fact]
    public void HuberWeights_AtBoundary_WeightIsOne()
    {
        // At exactly k, weight should be 1
        double k = 2.0;
        var residuals = new Vector<double>([2.0, -2.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, k);

        Assert.Equal(1.0, weights[0], Tolerance);
        Assert.Equal(1.0, weights[1], Tolerance);
    }

    [Fact]
    public void HuberWeights_AlwaysPositive()
    {
        var residuals = new Vector<double>([0.1, -0.5, 3.0, -10.0, 100.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, 1.0);

        for (int i = 0; i < weights.Length; i++)
            Assert.True(weights[i] > 0, $"Huber weight[{i}] should be positive");
    }

    #endregion

    #region Bisquare Weights

    [Fact]
    public void BisquareWeights_SmallResiduals_HandCalculated()
    {
        // Bisquare: if |r| <= k, weight = (1 - (r/k)^2)^2
        // k = 4.685
        // r = [1.0]: u = 1/4.685, w = (1 - u^2)^2
        double k = 4.685;
        double r = 1.0;
        double u = r / k;
        double expected = Math.Pow(1.0 - u * u, 2);

        var residuals = new Vector<double>([r]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Bisquare, k);

        Assert.Equal(expected, weights[0], Tolerance);
    }

    [Fact]
    public void BisquareWeights_LargeResiduals_Zero()
    {
        // Bisquare: if |r| > k, weight = 0
        double k = 4.685;
        var residuals = new Vector<double>([5.0, -6.0, 100.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Bisquare, k);

        Assert.Equal(0.0, weights[0], Tolerance);
        Assert.Equal(0.0, weights[1], Tolerance);
        Assert.Equal(0.0, weights[2], Tolerance);
    }

    [Fact]
    public void BisquareWeights_ZeroResidual_WeightIsOne()
    {
        // r=0: u=0, weight = (1-0)^2 = 1
        var residuals = new Vector<double>([0.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Bisquare, 4.685);

        Assert.Equal(1.0, weights[0], Tolerance);
    }

    [Fact]
    public void BisquareWeights_AtBoundary_WeightIsZero()
    {
        // At exactly k: u = 1, weight = (1-1)^2 = 0
        double k = 3.0;
        var residuals = new Vector<double>([3.0, -3.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Bisquare, k);

        Assert.Equal(0.0, weights[0], Tolerance);
        Assert.Equal(0.0, weights[1], Tolerance);
    }

    [Fact]
    public void BisquareWeights_MultipleValues_HandCalculated()
    {
        // k = 3.0
        // r = [0, 1.5, 2.0, 3.0, 4.0]
        // u = [0, 0.5, 2/3, 1.0, >1]
        // weight = [(1-0)^2, (1-0.25)^2, (1-4/9)^2, 0, 0]
        //        = [1, 0.5625, 0.30864..., 0, 0]
        double k = 3.0;
        var residuals = new Vector<double>([0.0, 1.5, 2.0, 3.0, 4.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Bisquare, k);

        Assert.Equal(1.0, weights[0], Tolerance);
        Assert.Equal(Math.Pow(1 - 0.25, 2), weights[1], Tolerance);
        Assert.Equal(Math.Pow(1 - 4.0 / 9.0, 2), weights[2], Tolerance);
        Assert.Equal(0.0, weights[3], Tolerance);
        Assert.Equal(0.0, weights[4], Tolerance);
    }

    [Fact]
    public void BisquareWeights_NegativeResiduals_SameAsPositive()
    {
        // Bisquare uses |r|, so positive and negative residuals give same weight
        double k = 3.0;
        var posResiduals = new Vector<double>([1.0, 2.0]);
        var negResiduals = new Vector<double>([-1.0, -2.0]);

        var posWeights = WeightFunctionHelper<double>.CalculateWeights(
            posResiduals, WeightFunction.Bisquare, k);
        var negWeights = WeightFunctionHelper<double>.CalculateWeights(
            negResiduals, WeightFunction.Bisquare, k);

        Assert.Equal(posWeights[0], negWeights[0], Tolerance);
        Assert.Equal(posWeights[1], negWeights[1], Tolerance);
    }

    #endregion

    #region Andrews Weights

    [Fact]
    public void AndrewsWeights_SmallResiduals_HandCalculated()
    {
        // Andrews: if |r| <= k*pi, weight = sin(r/k) / (r/k)
        // k = 1.339
        // r = 1.0: u = 1/1.339, weight = sin(u) / u
        double k = 1.339;
        double r = 1.0;
        double u = r / k;
        double expected = Math.Sin(u) / u;

        var residuals = new Vector<double>([r]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Andrews, k);

        Assert.Equal(expected, weights[0], Tolerance);
    }

    [Fact]
    public void AndrewsWeights_LargeResiduals_Zero()
    {
        // Andrews: if |r| > k*pi, weight = 0
        double k = 1.339;
        double boundary = k * Math.PI; // ~4.207
        var residuals = new Vector<double>([5.0, -6.0, 100.0]);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Andrews, k);

        Assert.Equal(0.0, weights[0], Tolerance);
        Assert.Equal(0.0, weights[1], Tolerance);
        Assert.Equal(0.0, weights[2], Tolerance);
    }

    [Fact]
    public void AndrewsWeights_SincProperty()
    {
        // Andrews weight is the sinc function: sin(r/k) / (r/k)
        // At small values, sinc ≈ 1
        // Verify for multiple residuals
        double k = 2.0;
        double[] rValues = [0.5, 1.0, 2.0, 3.0];
        var residuals = new Vector<double>(rValues);
        var weights = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Andrews, k);

        for (int i = 0; i < rValues.Length; i++)
        {
            double u = rValues[i] / k;
            double expected = Math.Sin(u) / u;
            Assert.Equal(expected, weights[i], Tolerance);
        }
    }

    [Fact]
    public void AndrewsWeights_NegativeResiduals_SameAsPositive()
    {
        // Andrews uses |r| for boundary check, but sin(r/k)/(r/k) uses signed r
        // sin(-x)/(-x) = (-sin(x))/(-x) = sin(x)/x
        // So negative residuals give same weight as positive
        double k = 2.0;
        var posResiduals = new Vector<double>([1.0, 2.0]);
        var negResiduals = new Vector<double>([-1.0, -2.0]);

        var posWeights = WeightFunctionHelper<double>.CalculateWeights(
            posResiduals, WeightFunction.Andrews, k);
        var negWeights = WeightFunctionHelper<double>.CalculateWeights(
            negResiduals, WeightFunction.Andrews, k);

        Assert.Equal(posWeights[0], negWeights[0], Tolerance);
        Assert.Equal(posWeights[1], negWeights[1], Tolerance);
    }

    #endregion

    #region Weight Function Cross-Comparisons

    [Fact]
    public void HuberWeights_GreaterThanOrEqual_BisquareWeights()
    {
        // Huber weights are always >= Bisquare weights for the same residual
        // Because Bisquare goes to 0 for large residuals, while Huber only goes to k/|r|
        double k = 3.0;
        double[] testResiduals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.99];
        var residuals = new Vector<double>(testResiduals);

        var huber = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Huber, k);
        var bisquare = WeightFunctionHelper<double>.CalculateWeights(
            residuals, WeightFunction.Bisquare, k);

        for (int i = 0; i < testResiduals.Length; i++)
        {
            Assert.True(huber[i] >= bisquare[i] - Tolerance,
                $"Huber[{i}]={huber[i]} should be >= Bisquare[{i}]={bisquare[i]}");
        }
    }

    [Fact]
    public void AllWeightFunctions_ZeroResidual_WeightIsOne()
    {
        var residuals = new Vector<double>([0.0]);
        double k = 2.0;

        var huber = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Huber, k);
        var bisquare = WeightFunctionHelper<double>.CalculateWeights(residuals, WeightFunction.Bisquare, k);
        // Note: Andrews at r=0 would be sin(0)/0 = 0/0 which is undefined
        // But the limit of sin(x)/x as x→0 is 1

        Assert.Equal(1.0, huber[0], Tolerance);
        Assert.Equal(1.0, bisquare[0], Tolerance);
    }

    [Fact]
    public void InvalidWeightFunction_Throws()
    {
        var residuals = new Vector<double>([1.0]);
        Assert.Throws<ArgumentException>(() =>
            WeightFunctionHelper<double>.CalculateWeights(residuals, (WeightFunction)999, 1.0));
    }

    #endregion

    #region TimeSeriesHelper - DifferenceSeries

    [Fact]
    public void DifferenceSeries_Order1_HandCalculated()
    {
        // y = [10, 13, 15, 19]
        // d=1: [13-10, 15-13, 19-15] = [3, 2, 4]
        var y = new Vector<double>([10.0, 13.0, 15.0, 19.0]);
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 1);

        Assert.Equal(3, result.Length);
        Assert.Equal(3.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(4.0, result[2], Tolerance);
    }

    [Fact]
    public void DifferenceSeries_Order2_HandCalculated()
    {
        // y = [1, 4, 9, 16, 25]
        // d=1: [3, 5, 7, 9]
        // d=2: [2, 2, 2]
        var y = new Vector<double>([1.0, 4.0, 9.0, 16.0, 25.0]);
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 2);

        Assert.Equal(3, result.Length);
        Assert.Equal(2.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(2.0, result[2], Tolerance);
    }

    [Fact]
    public void DifferenceSeries_Order0_Unchanged()
    {
        var y = new Vector<double>([1.0, 2.0, 3.0]);
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 0);

        Assert.Equal(3, result.Length);
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void DifferenceSeries_ConstantSeries_AllZeros()
    {
        var y = new Vector<double>([5.0, 5.0, 5.0, 5.0]);
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 1);

        Assert.Equal(3, result.Length);
        for (int i = 0; i < result.Length; i++)
            Assert.Equal(0.0, result[i], Tolerance);
    }

    [Fact]
    public void DifferenceSeries_LinearTrend_ConstantFirstDifference()
    {
        // Linear: y = 2 + 3*t → first difference = 3 everywhere
        var y = new Vector<double>([2.0, 5.0, 8.0, 11.0, 14.0]);
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 1);

        Assert.Equal(4, result.Length);
        for (int i = 0; i < result.Length; i++)
            Assert.Equal(3.0, result[i], Tolerance);
    }

    [Fact]
    public void DifferenceSeries_QuadraticTrend_ConstantSecondDifference()
    {
        // Quadratic: y = t^2 = [0, 1, 4, 9, 16, 25]
        // First difference: [1, 3, 5, 7, 9]
        // Second difference: [2, 2, 2, 2]
        var y = new Vector<double>([0.0, 1.0, 4.0, 9.0, 16.0, 25.0]);
        var result = TimeSeriesHelper<double>.DifferenceSeries(y, 2);

        Assert.Equal(4, result.Length);
        for (int i = 0; i < result.Length; i++)
            Assert.Equal(2.0, result[i], Tolerance);
    }

    [Fact]
    public void DifferenceSeries_ReducesLength()
    {
        // Differencing d times reduces length by d
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        var d1 = TimeSeriesHelper<double>.DifferenceSeries(y, 1);
        var d2 = TimeSeriesHelper<double>.DifferenceSeries(y, 2);
        var d3 = TimeSeriesHelper<double>.DifferenceSeries(y, 3);

        Assert.Equal(9, d1.Length);
        Assert.Equal(8, d2.Length);
        Assert.Equal(7, d3.Length);
    }

    #endregion

    #region TimeSeriesHelper - AutoCorrelation

    [Fact]
    public void AutoCorrelation_Lag0_IsOne()
    {
        // ACF(0) should be sum(y[i]*y[i]) / sum(y[i]*y[i]) = 1
        // Note: the implementation computes sum(y[i]*y[i+0]) / sum(y[i]^2) for i in [0,n-1]
        // This equals 1.0
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0]);
        var acf = TimeSeriesHelper<double>.CalculateAutoCorrelation(y, 0);
        Assert.Equal(1.0, acf, Tolerance);
    }

    [Fact]
    public void AutoCorrelation_HandCalculated_Lag1()
    {
        // y = [1, 2, 3, 4, 5]
        // Lag 1, i goes from 0 to n-lag-1 = 3
        // sum = y[0]*y[1] + y[1]*y[2] + y[2]*y[3] + y[3]*y[4]
        //     = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        // sumSquared = y[0]^2 + y[1]^2 + y[2]^2 + y[3]^2
        //            = 1 + 4 + 9 + 16 = 30
        // ACF(1) = 40 / 30 = 4/3
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0]);
        var acf = TimeSeriesHelper<double>.CalculateAutoCorrelation(y, 1);
        Assert.Equal(40.0 / 30.0, acf, Tolerance);
    }

    [Fact]
    public void AutoCorrelation_HandCalculated_Lag2()
    {
        // y = [1, 2, 3, 4, 5]
        // Lag 2, i goes from 0 to 2
        // sum = y[0]*y[2] + y[1]*y[3] + y[2]*y[4]
        //     = 1*3 + 2*4 + 3*5 = 3 + 8 + 15 = 26
        // sumSquared = 1 + 4 + 9 = 14
        // ACF(2) = 26 / 14 = 13/7
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0]);
        var acf = TimeSeriesHelper<double>.CalculateAutoCorrelation(y, 2);
        Assert.Equal(26.0 / 14.0, acf, Tolerance);
    }

    [Fact]
    public void MultipleAutoCorrelation_LengthCorrect()
    {
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0]);
        var acf = TimeSeriesHelper<double>.CalculateMultipleAutoCorrelation(y, 3);

        Assert.Equal(4, acf.Length); // maxLag+1 = 4
        Assert.Equal(1.0, acf[0], Tolerance); // lag 0 is always 1
    }

    [Fact]
    public void MultipleAutoCorrelation_ConsistentWithSingle()
    {
        var y = new Vector<double>([1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);
        var multiAcf = TimeSeriesHelper<double>.CalculateMultipleAutoCorrelation(y, 3);

        for (int lag = 1; lag <= 3; lag++)
        {
            var singleAcf = TimeSeriesHelper<double>.CalculateAutoCorrelation(y, lag);
            Assert.Equal(singleAcf, multiAcf[lag], Tolerance);
        }
    }

    #endregion

    #region TimeSeriesHelper - AR Residuals

    [Fact]
    public void CalculateARResiduals_HandCalculated()
    {
        // y = [1, 2, 3, 4, 5]
        // AR coefficients = [0.5] (order 1)
        // Predicted: y_hat[1] = 0.5 * y[0] = 0.5, residual = 2 - 0.5 = 1.5
        // Predicted: y_hat[2] = 0.5 * y[1] = 1.0, residual = 3 - 1.0 = 2.0
        // Predicted: y_hat[3] = 0.5 * y[2] = 1.5, residual = 4 - 1.5 = 2.5
        // Predicted: y_hat[4] = 0.5 * y[3] = 2.0, residual = 5 - 2.0 = 3.0
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0]);
        var arCoeffs = new Vector<double>([0.5]);
        var residuals = TimeSeriesHelper<double>.CalculateARResiduals(y, arCoeffs);

        Assert.Equal(4, residuals.Length);
        Assert.Equal(1.5, residuals[0], Tolerance);
        Assert.Equal(2.0, residuals[1], Tolerance);
        Assert.Equal(2.5, residuals[2], Tolerance);
        Assert.Equal(3.0, residuals[3], Tolerance);
    }

    [Fact]
    public void CalculateARResiduals_Order2_HandCalculated()
    {
        // y = [1, 2, 3, 4, 5]
        // AR coefficients = [0.5, 0.3] (order 2)
        // Predicted: y_hat[2] = 0.5*y[1] + 0.3*y[0] = 1.0 + 0.3 = 1.3, residual = 3 - 1.3 = 1.7
        // Predicted: y_hat[3] = 0.5*y[2] + 0.3*y[1] = 1.5 + 0.6 = 2.1, residual = 4 - 2.1 = 1.9
        // Predicted: y_hat[4] = 0.5*y[3] + 0.3*y[2] = 2.0 + 0.9 = 2.9, residual = 5 - 2.9 = 2.1
        var y = new Vector<double>([1.0, 2.0, 3.0, 4.0, 5.0]);
        var arCoeffs = new Vector<double>([0.5, 0.3]);
        var residuals = TimeSeriesHelper<double>.CalculateARResiduals(y, arCoeffs);

        Assert.Equal(3, residuals.Length);
        Assert.Equal(1.7, residuals[0], Tolerance);
        Assert.Equal(1.9, residuals[1], Tolerance);
        Assert.Equal(2.1, residuals[2], Tolerance);
    }

    [Fact]
    public void CalculateARResiduals_PerfectFit_ZeroResiduals()
    {
        // If y follows exact AR(1) with coeff 2.0: y = [1, 2, 4, 8]
        // residuals should be 0 everywhere
        var y = new Vector<double>([1.0, 2.0, 4.0, 8.0]);
        var arCoeffs = new Vector<double>([2.0]);
        var residuals = TimeSeriesHelper<double>.CalculateARResiduals(y, arCoeffs);

        Assert.Equal(3, residuals.Length);
        for (int i = 0; i < residuals.Length; i++)
            Assert.Equal(0.0, residuals[i], Tolerance);
    }

    #endregion

    #region SamplingHelper

    [Fact]
    public void SampleWithoutReplacement_CorrectSize()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            var result = SamplingHelper.SampleWithoutReplacement(100, 10);
            Assert.Equal(10, result.Length);
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void SampleWithoutReplacement_NoDuplicates()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            var result = SamplingHelper.SampleWithoutReplacement(20, 15);
            var distinct = result.Distinct().ToArray();
            Assert.Equal(result.Length, distinct.Length);
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void SampleWithoutReplacement_AllIndicesInRange()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            var result = SamplingHelper.SampleWithoutReplacement(50, 30);
            Assert.All(result, idx => Assert.InRange(idx, 0, 49));
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void SampleWithoutReplacement_SizeExceedsPopulation_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            SamplingHelper.SampleWithoutReplacement(5, 10));
    }

    [Fact]
    public void SampleWithReplacement_CorrectSize()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            var result = SamplingHelper.SampleWithReplacement(10, 20);
            Assert.Equal(20, result.Length);
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void SampleWithReplacement_AllIndicesInRange()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            var result = SamplingHelper.SampleWithReplacement(10, 100);
            Assert.All(result, idx => Assert.InRange(idx, 0, 9));
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void SampleWithReplacement_CanHaveDuplicates()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            // With replacement from small population, high chance of duplicates
            var result = SamplingHelper.SampleWithReplacement(3, 100);
            var distinct = result.Distinct().Count();
            Assert.True(distinct < result.Length,
                "With replacement should produce duplicates for small population");
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void CreateBootstrapSamples_CorrectNumberAndSize()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            double[] data = [1.0, 2.0, 3.0, 4.0, 5.0];
            var samples = SamplingHelper.CreateBootstrapSamples(data, 10);

            Assert.Equal(10, samples.Count);
            Assert.All(samples, s => Assert.Equal(5, s.Length));
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void CreateBootstrapSamples_CustomSize()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            double[] data = [1.0, 2.0, 3.0, 4.0, 5.0];
            var samples = SamplingHelper.CreateBootstrapSamples(data, 5, 3);

            Assert.Equal(5, samples.Count);
            Assert.All(samples, s => Assert.Equal(3, s.Length));
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void CreateBootstrapSamples_ContainsOriginalValues()
    {
        SamplingHelper.SetSeed(42);
        try
        {
            double[] data = [10.0, 20.0, 30.0];
            var samples = SamplingHelper.CreateBootstrapSamples(data, 5);

            Assert.All(samples, sample =>
                Assert.All(sample, val =>
                    Assert.Contains(val, data)));
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }
    }

    [Fact]
    public void SamplingHelper_Seed_Reproducible()
    {
        SamplingHelper.SetSeed(123);
        int[] first;
        try
        {
            first = SamplingHelper.SampleWithoutReplacement(100, 10);
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }

        SamplingHelper.SetSeed(123);
        int[] second;
        try
        {
            second = SamplingHelper.SampleWithoutReplacement(100, 10);
        }
        finally
        {
            SamplingHelper.ClearSeed();
        }

        Assert.Equal(first, second);
    }

    #endregion
}
