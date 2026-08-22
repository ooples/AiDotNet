#nullable disable
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Regression;

/// <summary>
/// Verifies that quantile regression estimates conditional QUANTILES, solving the linear program of
/// Koenker and Bassett (1978).
/// </summary>
/// <remarks>
/// CRITICAL: these tests exist because the previous implementation returned the ordinary
/// least-squares fit — the conditional MEAN — for any problem with at least one feature, which is
/// precisely what quantile regression must not do. If one fails, FIX THE MODEL.
/// </remarks>
public class QuantileRegressionLinearProgramTests
{
    private static Matrix<double> Column(double[] values)
    {
        var matrix = new Matrix<double>(values.Length, 1);
        for (int i = 0; i < values.Length; i++) matrix[i, 0] = values[i];
        return matrix;
    }

    private static QuantileRegression<double> Fit(
        Matrix<double> x, Vector<double> y, double quantile)
    {
        var model = new QuantileRegression<double>(
            new QuantileRegressionOptions<double> { Quantile = quantile });
        model.Train(x, y);
        return model;
    }

    /// <summary>
    /// The defining property: the fitted line for quantile τ leaves approximately a τ fraction of
    /// the observations below it. A least-squares fit has no such property, so this test fails
    /// outright against a mean estimator.
    /// </summary>
    [Theory]
    [InlineData(0.25)]
    [InlineData(0.50)]
    [InlineData(0.75)]
    public void Train_FractionOfPointsBelowTheFit_MatchesTheQuantile(double quantile)
    {
        // A deterministic spread of residuals around the line y = 2x + 1.
        int count = 40;
        var xs = new double[count];
        var ys = new double[count];
        for (int i = 0; i < count; i++)
        {
            xs[i] = i * 0.25;
            double residual = ((i % 8) - 3.5) * 2.0;   // symmetric, spread across the range
            ys[i] = 2.0 * xs[i] + 1.0 + residual;
        }

        var x = Column(xs);
        var y = Vector<double>.FromArray(ys);
        var model = Fit(x, y, quantile);

        var predictions = model.Predict(x);
        int below = 0;
        for (int i = 0; i < count; i++)
        {
            if (ys[i] < predictions[i]) below++;
        }

        double fractionBelow = (double)below / count;
        Assert.True(
            Math.Abs(fractionBelow - quantile) <= 0.15,
            $"Quantile {quantile}: {fractionBelow:P0} of points fell below the fit, expected about {quantile:P0}.");
    }

    /// <summary>
    /// Higher quantiles must sit above lower ones. A mean estimator returns the same line for every
    /// quantile, so this ordering cannot hold for it.
    /// </summary>
    [Fact]
    public void Train_HigherQuantiles_ProduceHigherFits()
    {
        int count = 40;
        var xs = new double[count];
        var ys = new double[count];
        for (int i = 0; i < count; i++)
        {
            xs[i] = i * 0.25;
            ys[i] = 2.0 * xs[i] + 1.0 + ((i % 8) - 3.5) * 2.0;
        }

        var x = Column(xs);
        var y = Vector<double>.FromArray(ys);

        var low = Fit(x, y, 0.1).Predict(x);
        var middle = Fit(x, y, 0.5).Predict(x);
        var high = Fit(x, y, 0.9).Predict(x);

        double lowMean = 0, middleMean = 0, highMean = 0;
        for (int i = 0; i < count; i++)
        {
            lowMean += low[i];
            middleMean += middle[i];
            highMean += high[i];
        }

        Assert.True(lowMean < middleMean, $"The 0.1 fit ({lowMean}) should sit below the median fit ({middleMean}).");
        Assert.True(middleMean < highMean, $"The median fit ({middleMean}) should sit below the 0.9 fit ({highMean}).");
    }

    /// <summary>
    /// The decisive test against the old behaviour. With strongly right-skewed residuals the
    /// conditional median is far below the conditional mean, so a median fit and a least-squares
    /// fit give visibly different intercepts.
    /// </summary>
    [Fact]
    public void Train_MedianFit_DiffersFromTheLeastSquaresFit_OnSkewedResiduals()
    {
        // Residuals: mostly zero, with a few enormous positive outliers. The median residual is 0;
        // the mean residual is large and positive.
        int count = 30;
        var xs = new double[count];
        var ys = new double[count];
        for (int i = 0; i < count; i++)
        {
            xs[i] = i;
            double residual = i % 10 == 0 ? 200.0 : 0.0;
            ys[i] = 3.0 * xs[i] + 5.0 + residual;
        }

        var x = Column(xs);
        var y = Vector<double>.FromArray(ys);

        var medianFit = Fit(x, y, 0.5);
        var predictions = medianFit.Predict(x);

        // The median fit should track the clean line y = 3x + 5, essentially ignoring the outliers.
        // A least-squares fit is dragged upward by them by roughly 200/10 = 20 on average.
        int onTheCleanLine = 0;
        for (int i = 0; i < count; i++)
        {
            if (i % 10 == 0) continue;                       // skip the outliers themselves
            if (Math.Abs(predictions[i] - (3.0 * xs[i] + 5.0)) < 1.0) onTheCleanLine++;
        }

        Assert.True(
            onTheCleanLine >= 24,
            $"Only {onTheCleanLine} of 27 clean points were tracked within 1.0; the median fit is " +
            "being dragged by the outliers, which is least-squares behaviour, not quantile behaviour.");
    }

    /// <summary>
    /// Quantile regression interpolates exactly: at the optimum of the linear program the fitted
    /// line passes through as many data points as there are parameters. With a perfectly linear
    /// dataset the fit must be exact for every quantile.
    /// </summary>
    [Theory]
    [InlineData(0.1)]
    [InlineData(0.5)]
    [InlineData(0.9)]
    public void Train_PerfectlyLinearData_RecoversTheExactLine(double quantile)
    {
        int count = 12;
        var xs = new double[count];
        var ys = new double[count];
        for (int i = 0; i < count; i++)
        {
            xs[i] = i;
            ys[i] = 4.0 * xs[i] - 7.0;
        }

        var model = Fit(Column(xs), Vector<double>.FromArray(ys), quantile);
        var predictions = model.Predict(Column(xs));

        for (int i = 0; i < count; i++)
        {
            Assert.Equal(ys[i], predictions[i], 6);
        }
    }

    /// <summary>
    /// Two features must both be recovered, confirming the linear program is built with the right
    /// column layout rather than working only in the single-feature case.
    /// </summary>
    [Fact]
    public void Train_MultipleFeatures_RecoversBothCoefficients()
    {
        int count = 15;
        var x = new Matrix<double>(count, 2);
        var ys = new double[count];
        for (int i = 0; i < count; i++)
        {
            x[i, 0] = i;
            x[i, 1] = (i % 5) * 2.0;
            ys[i] = 3.0 * x[i, 0] - 2.0 * x[i, 1] + 4.0;
        }

        var model = Fit(x, Vector<double>.FromArray(ys), 0.5);
        var predictions = model.Predict(x);

        for (int i = 0; i < count; i++)
        {
            Assert.Equal(ys[i], predictions[i], 5);
        }
    }
}
