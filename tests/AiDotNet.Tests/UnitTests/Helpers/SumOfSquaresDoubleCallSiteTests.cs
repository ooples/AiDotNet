using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.AnomalyDetection.TimeSeries;
using AiDotNet.Enums;
using AiDotNet.Finance.Evaluation;
using AiDotNet.Finance.Portfolio;
using AiDotNet.Finance.Risk;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// The features that took a sum of squares from <c>Engine.TensorSumOfSquares</c>, each on a double input of 16 or
/// more elements and against a value computed here with a plain loop. In AiDotNet.Tensors 0.150.0 that engine op is
/// wrong for such an input on an AVX2/FMA CPU (see <see cref="TensorSumOfSquaresDoubleTests"/>); the existing tests
/// of these features all used fewer than 16 values.
/// </summary>
public class SumOfSquaresDoubleCallSiteTests
{
    private const int Count = 40;

    // Different at even and odd indices, mixed sign: the defect cancels on a series whose two halves match.
    private static double[] Series(int n, int seed)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = 0.01 * Math.Sin(0.9 * i + seed) + 0.002 * ((i * 7 + seed) % 5) - 0.003;
        }

        return values;
    }

    private static double Mean(IReadOnlyList<double> v) => v.Sum() / v.Count;

    private static double SumSquaredDeviations(IReadOnlyList<double> v)
    {
        double mean = Mean(v), sum = 0;
        foreach (double x in v) sum += (x - mean) * (x - mean);
        return sum;
    }

    private static void AssertClose(double expected, double actual, double relativeTolerance = 1e-9)
    {
        Assert.True(
            Math.Abs(actual - expected) <= relativeTolerance * Math.Max(Math.Abs(expected), 1e-300),
            $"expected {expected:R}, got {actual:R}");
    }

    // ---- Finance ----

    [Fact]
    public void RiskRatios_Sharpe_UsesTheSampleStandardDeviation()
    {
        var returns = Series(Count, seed: 1);
        double sampleStd = Math.Sqrt(SumSquaredDeviations(returns) / (Count - 1));

        AssertClose(Mean(returns) / sampleStd * Math.Sqrt(252), RiskRatios<double>.Sharpe(returns));
    }

    [Fact]
    public void RiskRatios_Sortino_UsesTheDownsideDeviation()
    {
        var returns = Series(Count, seed: 2);
        double downside = 0;
        foreach (double r in returns)
        {
            if (r <= 0) downside += r * r;
        }

        AssertClose(Mean(returns) / Math.Sqrt(downside / Count) * Math.Sqrt(252), RiskRatios<double>.Sortino(returns));
    }

    [Fact]
    public void KellyCriterion_FromReturns_IsMeanOverPopulationVariance()
    {
        var returns = Series(Count, seed: 3);

        AssertClose(Mean(returns) / (SumSquaredDeviations(returns) / Count), KellyCriterion<double>.FromReturns(returns));
    }

    [Fact]
    public void InformationCoefficient_Pearson_MatchesTheDefinition()
    {
        var predicted = Series(Count, seed: 4);
        var noise = Series(Count, seed: 9);
        var realized = predicted.Select((p, i) => 0.5 * p + noise[i]).ToArray();

        double meanP = Mean(predicted), meanR = Mean(realized), covariance = 0;
        for (int i = 0; i < Count; i++) covariance += (predicted[i] - meanP) * (realized[i] - meanR);
        double expected = covariance / Math.Sqrt(SumSquaredDeviations(predicted) * SumSquaredDeviations(realized));

        AssertClose(expected, InformationCoefficient<double>.Pearson(predicted, realized));
    }

    [Fact]
    public void InformationCoefficient_InformationRatio_UsesTheSampleStandardDeviation()
    {
        var ics = Series(24, seed: 5);
        double std = Math.Sqrt(SumSquaredDeviations(ics) / (ics.Length - 1));

        var (meanIc, stdIc, icir) = InformationCoefficient<double>.InformationRatio(ics);

        AssertClose(Mean(ics), meanIc);
        AssertClose(std, stdIc);
        AssertClose(Mean(ics) / std * Math.Sqrt(ics.Length), icir);
    }

    [Fact]
    public void SharpeRatioPortfolioObjective_Moments_ReturnsThePopulationVolatility()
    {
        var returns = Series(Count, seed: 6);

        var (mean, volatility) = new SharpeRatioPortfolioObjective<double>().Moments(new Vector<double>(returns));

        AssertClose(Mean(returns), mean);
        AssertClose(Math.Sqrt(SumSquaredDeviations(returns) / Count), volatility);
    }

    [Fact]
    public void BootstrapConfidenceInterval_SharpeStatistic_IsMeanOverSampleStandardDeviation()
    {
        var sample = Series(Count, seed: 7);
        double std = Math.Sqrt(SumSquaredDeviations(sample) / (Count - 1));

        AssertClose(Mean(sample) / std, BootstrapConfidenceInterval<double>.SharpeStatistic(new Vector<double>(sample)));
    }

    // ---- Optimizers ----

    [Fact]
    public void SparseEmbeddingOptimizerHelpers_SumOfSquares_MatchesALoop()
    {
        // The parameter and update norms of Adafactor, LAMB and LARS come from this.
        var values = Series(24, seed: 8);
        double all = 0, firstTwenty = 0;
        for (int i = 0; i < values.Length; i++)
        {
            all += values[i] * values[i];
            if (i < 20) firstTwenty += values[i] * values[i];
        }

        AssertClose(all, SparseEmbeddingOptimizerHelpers.SumOfSquares(new Tensor<double>((double[])values.Clone(), new[] { 24 })));
        AssertClose(firstTwenty, SparseEmbeddingOptimizerHelpers.SumOfSquares(new Tensor<double>((double[])values.Clone(), new[] { 24 }), 20));
    }

    [Fact]
    public void Optimizer_R2OnlyDataSetStats_MatchTheDefinition()
    {
        // The R^2 every optimizer reports for the data sets it does not score in full (FitDetector reads it).
        var x = new Matrix<double>(Count, 1);
        var y = new Vector<double>(Count);
        var wobble = Series(Count, seed: 10);
        for (int i = 0; i < Count; i++)
        {
            x[i, 0] = i / 10.0;
            y[i] = 2.0 * x[i, 0] + 1.0 + 40.0 * wobble[i];
        }

        var model = new MultipleRegression<double>();
        model.Train(x, y);
        var predicted = model.Predict(x);

        double meanY = 0;
        for (int i = 0; i < Count; i++) meanY += y[i];
        meanY /= Count;
        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < Count; i++)
        {
            ssRes += (y[i] - predicted[i]) * (y[i] - predicted[i]);
            ssTot += (y[i] - meanY) * (y[i] - meanY);
        }

        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null);
        var method = typeof(OptimizerBase<double, Matrix<double>, Vector<double>>).GetMethod(
            "CalculateR2OnlyStatsForOptimizer", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);
        var stats = (DataSetStats<double, Matrix<double>, Vector<double>>)method!.Invoke(
            optimizer, new object[] { model, x, y, PredictionType.Regression })!;

        Assert.InRange(1.0 - ssRes / ssTot, 0.5, 0.999);   // a real fit with a real residual
        AssertClose(1.0 - ssRes / ssTot, stats.PredictionStats.R2);
    }

    // ---- Neural networks ----

    [Fact]
    public void SyntheticTabularGenerator_ScaledL2Norm_IsTheEuclideanNorm()
    {
        // The norm behind the generators' gradient clipping and column standard deviations.
        var values = Series(Count, seed: 11);
        double sumSquares = 0;
        foreach (double v in values) sumSquares += v * v;

        double norm = SyntheticTabularGeneratorBase<double>.ScaledL2Norm(new Tensor<double>((double[])values.Clone(), new[] { Count }));

        AssertClose(Math.Sqrt(sumSquares), norm);
    }

    [Fact]
    public void EmbeddingLayer_AuxiliaryLoss_IsHalfTheMeanSquaredWeight()
    {
        var layer = new EmbeddingLayer<double>(8, 5) { UseAuxiliaryLoss = true, AuxiliaryLossWeight = 1.0 };
        var weights = layer.GetParameters();
        Assert.Equal(40, weights.Length);
        double sumSquares = 0;
        for (int i = 0; i < weights.Length; i++) sumSquares += weights[i] * weights[i];
        Assert.True(sumSquares > 0);

        AssertClose(sumSquares / (2.0 * weights.Length), layer.ComputeAuxiliaryLoss());
    }

    // ---- Anomaly detection ----

    [Fact]
    public void SeasonalHybridEsd_FallbackResidualDeviation_IsTheRootMeanSquare()
    {
        // Season 4, flat at 0 except the first phase. More than half the residuals are exactly 0, so the MAD is 0 and
        // the detector falls back to the root-mean-square residual: trend 1, phase means {3, -1, -1, -1}, residuals
        // {-3, -2, -1, 0, 1, 2, 3, 4, -1, -3} at the first phase and 0 elsewhere, sum of squares 54.
        double[] firstPhase = { 1, 2, 3, 4, 5, 6, 7, 8, 3, 1 };
        double[] residuals = { -3, -2, -1, 0, 1, 2, 3, 4, -1, -3 };
        var series = new Matrix<double>(Count, 1);
        for (int k = 0; k < firstPhase.Length; k++) series[4 * k, 0] = firstPhase[k];
        double deviation = Math.Sqrt(54.0 / Count);

        var detector = new SeasonalHybridESDDetector<double>(seasonLength: 4);
        detector.Fit(series);
        var scores = detector.ScoreAnomalies(series);

        Assert.Equal(Count, scores.Length);
        for (int i = 0; i < Count; i++)
        {
            double expected = i % 4 == 0 ? Math.Abs(residuals[i / 4]) / deviation : 0.0;
            Assert.Equal(expected, scores[i], 9);
        }
    }
}
