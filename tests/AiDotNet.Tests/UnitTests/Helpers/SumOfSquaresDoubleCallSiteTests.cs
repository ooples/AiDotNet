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
            "CalculateR2OnlyStatsForOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("OptimizerBase.CalculateR2OnlyStatsForOptimizer was not found.");
        var stats = Assert.IsType<DataSetStats<double, Matrix<double>, Vector<double>>>(
            method.Invoke(optimizer, new object[] { model, x, y, PredictionType.Regression }));

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

    // ---- Training ----

    [Fact]
    public void FusedTrainingSession_AnyGradientNonZero_SeesASparseDoubleGradientAtOddIndices()
    {
        // A sparse gradient takes the reduction path, not the host scan. Its non-zero elements sit at odd flat indices
        // only, which the faulty double kernel of AiDotNet.Tensors 0.150.0 drops.
        var rows = new List<int>();
        var columns = new List<int>();
        var values = new List<double>();
        for (int r = 0; r < 4; r++)
        {
            for (int c = 1; c < 8; c += 2)
            {
                rows.Add(r);
                columns.Add(c);
                values.Add(0.25 + r + c);
            }
        }

        var gradient = new SparseTensor<double>(4, 8, rows.ToArray(), columns.ToArray(), values.ToArray());
        var gradients = new Dictionary<Tensor<double>, Tensor<double>> { [new Tensor<double>(new[] { 4, 8 })] = gradient };
        var method = typeof(AiDotNet.Training.FusedTrainingSession<double>).GetMethod(
            "AnyGradientNonZero", BindingFlags.Static | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("FusedTrainingSession.AnyGradientNonZero was not found.");

        Assert.True(Assert.IsType<bool>(method.Invoke(null, new object[] { gradients })));
    }

#if NET5_0_OR_GREATER
    // ---- Half: the sum of squares passes Half.MaxValue (65504) while every statistic built on it fits ----

    private const double HalfTolerance = 1e-2;

    // 40 values of 50..350: a sum of squared deviations near 2e5 and a standard deviation near 70.
    private static double[] LargeSeries(int n, int seed)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = (double)(Half)(200.0 + 150.0 * Math.Sin(0.9 * i + seed));
        }

        return values;
    }

    private static Half[] ToHalf(IEnumerable<double> values) => values.Select(v => (Half)v).ToArray();

    [Fact]
    public void RiskRatios_Sharpe_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var returns = LargeSeries(Count, seed: 1);
        Assert.True(SumSquaredDeviations(returns) > (double)Half.MaxValue);
        double expected = Mean(returns) / Math.Sqrt(SumSquaredDeviations(returns) / (Count - 1)) * Math.Sqrt(252);

        AssertClose(expected, (double)RiskRatios<Half>.Sharpe(ToHalf(returns)), HalfTolerance);
    }

    [Fact]
    public void RiskRatios_Sortino_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var returns = LargeSeries(Count, seed: 2).Select(r => (double)(Half)(r - 250.0)).ToArray();
        double downside = 0;
        foreach (double r in returns)
        {
            if (r <= 0) downside += r * r;
        }

        Assert.True(downside > (double)Half.MaxValue);
        double expected = Mean(returns) / Math.Sqrt(downside / Count) * Math.Sqrt(252);

        AssertClose(expected, (double)RiskRatios<Half>.Sortino(ToHalf(returns)), HalfTolerance);
    }

    [Fact]
    public void KellyCriterion_FromReturns_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var returns = LargeSeries(Count, seed: 3);

        AssertClose(Mean(returns) / (SumSquaredDeviations(returns) / Count),
            (double)KellyCriterion<Half>.FromReturns(ToHalf(returns)), HalfTolerance);
    }

    [Fact]
    public void SharpeRatioPortfolioObjective_Moments_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var returns = LargeSeries(Count, seed: 6);

        var (_, volatility) = new SharpeRatioPortfolioObjective<Half>().Moments(new Vector<Half>(ToHalf(returns)));

        AssertClose(Math.Sqrt(SumSquaredDeviations(returns) / Count), (double)volatility, HalfTolerance);
    }

    [Fact]
    public void BootstrapConfidenceInterval_SharpeStatistic_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var sample = LargeSeries(Count, seed: 7);
        double std = Math.Sqrt(SumSquaredDeviations(sample) / (Count - 1));

        AssertClose(Mean(sample) / std,
            (double)BootstrapConfidenceInterval<Half>.SharpeStatistic(new Vector<Half>(ToHalf(sample))), HalfTolerance);
    }

    [Fact]
    public void InformationCoefficient_InformationRatio_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var ics = LargeSeries(Count, seed: 5);
        double std = Math.Sqrt(SumSquaredDeviations(ics) / (Count - 1));

        var (_, stdIc, _) = InformationCoefficient<Half>.InformationRatio(ToHalf(ics));

        AssertClose(std, (double)stdIc, HalfTolerance);
    }

    [Fact]
    public void InformationCoefficient_Pearson_Half_SurvivesSumsOfSquaresPastHalfMaxValue()
    {
        var predicted = LargeSeries(Count, seed: 4);
        var noise = LargeSeries(Count, seed: 9);
        var realized = predicted.Select((p, i) => (double)(Half)(0.5 * p + 0.5 * noise[i])).ToArray();

        double meanP = Mean(predicted), meanR = Mean(realized), covariance = 0;
        for (int i = 0; i < Count; i++) covariance += (predicted[i] - meanP) * (realized[i] - meanR);
        double expected = covariance / Math.Sqrt(SumSquaredDeviations(predicted) * SumSquaredDeviations(realized));

        AssertClose(expected, (double)InformationCoefficient<Half>.Pearson(ToHalf(predicted), ToHalf(realized)), HalfTolerance);
    }

    [Fact]
    public void InformationCoefficient_Spearman_Half_OfTwoHundredValues_IsTheRankCorrelation()
    {
        // The ranks of 200 values reach 200 and their squared deviations sum to about 6.7e5, past Half.MaxValue.
        // y = x on all but the first ten, which are reversed among themselves.
        const int n = 200;
        var x = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        var y = (double[])x.Clone();
        Array.Reverse(y, 0, 10);

        double sumD2 = 0;
        for (int i = 0; i < n; i++) sumD2 += (x[i] - y[i]) * (x[i] - y[i]);
        double expected = 1.0 - 6.0 * sumD2 / (n * ((double)n * n - 1));   // no ties

        AssertClose(expected, (double)InformationCoefficient<Half>.Spearman(ToHalf(x), ToHalf(y)), HalfTolerance);
    }

    [Fact]
    public void EmbeddingLayer_AuxiliaryLoss_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        var layer = new EmbeddingLayer<Half>(8, 5) { UseAuxiliaryLoss = true, AuxiliaryLossWeight = (Half)1.0 };
        var weights = new Half[40];
        double sumSquares = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (Half)(100.0 + i);
            sumSquares += (double)weights[i] * (double)weights[i];
        }

        layer.SetParameters(new Vector<Half>(weights));
        Assert.True(sumSquares > (double)Half.MaxValue);

        AssertClose(sumSquares / (2.0 * weights.Length), (double)layer.ComputeAuxiliaryLoss(), HalfTolerance);
    }

    [Fact]
    public void SeasonalHybridEsd_FallbackResidualDeviation_Half_SurvivesASumOfSquaresPastHalfMaxValue()
    {
        // The double test's series scaled by 100: residual sum of squares 5.4e5, past Half.MaxValue.
        double[] firstPhase = { 1, 2, 3, 4, 5, 6, 7, 8, 3, 1 };
        double[] residuals = { -3, -2, -1, 0, 1, 2, 3, 4, -1, -3 };
        var series = new Matrix<Half>(Count, 1);
        for (int k = 0; k < firstPhase.Length; k++) series[4 * k, 0] = (Half)(100.0 * firstPhase[k]);
        double deviation = Math.Sqrt(54.0 * 1e4 / Count);

        var detector = new SeasonalHybridESDDetector<Half>(seasonLength: 4);
        detector.Fit(series);
        var scores = detector.ScoreAnomalies(series);

        Assert.Equal(Count, scores.Length);
        for (int i = 0; i < Count; i++)
        {
            double expected = i % 4 == 0 ? Math.Abs(100.0 * residuals[i / 4]) / deviation : 0.0;
            Assert.True(Math.Abs((double)scores[i] - expected) <= HalfTolerance * Math.Max(expected, 1.0),
                $"score {i}: expected {expected:R}, got {(double)scores[i]:R}");
        }
    }
#endif
}
