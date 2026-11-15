using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.AnomalyDetection.TimeSeries;
using AiDotNet.Audio.Enhancement;
using AiDotNet.Enums;
using AiDotNet.Finance.Evaluation;
using AiDotNet.Finance.Forecasting.Foundation;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Finance.Portfolio;
using AiDotNet.Finance.Risk;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Optimizers;
using AiDotNet.Regression;
using Moq;
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

    // ---- Half model sites ----
    //
    // The model sites below are private members of networks whose constructors build full layer stacks. They are
    // reached on an uninitialized instance whose NumOps field is set by hand: every member under test reads only
    // NumOps and the Engine property (AiDotNetEngine.Current), so nothing else of the instance is touched.

    private static TModel UninitializedHalfNetwork<TModel>() where TModel : NeuralNetworkBase<Half>
    {
        var model = (TModel)RuntimeHelpers.GetUninitializedObject(typeof(TModel));
        GC.SuppressFinalize(model);
        var numOps = typeof(NeuralNetworkBase<Half>).GetField("NumOps", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("NeuralNetworkBase.NumOps was not found.");
        numOps.SetValue(model, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>());
        return model;
    }

    private static object? InvokePrivate(object target, string name, params object[] arguments)
    {
        var method = target.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException($"{target.GetType().Name}.{name} was not found.");
        return method.Invoke(target, arguments);
    }

    // Errors of about 100 at 40 points: each squared error (up to 1.44e4) fits Half, their sum (about 4e5) does not.
    private static (Tensor<Half> Predictions, Tensor<Half> Actuals, double Mse, double Mae) LargeErrors(int n)
    {
        var predictions = new Half[n];
        var actuals = new Half[n];
        double sumSquared = 0, sumAbsolute = 0;
        for (int i = 0; i < n; i++)
        {
            actuals[i] = (Half)(50.0 + i);
            predictions[i] = (Half)((double)actuals[i] + 100.0 + 20.0 * Math.Sin(0.9 * i));
            double error = (double)predictions[i] - (double)actuals[i];
            sumSquared += error * error;
            sumAbsolute += Math.Abs(error);
        }

        Assert.True(sumSquared > (double)Half.MaxValue);
        return (new Tensor<Half>(predictions, new[] { n }), new Tensor<Half>(actuals, new[] { n }), sumSquared / n, sumAbsolute / n);
    }

    [Fact]
    public void FinancialModelBase_Evaluate_Half_ReportsAFiniteMseWhereTheSumPassesHalfMaxValue()
    {
        // TimesNet.Evaluate is the plain FinancialModelBase error path: MeanSquaredAndAbsoluteError, then RMSE.
        var (predictions, actuals, mse, mae) = LargeErrors(Count);
        var model = UninitializedHalfNetwork<TimesNet<Half>>();

        var metrics = model.Evaluate(predictions, actuals);

        AssertClose(mse, (double)metrics["MSE"], HalfTolerance);
        AssertClose(mae, (double)metrics["MAE"], HalfTolerance);
        AssertClose(Math.Sqrt(mse), (double)metrics["RMSE"], HalfTolerance);
    }

    [Fact]
    public void MultiTaskEvaluationHarness_EvaluateForecasting_Half_ReportsAFiniteMse()
    {
        var (predictions, actuals, mse, mae) = LargeErrors(Count);
        var model = new Mock<ITimeSeriesFoundationModel<Half>>();
        model.Setup(m => m.Forecast(It.IsAny<Tensor<Half>>(), It.IsAny<double[]>())).Returns(predictions);

        var metrics = new MultiTaskEvaluationHarness<Half>().EvaluateForecasting(
            model.Object, new[] { (new Tensor<Half>(new[] { Count }), actuals) });

        AssertClose(mse, (double)metrics["MSE"], HalfTolerance);
        AssertClose(mae, (double)metrics["MAE"], HalfTolerance);
        AssertClose(Math.Sqrt(mse), (double)metrics["RMSE"], HalfTolerance);
    }

    [Fact]
    public void Optimizer_R2OnlyDataSetStats_Half_IsFiniteWhereBothSumsPassHalfMaxValue()
    {
        // Residuals up to 80 and a spread near 150 over 40 rows: both sums of squares pass Half.MaxValue.
        var x = new Matrix<Half>(Count, 1);
        var y = new Half[Count];
        var predicted = new Half[Count];
        for (int i = 0; i < Count; i++)
        {
            x[i, 0] = (Half)i;
            y[i] = (Half)(200.0 + 150.0 * Math.Sin(0.9 * i));
            predicted[i] = (Half)((double)y[i] + 80.0 * Math.Cos(1.3 * i));
        }

        double meanY = y.Average(v => (double)v), ssRes = 0, ssTot = 0;
        for (int i = 0; i < Count; i++)
        {
            ssRes += ((double)y[i] - (double)predicted[i]) * ((double)y[i] - (double)predicted[i]);
            ssTot += ((double)y[i] - meanY) * ((double)y[i] - meanY);
        }

        Assert.True(ssRes > (double)Half.MaxValue && ssTot > (double)Half.MaxValue);

        var model = new Mock<IFullModel<Half, Matrix<Half>, Vector<Half>>>();
        model.Setup(m => m.Predict(It.IsAny<Matrix<Half>>())).Returns(new Vector<Half>(predicted));
        var optimizer = new AdamOptimizer<Half, Matrix<Half>, Vector<Half>>(null);
        var method = typeof(OptimizerBase<Half, Matrix<Half>, Vector<Half>>).GetMethod(
            "CalculateR2OnlyStatsForOptimizer", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("OptimizerBase.CalculateR2OnlyStatsForOptimizer was not found.");
        var stats = Assert.IsType<DataSetStats<Half, Matrix<Half>, Vector<Half>>>(
            method.Invoke(optimizer, new object[] { model.Object, x, new Vector<Half>(y), PredictionType.Regression }));

        AssertClose(1.0 - ssRes / ssTot, (double)stats.PredictionStats.R2, HalfTolerance);
    }

    [Fact]
    public void TFGridNet_Rms_Half_IsFiniteWhereTheSumOfSquaresPassesHalfMaxValue()
    {
        // A spectrogram-sized input of magnitudes near 50: sum of squares about 1e7, RMS about 50.
        const int n = 4000;
        var values = Enumerable.Range(0, n).Select(i => (Half)(50.0 + 10.0 * Math.Sin(0.37 * i))).ToArray();
        double expected = Math.Sqrt(values.Sum(v => (double)v * (double)v) / n);
        var model = UninitializedHalfNetwork<TFGridNet<Half>>();

        double rms = Assert.IsType<double>(InvokePrivate(model, "ComputeRms", new Tensor<Half>(values, new[] { n })));

        AssertClose(expected, rms, HalfTolerance);
    }

    [Fact]
    public void MGTSD_InstanceStats_Half_GiveAFiniteStd()
    {
        var series = LargeSeries(Count, seed: 12);
        double expectedStd = Math.Sqrt(SumSquaredDeviations(series) / Count + 1e-5);
        var model = UninitializedHalfNetwork<MGTSD<Half>>();

        var (mean, std) = Assert.IsType<ValueTuple<Half, Half>>(
            InvokePrivate(model, "ComputeInstanceStats", new Tensor<Half>(ToHalf(series), new[] { Count })));

        AssertClose(Mean(series), (double)mean, HalfTolerance);
        AssertClose(expectedStd, (double)std, HalfTolerance);
    }

    [Fact]
    public void NonStationaryTransformer_Stationarization_Half_NormalizesWithAFiniteStd()
    {
        var series = LargeSeries(Count, seed: 13);
        var model = UninitializedHalfNetwork<NonStationaryTransformer<Half>>();

        var normalized = Assert.IsAssignableFrom<Tensor<Half>>(InvokePrivate(
            model, "ApplySeriesStationarization", new Tensor<Half>(ToHalf(series), new[] { Count }), true));

        AssertNormalized(series, normalized);
    }

    [Fact]
    public void TSMixer_RevIN_Half_NormalizesWithAFiniteStd()
    {
        var series = LargeSeries(Count, seed: 14);
        var model = UninitializedHalfNetwork<TSMixer<Half>>();

        var normalized = Assert.IsAssignableFrom<Tensor<Half>>(InvokePrivate(
            model, "ApplyRevIN", new Tensor<Half>(ToHalf(series), new[] { Count }), true));

        AssertNormalized(series, normalized);
    }

    // (x - mean) / (population std + 1e-5), the stationarization both models apply. An infinite std made it all zeros.
    private static void AssertNormalized(double[] series, Tensor<Half> normalized)
    {
        double mean = Mean(series);
        double std = Math.Sqrt(SumSquaredDeviations(series) / series.Length) + 1e-5;
        var actual = normalized.ToArray();
        Assert.Equal(series.Length, actual.Length);
        for (int i = 0; i < series.Length; i++)
        {
            double expected = (series[i] - mean) / std;
            Assert.True(Math.Abs((double)actual[i] - expected) <= HalfTolerance * Math.Max(Math.Abs(expected), 1.0),
                $"element {i}: expected {expected:R}, got {(double)actual[i]:R}");
        }
    }

    // ---- Half error sums written by hand in src/Finance and src/TimeSeries ----

    private static TModel UninitializedHalfTimeSeries<TModel>() where TModel : AiDotNet.TimeSeries.TimeSeriesModelBase<Half>
    {
        var model = (TModel)RuntimeHelpers.GetUninitializedObject(typeof(TModel));
        GC.SuppressFinalize(model);
        var numOps = typeof(AiDotNet.TimeSeries.TimeSeriesModelBase<Half>).GetProperty(
            "NumOps", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("TimeSeriesModelBase.NumOps was not found.");
        numOps.SetValue(model, AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>());
        return model;
    }

    private static FieldInfo PrivateField(object target, string name)
    {
        for (Type? type = target.GetType(); type is not null; type = type.BaseType)
        {
            var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (field is not null)
            {
                return field;
            }
        }

        throw new InvalidOperationException($"{target.GetType().Name}.{name} was not found.");
    }

    private static void SetPrivate(object target, string name, object value) => PrivateField(target, name).SetValue(target, value);

    private static TValue GetPrivate<TValue>(object target, string name)
        => PrivateField(target, name).GetValue(target) is TValue value
            ? value
            : throw new InvalidOperationException($"{target.GetType().Name}.{name} is not a {typeof(TValue).Name}.");

    // A private or protected member declared on declaringType, chosen by the arguments' exact types.
    private static object? InvokeDeclared(Type declaringType, object target, string name, params object[] arguments)
    {
        var method = declaringType.GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic, null,
            arguments.Select(a => a.GetType()).ToArray(), null)
            ?? throw new InvalidOperationException($"{declaringType.Name}.{name} was not found.");
        return method.Invoke(target, arguments);
    }

    // 150 + 80 sin(.): every square (at most 230^2 = 52900) fits Half, a sum of 40 of them (about 1e6) does not.
    private static double[] ModerateSeries(int n, int seed)
        => Enumerable.Range(0, n).Select(i => (double)(Half)(150.0 + 80.0 * Math.Sin(0.9 * i + seed))).ToArray();

    private static void AssertErrorMetrics(double sumSquared, double sumAbsolute, int n, IReadOnlyDictionary<string, Half> metrics)
    {
        AssertClose(sumSquared / n, (double)metrics["MSE"], HalfTolerance);
        AssertClose(sumAbsolute / n, (double)metrics["MAE"], HalfTolerance);
        AssertClose(Math.Sqrt(sumSquared / n), (double)metrics["RMSE"], HalfTolerance);
    }

    [Fact]
    public void Chronos_Evaluate_Half_ReportsAFiniteMseWhereTheSumPassesHalfMaxValue()
    {
        var (predictions, actuals, mse, mae) = LargeErrors(Count);

        var metrics = UninitializedHalfNetwork<Chronos<Half>>().Evaluate(predictions, actuals);

        AssertErrorMetrics(mse * Count, mae * Count, Count, metrics);
    }

    // Targets 150 below the model's own forecast: each squared error (about 2.3e4) fits Half, their sum does not.
    private static void AssertEvaluateAgainstOffsetTargets(
        Tensor<Half> forecast, Func<Tensor<Half>, Dictionary<string, Half>> evaluate)
    {
        var values = forecast.ToArray();
        Assert.All(values, v => Assert.False(Half.IsNaN(v) || Half.IsInfinity(v), "the forecast is not finite"));
        var targets = new Half[values.Length];
        double sumSquared = 0, sumAbsolute = 0;
        for (int i = 0; i < values.Length; i++)
        {
            targets[i] = (Half)((double)values[i] - 150.0 - 5.0 * Math.Sin(i));
            double error = (double)values[i] - (double)targets[i];
            sumSquared += error * error;
            sumAbsolute += Math.Abs(error);
        }

        Assert.True(sumSquared > (double)Half.MaxValue);
        AssertErrorMetrics(sumSquared, sumAbsolute, values.Length, evaluate(new Tensor<Half>(targets, forecast.Shape.ToArray())));
    }

    [Fact]
    public void Autoformer_Evaluate_Half_ReportsAFiniteMseWhereTheSumPassesHalfMaxValue()
    {
        var options = new AiDotNet.Models.Options.AutoformerOptions<Half>
        {
            LookbackWindow = 24, LabelLength = 12, ForecastHorizon = 6, EmbeddingDim = 16, FeedForwardDim = 32,
            NumEncoderLayers = 1, NumDecoderLayers = 1, NumAttentionHeads = 2, MovingAverageKernel = 5, DropoutRate = 0.0,
            Seed = 3,
        };
        var model = new Autoformer<Half>(new NeuralNetworkArchitecture<Half>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        var window = new Tensor<Half>(new[] { 1, 24, 1 });
        for (int t = 0; t < 24; t++) window[0, t, 0] = (Half)(Math.Sin(t * 0.8) + 0.05 * t);

        AssertEvaluateAgainstOffsetTargets(model.Forecast(window), targets => model.Evaluate(window, targets));
    }

    [Fact]
    public void Informer_Evaluate_Half_ReportsAFiniteMseWhereTheSumPassesHalfMaxValue()
    {
        var options = new AiDotNet.Models.Options.InformerOptions<Half>
        {
            LookbackWindow = 16, LabelLength = 8, ForecastHorizon = 4, EmbeddingDim = 16, FeedForwardDim = 32,
            NumEncoderLayers = 2, NumDecoderLayers = 1, NumAttentionHeads = 2, SamplingFactor = 2, DropoutRate = 0.0,
            Seed = 3,
        };
        var model = new Informer<Half>(new NeuralNetworkArchitecture<Half>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        var window = new Tensor<Half>(new[] { 1, 16, 1 });
        for (int t = 0; t < 16; t++) window[0, t, 0] = (Half)Math.Sin(t * 0.7);

        AssertEvaluateAgainstOffsetTargets(model.Forecast(window), targets => model.Evaluate(window, targets));
    }

    [Fact]
    public void MultiTaskEvaluationHarness_EvaluateImputation_Half_ReportsAFiniteMse()
    {
        var (imputed, original, mse, _) = LargeErrors(Count);
        var model = new Mock<ITimeSeriesFoundationModel<Half>>();
        model.Setup(m => m.SupportedTasks).Returns(new[] { TimeSeriesFoundationModelTask.Imputation });
        model.Setup(m => m.Impute(It.IsAny<Tensor<Half>>(), It.IsAny<Tensor<Half>>())).Returns(imputed);
        var mask = new Tensor<Half>(new[] { Count }); // all zero: every position was masked

        var metrics = new MultiTaskEvaluationHarness<Half>().EvaluateImputation(model.Object, new[] { original }, new[] { mask });

        AssertClose(mse, (double)metrics["MSE"], HalfTolerance);
        AssertClose(Math.Sqrt(mse), (double)metrics["RMSE"], HalfTolerance);
        Assert.Equal((double)Count, (double)metrics["NumMaskedPoints"]);
    }

    // 800 forecast errors near 100 and a naive history whose 799 steps are near 100: both absolute sums pass Half.MaxValue.
    private const int GiftHorizon = 800;

    private static Half[] NaiveHistory()
        => Enumerable.Range(0, GiftHorizon).Select(i => (Half)(i % 2 == 0 ? 0.0 : 100.0 + 10.0 * Math.Sin(i))).ToArray();

    private static double NaiveMae(Half[] history)
    {
        double sum = 0;
        for (int i = 1; i < history.Length; i++) sum += Math.Abs((double)history[i] - (double)history[i - 1]);
        Assert.True(sum > (double)Half.MaxValue);
        return sum / (history.Length - 1);
    }

    // Sum over the horizon of the pinball loss of error = actual - predicted at level q, and sum |actual|.
    private static (double Loss, double AbsoluteTarget) PinballReference(Tensor<Half> predictions, Tensor<Half> actuals, double q)
    {
        var p = predictions.ToArray();
        var a = actuals.ToArray();
        double loss = 0, absoluteTarget = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double error = (double)a[i] - (double)p[i];
            loss += error >= 0 ? q * error : (q - 1.0) * error;
            absoluteTarget += Math.Abs((double)a[i]);
        }

        Assert.True(absoluteTarget > (double)Half.MaxValue);
        return (loss, absoluteTarget);
    }

    [Fact]
    public void GIFTEval_ComputeMASE_Half_IsFiniteWhereBothAbsoluteSumsPassHalfMaxValue()
    {
        var (predictions, actuals, _, mae) = LargeErrors(GiftHorizon);
        Assert.True(mae * GiftHorizon > (double)Half.MaxValue);
        var history = NaiveHistory();

        var mase = new GIFTEvalBenchmark<Half>().ComputeMASE(
            predictions, actuals, new Tensor<Half>(history, new[] { GiftHorizon }), seasonalPeriod: 1);

        AssertClose(mae / NaiveMae(history), (double)mase, HalfTolerance);
    }

    [Fact]
    public void GIFTEval_ComputeQuantileLossAndCRPS_Half_AreFiniteWhereTheSumsPassHalfMaxValue()
    {
        var (predictions, actuals, _, _) = LargeErrors(GiftHorizon);
        var (loss, absoluteTarget) = PinballReference(predictions, actuals, 0.5);
        var benchmark = new GIFTEvalBenchmark<Half>();

        AssertClose(loss / GiftHorizon, (double)benchmark.ComputeQuantileLoss(predictions, actuals, 0.5), HalfTolerance);
        AssertClose(2.0 * loss / absoluteTarget,
            (double)benchmark.ComputeCRPS(new Dictionary<double, Tensor<Half>> { [0.5] = predictions }, actuals), HalfTolerance);
    }

    [Fact]
    public void GIFTEval_RunBenchmark_Half_ScoresTheSeriesWhereItsSumsPassHalfMaxValue()
    {
        var (predictions, actuals, _, mae) = LargeErrors(GiftHorizon);
        var (loss, absoluteTarget) = PinballReference(predictions, actuals, 0.5);
        var history = NaiveHistory();
        var series = new Tensor<Half>(history.Concat(actuals.ToArray()).ToArray(), new[] { 2 * GiftHorizon });
        var model = new Mock<ITimeSeriesFoundationModel<Half>>();
        model.Setup(m => m.Forecast(It.IsAny<Tensor<Half>>(), It.IsAny<double[]>())).Returns(predictions);

        var results = new GIFTEvalBenchmark<Half>().RunBenchmark(
            model.Object, new[] { series }, new[] { GiftHorizon }, new[] { 1 }, new[] { 0.5 });

        Assert.Equal(1.0, (double)results["NumCRPSEvaluated"]);
        AssertClose(2.0 * loss / absoluteTarget, (double)results["CRPS"], HalfTolerance);
        AssertClose(mae / NaiveMae(history), (double)results["MASE"], HalfTolerance);
    }

    [Fact]
    public void TimeSeriesModelBase_CalculateErrorMetrics_Half_ReportsAFiniteMse()
    {
        var (predictions, actuals, mse, mae) = LargeErrors(Count);
        var p = predictions.ToArray();
        var a = actuals.ToArray();
        double mape = 100.0 * Enumerable.Range(0, Count).Average(i => Math.Abs((double)p[i] - (double)a[i]) / Math.Abs((double)a[i]));
        var model = UninitializedHalfTimeSeries<AiDotNet.TimeSeries.ARIMAModel<Half>>();

        var metrics = Assert.IsType<Dictionary<string, Half>>(InvokeDeclared(
            typeof(AiDotNet.TimeSeries.TimeSeriesModelBase<Half>), model, "CalculateErrorMetrics",
            new Vector<Half>(p), new Vector<Half>(a)));

        AssertErrorMetrics(mse * Count, mae * Count, Count, metrics);
        AssertClose(mape, (double)metrics["MAPE"], HalfTolerance);
    }

    // Absolute residuals near 200 with a spread near 150: the sum of squared deviations (about 4.5e5) passes Half.MaxValue.
    private static void AssertAbsoluteResidualStatistics(double[] absoluteResiduals, object model)
    {
        double mean = Mean(absoluteResiduals);
        double std = Math.Sqrt(SumSquaredDeviations(absoluteResiduals) / absoluteResiduals.Length);
        Assert.True(SumSquaredDeviations(absoluteResiduals) > (double)Half.MaxValue);
        AssertClose(mean, (double)GetPrivate<Half>(model, "_residualMean"), HalfTolerance);
        AssertClose(std, (double)GetPrivate<Half>(model, "_residualStdDev"), HalfTolerance);
        Assert.False(Half.IsInfinity(GetPrivate<Half>(model, "_anomalyThreshold")));
    }

    [Fact]
    public void ARIMA_AnomalyThreshold_Half_HasAFiniteResidualStd()
    {
        var residuals = LargeSeries(Count, seed: 21);
        var model = UninitializedHalfTimeSeries<AiDotNet.TimeSeries.ARIMAModel<Half>>();
        SetPrivate(model, "_arimaOptions", new AiDotNet.Models.Options.ARIMAOptions<Half>());

        InvokeDeclared(typeof(AiDotNet.TimeSeries.ARIMAModel<Half>), model, "ComputeAnomalyThreshold",
            new Vector<Half>(ToHalf(residuals)));

        AssertAbsoluteResidualStatistics(residuals, model);
    }

    // A Prophet whose fit is fixed, so the anomaly statistics see known residuals.
    private sealed class FixedFitProphet : AiDotNet.TimeSeries.ProphetModel<Half, Matrix<Half>, Vector<Half>>
    {
        public FixedFitProphet() : base(null)
        {
        }

        public Vector<Half> Fitted { get; set; } = new Vector<Half>(0);

        public override Vector<Half> Predict(Matrix<Half> input) => Fitted;
    }

    [Fact]
    public void Prophet_AnomalyThreshold_Half_HasAFiniteResidualStd()
    {
        var observed = LargeSeries(Count, seed: 22);
        var model = UninitializedHalfTimeSeries<FixedFitProphet>();
        model.Fitted = new Vector<Half>(Count);
        SetPrivate(model, "_prophetOptions", new AiDotNet.Models.Options.ProphetOptions<Half, Matrix<Half>, Vector<Half>>());

        InvokeDeclared(typeof(AiDotNet.TimeSeries.ProphetModel<Half, Matrix<Half>, Vector<Half>>), model,
            "ComputeAnomalyThresholdFromTraining", new Matrix<Half>(Count, 1), new Vector<Half>(ToHalf(observed)));

        AssertAbsoluteResidualStatistics(observed, model);
    }

    [Fact]
    public void MA_NoiseVariance_Half_IsFiniteWhereTheSumOfSquaresPassesHalfMaxValue()
    {
        // A zero MA(1) coefficient makes every residual the observation itself.
        var y = ModerateSeries(Count, seed: 23);
        double expected = y.Skip(1).Sum(v => v * v) / (Count - 1);
        var model = UninitializedHalfTimeSeries<AiDotNet.TimeSeries.MAModel<Half>>();

        var variance = Assert.IsType<Half>(InvokeDeclared(typeof(AiDotNet.TimeSeries.MAModel<Half>), model,
            "EstimateNoiseVariance", new Vector<Half>(ToHalf(y)), new Vector<Half>(1)));

        AssertClose(expected, (double)variance, HalfTolerance);
    }

    [Fact]
    public void BSTS_UpdateModelParameters_Half_GivesFiniteLevelAndObservationVariances()
    {
        // A level stepping by about 100 and observations about 100 above it: both sums of squares pass Half.MaxValue.
        var states = new Matrix<Half>(Count, 1);
        var y = new Half[Count];
        double levelSum = 0, observationSum = 0;
        for (int t = 0; t < Count; t++)
        {
            states[t, 0] = (Half)((t % 2 == 0 ? 0.0 : 100.0) + 5.0 * Math.Sin(t));
            y[t] = (Half)((double)states[t, 0] + 100.0 + 10.0 * Math.Sin(1.3 * t));
            double error = (double)y[t] - (double)states[t, 0];
            observationSum += error * error;
            if (t > 0)
            {
                double step = (double)states[t, 0] - (double)states[t - 1, 0];
                levelSum += step * step;
            }
        }

        Assert.True(levelSum > (double)Half.MaxValue && observationSum > (double)Half.MaxValue);
        var model = UninitializedHalfTimeSeries<AiDotNet.TimeSeries.BayesianStructuralTimeSeriesModel<Half>>();
        SetPrivate(model, "_bayesianOptions", new AiDotNet.Models.Options.BayesianStructuralTimeSeriesOptions<Half>
        {
            IncludeTrend = false,
            IncludeRegression = false,
        });
        SetPrivate(model, "_seasonalComponents", new List<Vector<Half>>());
        var covariance = new Matrix<Half>(1, 1);
        SetPrivate(model, "_stateCovariance", covariance);

        InvokeDeclared(typeof(AiDotNet.TimeSeries.BayesianStructuralTimeSeriesModel<Half>), model, "UpdateModelParameters",
            new Matrix<Half>(Count, 1), new Vector<Half>(y), states);

        AssertClose(levelSum / (Count - 1), (double)covariance[0, 0], HalfTolerance);
        AssertClose(observationSum / Count, (double)GetPrivate<Half>(model, "_observationVariance"), HalfTolerance);
    }

    [Fact]
    public void DynamicRegression_LogLikelihood_Half_IsFiniteWhereTheSumOfSquaresPassesHalfMaxValue()
    {
        var residuals = ModerateSeries(Count, seed: 24);
        double variance = residuals.Sum(r => r * r) / Count;
        double expected = -0.5 * Count * (Math.Log(2 * Math.PI * variance) + 1.0);
        var model = UninitializedHalfTimeSeries<AiDotNet.TimeSeries.DynamicRegressionWithARIMAErrors<Half>>();

        var logLikelihood = Assert.IsType<Half>(InvokeDeclared(typeof(AiDotNet.TimeSeries.DynamicRegressionWithARIMAErrors<Half>),
            model, "ComputeLogLikelihood", new Vector<Half>(ToHalf(residuals))));

        AssertClose(expected, (double)logLikelihood, HalfTolerance);
    }
#endif
}
