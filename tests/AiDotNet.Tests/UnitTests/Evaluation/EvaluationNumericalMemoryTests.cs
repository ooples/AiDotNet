#if NET5_0_OR_GREATER
using AiDotNet.Evaluation;
using AiDotNet.Enums;
using AiDotNet.Evaluation.Calibration;
using AiDotNet.Evaluation.Engines;
using AiDotNet.Models.Options;
using AiDotNet.Evaluation.Metrics.Classification;
using AiDotNet.Evaluation.Metrics.Probabilistic;
using AiDotNet.Evaluation.Metrics.Regression;
using AiDotNet.Evaluation.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

public class EvaluationNumericalMemoryTests
{
    [Fact]
    public void HalfValidationSummaries_KeepLargeCountsAndSampleVarianceWide()
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic;
        var cv = typeof(CrossValidationEngine<Half>).GetMethod("Summarize", flags)!;
        var curve = typeof(LearningCurveEngine<Half>).GetMethod("MeanAndStandardDeviation", flags)!;
        var values = Enumerable.Repeat((Half)60000, 70000).ToList();
        Assert.Equal((60000.0, 0.0, 0.0), Assert.IsType<(double, double, double)>(cv.Invoke(null, new object[] { values })));
        Assert.Equal((60000.0, 0.0), Assert.IsType<(double, double)>(curve.Invoke(null, new object[] { values })));
        values = new List<Half> { (Half)(-60000), (Half)60000 };
        var (mean, std, margin) = Assert.IsType<(double, double, double)>(cv.Invoke(null, new object[] { values }));
        Assert.Equal(0.0, mean);
        Assert.Equal(Math.Sqrt(7200000000.0), std, 8);
        Assert.Equal(240000.0, margin, 8);
        var (curveMean, curveStd) = Assert.IsType<(double, double)>(curve.Invoke(null, new object[] { values }));
        Assert.Equal(mean, curveMean);
        Assert.Equal(std, curveStd);
        Assert.Equal((60000.0, 0.0), Assert.IsType<(double, double)>(
            curve.Invoke(null, new object[] { new List<Half> { (Half)60000 } })));
    }

    [Fact]
    public void PredictionTypeInference_FloatIntegerBoundsDoNotRoundOutward()
    {
        Assert.Equal(PredictionType.Regression, PredictionTypeInference.Infer(
            new Vector<float>(Enumerable.Repeat(2147483648f, 10).ToArray())));
        Assert.Equal(PredictionType.Regression, PredictionTypeInference.Infer(
            new Vector<float>(Enumerable.Repeat(-2147483904f, 10).ToArray())));
        Assert.Equal(PredictionType.MultiClass, PredictionTypeInference.Infer(
            new Vector<float>(Enumerable.Repeat(2147483520f, 10).ToArray())));
        Assert.Equal(PredictionType.MultiClass, PredictionTypeInference.Infer(
            new Vector<float>(Enumerable.Repeat(-2147483648f, 10).ToArray())));
    }


    [Fact]
    public void HalfPercentageMetrics_KeepCountsErrorsAndPercentageScalingWide()
    {
        var actual = Enumerable.Repeat((Half)60000, 70000).ToArray();
        var predicted = new Half[actual.Length];
        Assert.Equal((Half)100, new MAPEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)200, new SymmetricMAPEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)100, new WeightedMAPEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)200, new AiDotNet.Evaluation.Metrics.TimeSeries.SMAPEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)1, new AiDotNet.Evaluation.Metrics.TimeSeries.WAPEMetric<Half>().Compute(predicted, actual));
        Half[] opposite = { (Half)(-60000) };
        Half[] positive = { (Half)60000 };
        Assert.Equal((Half)200, new SymmetricMAPEMetric<Half>().Compute(opposite, positive));
        Assert.Equal((Half)200, new WeightedMAPEMetric<Half>().Compute(opposite, positive));
        Assert.Equal((Half)2, new AiDotNet.Evaluation.Metrics.TimeSeries.WAPEMetric<Half>().Compute(opposite, positive));
        Assert.Equal((Half)0, new MAPEMetric<Half>().Compute(new Half[1], new Half[1]));
        Assert.Equal((Half)0, new WeightedMAPEMetric<Half>().Compute(new Half[1], new Half[1]));
        Assert.Equal((Half)0, new AiDotNet.Evaluation.Metrics.TimeSeries.WAPEMetric<Half>().Compute(new Half[1], new Half[1]));
    }

    [Fact]
    public void HalfMeanBias_AllowsCancellationBeforeNarrowing()
    {
        Half[] predicted = { (Half)60000, (Half)(-60000) };
        Half[] actual = { (Half)(-60000), (Half)60000 };
        Assert.Equal((Half)0, new MeanBiasErrorMetric<Half>().Compute(predicted, actual));
    }

    [Fact]
    public void HalfLogMetrics_KeepSmallErrorsUntilTheFinalReduction()
    {
        Half[] predicted = { (Half)0.0001 };
        var actual = new Half[1];
        double logError = Math.Log(1 + (double)predicted[0]);
        Assert.Equal((Half)logError, new RMSLEMetric<Half>().Compute(predicted, actual));
        predicted = Enumerable.Repeat((Half)60000, 70000).ToArray();
        actual = new Half[predicted.Length];
        logError = Math.Log(60001);
        Assert.Equal((Half)(logError * logError), new MeanSquaredLogErrorMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)logError, new RMSLEMetric<Half>().Compute(predicted, actual));
        Assert.True(double.IsNaN(new RMSLEMetric<double>().Compute(new[] { double.NaN }, new[] { 0.0 })));
    }

    [Theory]
    [InlineData(1e-9, -60000, 60000)]
    [InlineData(0.999999999, 60000, -60000)]
    public void HalfQuantileMetrics_PreserveSmallWeightsAndWideDifferences(double quantile, double prediction, double actual)
    {
        Half[] predicted = { (Half)prediction };
        Half[] observed = { (Half)actual };
        double difference = actual - prediction;
        Half expected = (Half)(difference >= 0 ? quantile * difference : (quantile - 1) * difference);
        Assert.Equal(expected, new QuantileLossMetric<Half>(quantile).Compute(predicted, observed));
        Assert.Equal(expected, new PinballLossMetric<Half>(quantile).Compute(predicted, observed));
    }


    [Fact]
    public void PlattScaling_PublicFitMatchesSmoothedClosedForm()
    {
        double[] scores = { 0, 0, 1, 1 };
        double[] labels = { 0, 0, 1, 1 };
        var options = new ProbabilityCalibratorOptions
            { CalibratorMethod = ProbabilityCalibrationMethod.PlattScaling, MaxIterations = 20, Tolerance = 1e-12 };
        var wide = new ProbabilityCalibrator<double>(options);
        var result = wide.FitTransform(new Vector<double>(scores), new Vector<double>(labels));
        var (a, b) = wide.GetPlattParameters();
        Assert.Equal(2 * Math.Log(3), a, 6);
        Assert.Equal(-Math.Log(3), b, 6);
        for (int i = 0; i < scores.Length; i++)
            Assert.Equal(i < 2 ? 0.25 : 0.75, result[i], 6);
        var half = new ProbabilityCalibrator<Half>(options);
        var halfResult = half.FitTransform(new Vector<Half>(scores.Select(x => (Half)x).ToArray()),
            new Vector<Half>(labels.Select(x => (Half)x).ToArray()));
        for (int i = 0; i < scores.Length; i++)
            Assert.InRange(Math.Abs((double)halfResult[i] - (i < 2 ? 0.25 : 0.75)), 0, 0.001);
        var (halfA, halfB) = half.GetPlattParameters();
        Assert.InRange(Math.Abs((double)halfA - 2 * Math.Log(3)), 0, 0.002);
        Assert.InRange(Math.Abs((double)halfB + Math.Log(3)), 0, 0.001);
    }


    [Fact]
    public void HalfVarianceRatios_PreserveFiniteRatiosAndConstantCases()
    {
        Half[] actual = { (Half)(-60000), (Half)(-30000), (Half)30000, (Half)60000 };
        var predicted = new Half[actual.Length];
        Assert.Equal((Half)1, new NormalizedMSEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)1, new RelativeAbsoluteErrorMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)1, new RelativeSquaredErrorMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)0, new ExplainedVarianceMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)(-0.5), new AdjustedR2Metric<Half>(1).Compute(predicted, actual));
        var constant = Enumerable.Repeat((Half)1, 4).ToArray();
        Assert.Equal((Half)1, new AdjustedR2Metric<Half>(1).Compute(constant, constant));
        Assert.Equal((Half)1, new ExplainedVarianceMetric<Half>().Compute(predicted, constant));
        Assert.Equal((Half)0, new NormalizedMSEMetric<Half>().Compute(constant, constant));
        Assert.Equal((Half)0, new RelativeAbsoluteErrorMetric<Half>().Compute(constant, constant));
        Assert.Equal((Half)0, new RelativeSquaredErrorMetric<Half>().Compute(constant, constant));
        Assert.Equal((Half)0, new AdjustedR2Metric<Half>(int.MaxValue).Compute(predicted, actual));
        Assert.Equal(1f, new NormalizedMSEMetric<float>().Compute(new[] { 0f, 0f }, new[] { -1e20f, 1e20f }));
    }

    [Fact]
    public void HalfSpearman_PreservesLargeTiedRankSumsAndConstantRanks()
    {
        var values = Enumerable.Range(0, 70000).Select(i => (Half)(i % 2)).ToArray();
        var reversed = values.Select(value => (Half)(1 - (double)value)).ToArray();
        var metric = new SpearmanCorrelationMetric<Half>();
        Assert.Equal((Half)1, metric.Compute(values, values));
        Assert.Equal((Half)(-1), metric.Compute(values, reversed));
        Assert.Equal((Half)0, metric.Compute(new Half[values.Length], values));
    }

    [Fact]
    public void WideMedianErrors_AverageBeforeRoundingToHalf()
    {
        var metric = new MedianAbsoluteErrorMetric<Half>();
        Assert.Equal((Half)60000, metric.Compute(new[] { (Half)60000, (Half)60000 }, new Half[2]));
        Assert.Equal((Half)60000, metric.Compute(new[] { (Half)0, (Half)60000 }, new[] { (Half)0, (Half)(-60000) }));
        Assert.Equal(1.0, new MedianAbsoluteErrorMetric<double>().Compute(new[] { double.NaN, 1.0, 2.0 }, new double[3]));
    }

    [Theory]
    [InlineData(3, 5)]
    [InlineData(257, 259)]
    public void DeLong_BoundedComponentsMatchScalarPairOracle(int positives, int negatives)
    {
        double[] scores = positives == 3
            ? new[] { 0.5, 1.0, double.PositiveInfinity, 0.5, 0.5 + 5e-11, double.NegativeInfinity, double.PositiveInfinity, double.NaN }
            : Enumerable.Range(0, positives + negatives).Select(i => (double)(i % 7)).ToArray();
        var positiveMask = new Tensor<double>(Enumerable.Range(0, scores.Length).Select(i => i < positives ? 1.0 : 0.0).ToArray(), new[] { scores.Length });
        var negativeMask = new Tensor<double>(Enumerable.Range(0, scores.Length).Select(i => i < positives ? 0.0 : 1.0).ToArray(), new[] { scores.Length });
        var method = typeof(DeLongTest<double>).GetMethod("ComputeComponents", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        var (auc, rows, columns) = Assert.IsType<(double, Tensor<double>, Tensor<double>)>(method.Invoke(null, new object[] { scores, positiveMask, negativeMask }));
        var rowSums = new double[positives];
        var columnSums = new double[negatives];
        double total = 0;
        for (int i = 0; i < positives; i++)
        for (int j = 0; j < negatives; j++)
        {
            double left = scores[i], right = scores[positives + j];
            double kernel = left > right ? 1 : Math.Abs(left - right) < 1e-10 ? 0.5 : 0;
            rowSums[i] += kernel;
            columnSums[j] += kernel;
            total += kernel;
        }
        Assert.Equal(total / ((double)positives * negatives), auc, 12);
        for (int i = 0; i < positives; i++) Assert.Equal(rowSums[i] / negatives, rows.GetFlat(i), 12);
        for (int j = 0; j < negatives; j++) Assert.Equal(columnSums[j] / positives, columns.GetFlat(j), 12);
    }

    [Fact]
    public void DeLong_HalfKeepsPairTotalsWide()
    {
        var labels = Enumerable.Range(0, 514).Select(i => i < 257 ? (Half)1 : (Half)0).ToArray();
        var tied = Enumerable.Repeat((Half)0.5, labels.Length).ToArray();
        var result = new DeLongTest<Half>().Test(labels, tied, labels);
        Assert.Equal((Half)0.5, result.EffectSize);
        Assert.Equal((Half)0, result.Statistic);
        Assert.Equal((Half)1, result.PValue);
    }

    [Fact]
    public void Calibration_HalfRetainsExactLargeBinCountsAndWideMeans()
    {
        var scores = new Vector<Half>(Enumerable.Repeat((Half)0.75, 70000).ToArray());
        var labels = new Vector<Half>(Enumerable.Repeat((Half)1, scores.Length).ToArray());
        var calibrator = new ProbabilityCalibrator<Half>();
        Assert.Equal(0.25, calibrator.ComputeECE(scores, labels));
        Assert.Equal(0.25, calibrator.ComputeMCE(scores, labels));
        Assert.Equal(0.0625, calibrator.ComputeBrierScore(scores, labels));
        var diagram = calibrator.GetReliabilityDiagram(scores, labels);
        Assert.Equal(70000, diagram.binCounts[7]);
        Assert.Equal((Half)0.75, diagram.meanPredicted[7]);
        Assert.Equal((Half)1, diagram.fractionPositives[7]);
        var analysis = new CalibrationEngine<Half>().Analyze(scores.ToArray(), labels.ToArray());
        Assert.Equal(70000, Assert.Single(analysis.Bins).SampleCount);
        Assert.Equal((Half)0.25, analysis.ExpectedCalibrationError);
        Assert.Equal((Half)0.0625, analysis.BrierScore);
    }

    [Fact]
    public void Calibration_BlockedHistogramMatchesAnalyticBinsAndFallbacks()
    {
        const int count = 131073;
        var scores = new Vector<double>(Enumerable.Range(0, count).Select(i => i % 2 == 0 ? 0.25 : 0.75).ToArray());
        var labels = new Vector<double>(Enumerable.Range(0, count).Select(i => i % 2 == 0 ? 0.0 : 1.0).ToArray());
        var calibrator = new ProbabilityCalibrator<double>(new ProbabilityCalibratorOptions
            { CalibratorMethod = ProbabilityCalibrationMethod.HistogramBinning, NumBins = 32 });
        var diagram = calibrator.GetReliabilityDiagram(scores, labels, 32);
        Assert.Equal(65537, diagram.binCounts[8]);
        Assert.Equal(65536, diagram.binCounts[24]);
        Assert.Equal(0.25, calibrator.ComputeECE(scores, labels, 32));
        calibrator.Fit(scores, labels);
        var calibrated = calibrator.Transform(scores);
        for (int i = 0; i < count; i++) Assert.Equal(i % 2 == 0 ? 0.0 : 1.0, calibrated[i]);
        Assert.All(calibrator.Transform(new Vector<double>(new[] { double.NaN, -1.0, 1.0, double.PositiveInfinity })).ToArray(),
            value => Assert.Equal(0.984375, value));
    }

    [Theory]
    [InlineData(ProbabilityCalibrationMethod.PlattScaling)]
    [InlineData(ProbabilityCalibrationMethod.TemperatureScaling)]
    [InlineData(ProbabilityCalibrationMethod.BetaCalibration)]
    public void Calibration_HalfFittingMatchesWideReferenceAtProbabilityEndpoints(ProbabilityCalibrationMethod method)
    {
        var options = new ProbabilityCalibratorOptions { CalibratorMethod = method, MaxIterations = 5 };
        double[] scores = { 0, 0, 0.5, 1, 1 };
        double[] labels = { 0, 1, 0, 1, 1 };
        var wide = new ProbabilityCalibrator<double>(options);
        var half = new ProbabilityCalibrator<Half>(options);
        var expected = wide.FitTransform(new Vector<double>(scores), new Vector<double>(labels));
        var actual = half.FitTransform(new Vector<Half>(scores.Select(x => (Half)x).ToArray()),
            new Vector<Half>(labels.Select(x => (Half)x).ToArray()));
        for (int i = 0; i < scores.Length; i++) Assert.InRange(Math.Abs(expected[i] - (double)actual[i]), 0, 0.002);
    }

    [Fact]
    public void Calibration_HalfBayesianBinningPreservesLargePrefixCounts()
    {
        var scores = new Vector<Half>(Enumerable.Range(0, 70000).Select(i => i < 35000 ? (Half)0.25 : (Half)0.75).ToArray());
        var labels = new Vector<Half>(Enumerable.Range(0, 70000).Select(i => i < 35000 ? (Half)0 : (Half)1).ToArray());
        var calibrator = new ProbabilityCalibrator<Half>(new ProbabilityCalibratorOptions
            { CalibratorMethod = ProbabilityCalibrationMethod.BayesianBinning, NumBins = 4 });
        calibrator.Fit(scores, labels);
        Assert.Equal(new[] { (Half)0, (Half)1 }, calibrator.Transform(new Vector<Half>(new[] { (Half)0.25, (Half)0.75 })).ToArray());
    }

    [Fact]
    public void CalibrationEngine_HalfPlattUsesWideOptimizerArithmetic()
    {
        double[] scores = { 0, 0.25, 0.75, 1 };
        double[] labels = { 0, 0, 1, 1 };
        var expected = new CalibrationEngine<double>().PlattScaling(scores, labels);
        var actual = new CalibrationEngine<Half>().PlattScaling(scores.Select(x => (Half)x).ToArray(), labels.Select(x => (Half)x).ToArray());
        Assert.Equal(expected.A, actual.A, 10);
        Assert.Equal(expected.B, actual.B, 10);
        for (int i = 0; i < scores.Length; i++)
            Assert.Equal((Half)expected.CalibratedProbabilities[i], actual.CalibratedProbabilities[i]);
    }

    [Fact]
    public void HingeLoss_SelectsInfiniteMarginsWithoutMaskMultiplication()
    {
        var metric = new HingeLossMetric<double>();
        Assert.Equal(0.0, metric.Compute(new[] { double.PositiveInfinity, double.NegativeInfinity }, new[] { 1.0, 0.0 }));
        Assert.Equal(double.PositiveInfinity, metric.Compute(new[] { double.PositiveInfinity }, new[] { 0.0 }));
        Assert.True(double.IsNaN(metric.Compute(new[] { double.NaN }, new[] { 1.0 })));
        Assert.Equal(0.5, metric.Compute(new[] { 0.25, 0.75 }, new[] { 0.0, 1.0 }));
    }

    [Fact]
    public void HalfProbabilisticMetrics_KeepReductionsAndVarianceFloorWide()
    {
        var predicted = Enumerable.Repeat((Half)1, 70000).ToArray();
        var actual = new Half[predicted.Length];
        Assert.Equal((Half)1, new CRPSMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)2, new HingeLossMetric<Half>().Compute(predicted, actual));
        var logScore = new LogScoreMetric<Half>();
        Assert.Equal((Half)(0.5 * Math.Log(2 * Math.PI) + 0.5), logScore.Compute(predicted, actual));
        Assert.Equal((Half)(0.5 * Math.Log(2 * Math.PI * 1e-15)), logScore.Compute(actual, actual));
        Assert.Equal((Half)30000, new CRPSMetric<Half>().Compute(
            new[] { (Half)60000, (Half)0, (Half)0, (Half)0 },
            new[] { (Half)(-60000), (Half)0, (Half)0, (Half)0 }));
        Assert.Equal((Half)(0.5 * Math.Log(2 * Math.PI * 60000.0 * 60000) + 0.5),
            logScore.Compute(new[] { (Half)60000 }, new[] { (Half)0 }));
    }

    [Fact]
    public void HalfLossMeans_KeepLargeSampleCountsWide()
    {
        var predicted = Enumerable.Repeat((Half)1, 70000).ToArray();
        var actual = new Half[predicted.Length];
        Assert.Equal((Half)1, new MSEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)1, new RMSEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)1, new MAEMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)0.5, new HuberLossMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)Math.Log(Math.Cosh(1)), new LogCoshLossMetric<Half>().Compute(predicted, actual));
        Assert.Equal((Half)1, new BrierScoreMetric<Half>().Compute(predicted, actual));
    }

    [Fact]
    public void WideLossArithmetic_RoundsOnlyTheFinalRepresentableResult()
    {
        Assert.Equal(1e20f, new RMSEMetric<float>().Compute(new[] { 1e20f }, new[] { 0f }));
        Assert.Equal((Half)30000, new MAEMetric<Half>().Compute(
            new[] { (Half)60000, (Half)0, (Half)0, (Half)0 },
            new[] { (Half)(-60000), (Half)0, (Half)0, (Half)0 }));
        Assert.Equal((Half)60, new HuberLossMetric<Half>(1e-9).Compute(new[] { (Half)60000 }, new[] { (Half)0 }));
    }

    [Fact]
    public void HalfR2_PreservesConstantAndHighVarianceCases()
    {
        var metric = new R2ScoreMetric<Half>();
        Assert.Equal((Half)1, metric.Compute(new[] { (Half)1, (Half)1 }, new[] { (Half)1, (Half)1 }));
        Assert.Equal((Half)0, metric.Compute(new[] { (Half)0, (Half)0 }, new[] { (Half)1, (Half)1 }));
        Assert.Equal((Half)0, metric.Compute(new[] { (Half)0, (Half)0 }, new[] { (Half)(-60000), (Half)60000 }));
    }

    [Fact]
    public void HalfLogLoss_KeepsPositiveProbabilityFloorInBinaryAndMulticlassPaths()
    {
        var metric = new LogLossMetric<Half>(1e-6);
        Half expected = (Half)(-Math.Log(1e-6));
        Assert.Equal(expected, metric.Compute(new[] { (Half)1 }, new[] { (Half)0 }));
        Assert.Equal(expected, metric.Compute(new[] { (Half)0, (Half)0.5, (Half)0.5 }, new[] { (Half)0 }, 3));
        var probabilities = new Half[2053];
        probabilities[2052] = (Half)1;
        Assert.Equal((Half)0, new BrierScoreMetric<Half>().Compute(probabilities, new[] { (Half)2052 }, 2053));
    }

    [Fact]
    public void DatasetRanks_PreserveIndependentTiesAndDatasetOrder()
    {
        double[][] samples = { new[] { 3.0, 1, 2 }, new[] { 2.0, 3, 2 }, new[] { 1.0, 2, 3 } };
        var result = new NemenyiPostHocTest<double>().Test(samples);
        Assert.Equal(6.5 / 3, result.AverageRanks[0], 12);
        Assert.Equal(5.5 / 3, result.AverageRanks[1], 12);
        Assert.Equal(2.0, result.AverageRanks[2], 12);
        Assert.Equal(2.0 / 35, new FriedmanTest<double>().Test(samples).Statistic, 12);
    }

    [Fact]
    public void EnergyScore_LargeEnsembleMatchesTwoPointDistribution()
    {
        const int members = 513, dimensions = 513;
        var samples = Enumerable.Range(0, members)
            .Select(i => Enumerable.Repeat((double)(i % 2), dimensions).ToArray()).ToArray();
        var observation = Enumerable.Repeat(0.5, dimensions).ToArray();
        double distance = Math.Sqrt(dimensions);
        double expected = 0.5 * distance - (double)(members / 2) * (members - members / 2) / (members * members) * distance;
        Assert.Equal(expected, new EnergyScoreMetric<double>().ComputeFromEnsemble(samples, observation), 10);
    }

    [Fact]
    public void EnergyScore_ArbitraryVectorsMatchScalarPairOracle()
    {
        double[][] samples = { new[] { 1.0, -2, 3 }, new[] { 4.0, 0, -1 }, new[] { -2.0, 5, 2 } };
        double[] observation = { 0.5, 1, -3 };
        static double Distance(double[] a, double[] b)
        {
            double sum = 0;
            for (int d = 0; d < a.Length; d++) sum += (a[d] - b[d]) * (a[d] - b[d]);
            return Math.Sqrt(sum);
        }
        double expected = samples.Sum(s => Distance(s, observation)) / samples.Length;
        foreach (var left in samples)
        foreach (var right in samples) expected -= 0.5 * Distance(left, right) / (samples.Length * samples.Length);
        var metric = new EnergyScoreMetric<double>();
        Assert.Equal(expected, metric.ComputeFromEnsemble(samples, observation), 12);
        Assert.Equal(0.0, metric.ComputeFromEnsemble(new[] { Array.Empty<double>() }, Array.Empty<double>()));
        Assert.Equal(Distance(samples[0], observation), metric.ComputeFromEnsemble(new[] { samples[0] }, observation), 12);
    }

    [Fact]
    public void HalfEnergyScore_KeepsRepresentableResultAfterWideDistances()
    {
        Half[][] samples = { new[] { (Half)(-60000) }, new[] { (Half)60000 } };
        Assert.Equal((Half)30000, new EnergyScoreMetric<Half>().ComputeFromEnsemble(samples, new[] { (Half)0 }));
    }

    [Fact]
    public void Pearson_WideMomentsPreserveFiniteCorrelations()
    {
        var metric = new PearsonCorrelationMetric<float>();
        float[] values = { -1e20f, 1e20f };
        Assert.Equal(1f, metric.Compute(values, values));
        Assert.Equal(-1f, metric.Compute(values, new[] { 1e20f, -1e20f }));
        Half[] half = { (Half)(-1000), (Half)1000 };
        Assert.Equal((Half)1, new PearsonCorrelationMetric<Half>().Compute(half, half));
        Assert.Equal((Half)0, new PearsonCorrelationMetric<Half>().Compute(new[] { (Half)1, (Half)1 }, half));
    }

    [Fact]
    public void HalfAucAndAveragePrecision_KeepLargeCountsWide()
    {
        var scores = Enumerable.Repeat((Half)0.5, 70000).ToArray();
        var labels = Enumerable.Range(0, scores.Length).Select(i => (Half)(i % 2)).ToArray();
        Assert.Equal((Half)0.5, new AUCROCMetric<Half>().Compute(scores, labels));
        Assert.Equal((Half)0.5, new AUCPRMetric<Half>().Compute(scores, labels));
        scores = Enumerable.Range(0, 514).Select(i => (Half)(i / 257)).ToArray();
        Assert.Equal((Half)1, new AUCROCMetric<Half>().Compute(scores, scores));
    }

    [Fact]
    public void HalfMidRanks_DoNotOverflowBeforeHalving()
    {
        const int size = 65504, group = size / 2;
        var values = new Tensor<Half>(Enumerable.Range(0, size).Select(i => (Half)(i / group)).ToArray(), new[] { size });
        var ranks = EvaluationTensorOps<Half>.MidRanks(values);
        Assert.Equal((Half)((group + 1.0) / 2), ranks.GetFlat(0));
        Assert.Equal((Half)((group + 1.0 + size) / 2), ranks.GetFlat(size - 1));
    }

    [Fact]
    public void HalfRunRanks_PreserveTieRunsBeyond2048Groups()
    {
        const int size = 4000;
        var values = Enumerable.Range(0, size).Select(i => (Half)(i / 4.0)).ToArray();
        var tensor = new Tensor<Half>(values, new[] { size });
        var (_, _, ranks) = EvaluationTensorOps<Half>.SortedRunRanks(tensor, 1e-10);
        var rows = new Tensor<Half>(values.Concat(values.Reverse()).ToArray(), new[] { 2, size });
        var descending = EvaluationTensorOps<Half>.RowRanksDescending(rows, 1e-10);
        for (int start = 0; start < size;)
        {
            int end = start + 1;
            while (end < size && values[end] == values[start]) end++;
            Half ascendingRank = (Half)((start + 1.0 + end) / 2);
            Half descendingRank = (Half)((size - end + 1.0 + size - start) / 2);
            for (int index = start; index < end; index++)
            {
                Assert.Equal(ascendingRank, ranks.GetFlat(index));
                Assert.Equal(descendingRank, descending[0, index]);
                Assert.Equal(descendingRank, descending[1, size - 1 - index]);
            }
            start = end;
        }
    }

    [Fact]
    public void HalfOneHot_UsesExactIntegerClassIndices()
    {
        var result = EvaluationTensorOps<Half>.RoundedOneHot(new Tensor<Half>(new[] { (Half)2052 }, new[] { 1 }), 2053);
        Assert.Equal((Half)1, result[0, 2052]);
        Assert.Equal(1, result.ToArray().Count(value => value != (Half)0));
    }

    [Fact]
    public void DecimalBestIndex_DoesNotConstructNaNForFiniteValues()
    {
        var values = new Tensor<decimal>(new[] { 3m, 9m, 9m, -2m }, new[] { 4 });
        Assert.Equal(1, EvaluationTensorOps<decimal>.CompareScanBestIndex(values, true));
        Assert.Equal(3, EvaluationTensorOps<decimal>.CompareScanBestIndex(values, false));
    }

    [Fact]
    public void HalfDirectionalAccuracy_CountsFlatAndExtremeChangesCorrectly()
    {
        var constant = Enumerable.Repeat((Half)60000, 70000).ToArray();
        Assert.Equal((Half)1, new MeanDirectionalAccuracyMetric<Half>().Compute(constant, constant));
        Half[] rising = { (Half)(-60000), (Half)0, (Half)60000 };
        Half[] falling = { (Half)60000, (Half)0, (Half)(-60000) };
        Assert.Equal((Half)1, new MeanDirectionalAccuracyMetric<Half>().Compute(rising, rising));
        Assert.Equal((Half)0, new MeanDirectionalAccuracyMetric<Half>().Compute(rising, falling));
    }

    [Fact]
    public void HalfTimeSeriesScaling_PreservesFiniteRatiosAndZeroBaselines()
    {
        const int count = 70000;
        var actual = Enumerable.Range(0, count).Select(i => i % 2 == 0 ? (Half)(-60000) : (Half)60000).ToArray();
        var predicted = new Half[count];
        var mase = new AiDotNet.Evaluation.Metrics.TimeSeries.MASEMetric<Half>();
        var seasonal = new AiDotNet.Evaluation.Metrics.TimeSeries.MASESeasonalMetric<Half>(1);
        var theil = new AiDotNet.Evaluation.Metrics.TimeSeries.TheilUMetric<Half>();
        Assert.Equal((Half)0.5, mase.Compute(predicted, actual));
        Assert.Equal((Half)0.5, seasonal.Compute(predicted, actual));
        Assert.Equal((Half)Math.Sqrt((double)count / (4 * (count - 1))), theil.Compute(predicted, actual));
        var zeros = new Half[3];
        Half[] ones = { (Half)1, (Half)1, (Half)1 };
        Assert.Equal((Half)0, mase.Compute(zeros, zeros));
        Assert.Equal((Half)0, seasonal.Compute(zeros, zeros));
        Assert.Equal((Half)0, theil.Compute(zeros, zeros));
        Assert.Equal(Half.MaxValue, mase.Compute(ones, zeros));
        Assert.Equal(Half.MaxValue, seasonal.Compute(ones, zeros));
        Assert.Equal(Half.MaxValue, theil.Compute(ones, zeros));
    }

    [Fact]
    public void HalfPoissonDeviance_PreservesPositiveFloorAndLargeReduction()
    {
        var metric = new PoissonDevianceMetric<Half>();
        Assert.Equal((Half)(2 * (Math.Log(1e10) - (1 - 1e-10))),
            metric.Compute(new Half[1], new[] { (Half)1 }));
        var predicted = Enumerable.Repeat((Half)1, 70000).ToArray();
        var actual = Enumerable.Repeat((Half)2, predicted.Length).ToArray();
        Assert.Equal((Half)(2 * (2 * Math.Log(2) - 1)), metric.Compute(predicted, actual));
    }

    [Theory]
    [InlineData(1.0, 1.0)]
    [InlineData(1.5, 0.0001)]
    [InlineData(2.0, 0.000000059604644775390625)]
    public void HalfTweedieLoss_PreservesFloorAcrossDistributionBranches(double power, double label)
    {
        Half observed = (Half)label;
        double y = (double)observed, mu = 1e-10;
        double expected = power == 1
            ? 2 * (y * Math.Log(y / mu) - (y - mu))
            : power == 2
                ? 2 * ((y - mu) / mu - Math.Log(y / mu))
                : 2 * (Math.Max(1e-10, Math.Pow(y, 2 - power)) / ((1 - power) * (2 - power))
                    - y * Math.Pow(mu, 1 - power) / (1 - power)
                    + Math.Pow(mu, 2 - power) / (2 - power));
        Assert.True(Half.IsFinite((Half)expected));
        Assert.Equal((Half)expected, new TweedieLossMetric<Half>(power).Compute(new Half[1], new[] { observed }));
    }

    [Fact]
    public void HalfTweedieNormal_KeepsLargeSquaredErrorsUntilMean()
    {
        var predicted = new Half[70000];
        predicted[0] = (Half)60000;
        Assert.Equal((Half)(3600000000.0 / predicted.Length),
            new TweedieLossMetric<Half>(0).Compute(predicted, new Half[predicted.Length]));
    }

    [Fact]
    public void HalfCochranQ_ReducesLargeSuccessCountsWithoutOverflow()
    {
        const int count = 70000;
        var first = Enumerable.Range(0, count).Select(i => i < 35100 ? (Half)0 : (Half)1).ToArray();
        var second = Enumerable.Range(0, count).Select(i => i < 35000 ? (Half)0 : (Half)1).ToArray();
        var metric = new CochranQTest<Half>();
        var result = metric.Test(new[] { first, second }, new Half[count]);
        Assert.Equal((Half)100, result.Statistic);
        Assert.True(result.IsSignificant);
        Assert.Equal((Half)0, result.PValue);
        var identical = metric.Test(new[] { first, first }, new Half[count]);
        Assert.Equal((Half)0, identical.Statistic);
        Assert.Equal((Half)1, identical.PValue);
    }

    [Fact]
    public void HalfPairedT_KeepsDifferencesAndVarianceWide()
    {
        var metric = new PairedTTest<Half>();
        Half[] first = { (Half)60000, (Half)60000, (Half)60000 };
        Half[] second = { (Half)(-60000), (Half)0, (Half)60000 };
        var result = metric.Test(first, second);
        Assert.Equal((Half)Math.Sqrt(3), result.Statistic);
        Assert.Equal((Half)1, result.EffectSize);
        var symmetric = metric.Test(new[] { (Half)(-60000), (Half)0, (Half)60000 },
            new[] { (Half)60000, (Half)0, (Half)(-60000) });
        Assert.Equal((Half)0, symmetric.Statistic);
        Assert.Equal((Half)0, symmetric.EffectSize);
        Assert.Equal((Half)1, symmetric.PValue);
    }

}
#endif
