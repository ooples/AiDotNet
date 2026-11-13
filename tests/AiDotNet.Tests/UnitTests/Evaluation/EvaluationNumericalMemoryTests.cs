#if NET5_0_OR_GREATER
using ConfidenceIntervalMethod = AiDotNet.Evaluation.Enums.ConfidenceIntervalMethod;
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
    [Theory]
    [InlineData(ConfidenceIntervalMethod.PercentileBootstrap, 1)]
    [InlineData(ConfidenceIntervalMethod.BasicBootstrap, 1)]
    [InlineData(ConfidenceIntervalMethod.BCaBootstrap, 1)]
    [InlineData(ConfidenceIntervalMethod.StudentizedBootstrap, 1)]
    [InlineData(ConfidenceIntervalMethod.PercentileBootstrap, 5)]
    [InlineData(ConfidenceIntervalMethod.BasicBootstrap, 5)]
    [InlineData(ConfidenceIntervalMethod.BCaBootstrap, 5)]
    [InlineData(ConfidenceIntervalMethod.StudentizedBootstrap, 5)]
    public void HalfEnergyIntervals_ConstantAndSingletonErrorsHaveExactBounds(ConfidenceIntervalMethod method, int count)
    {
        var metric = new EnergyScoreMetric<Half>();
        var result = metric.ComputeWithCI(Enumerable.Repeat((Half)40000, count).ToArray(), new Half[count],
            method, bootstrapSamples: 16, randomSeed: 7);
        Assert.Equal((Half)40000, result.Value);
        Assert.Equal((Half)40000, result.LowerBound);
        Assert.Equal((Half)40000, result.UpperBound);
        var empty = metric.ComputeWithCI(Array.Empty<Half>(), Array.Empty<Half>(), method, bootstrapSamples: 2);
        Assert.Equal((Half)0, empty.LowerBound);
        Assert.Equal((Half)0, empty.UpperBound);
    }

    [Theory]
    [InlineData(ConfidenceIntervalMethod.PercentileBootstrap)]
    [InlineData(ConfidenceIntervalMethod.BasicBootstrap)]
    [InlineData(ConfidenceIntervalMethod.BCaBootstrap)]
    [InlineData(ConfidenceIntervalMethod.StudentizedBootstrap)]
    public void HalfEnergyIntervals_MatchWideWorkingScores(ConfidenceIntervalMethod method)
    {
        Half[] errors = { (Half)20000, (Half)23000, (Half)26000, (Half)29000, (Half)32000 };
        var wide = new EnergyScoreMetric<double>().ComputeWithCI(errors.Select(x => (double)x).ToArray(),
            new double[errors.Length], method, confidenceLevel: 0.8, bootstrapSamples: 64, randomSeed: 17);
        var actual = new EnergyScoreMetric<Half>().ComputeWithCI(errors, new Half[errors.Length], method,
            confidenceLevel: 0.8, bootstrapSamples: 64, randomSeed: 17);
        Assert.Equal((Half)Assert.IsType<double>(wide.LowerBound), actual.LowerBound);
        Assert.Equal((Half)Assert.IsType<double>(wide.UpperBound), actual.UpperBound);
        Assert.True(!Half.IsNaN(Assert.IsType<Half>(actual.LowerBound)));
        Assert.True(!Half.IsInfinity(Assert.IsType<Half>(actual.UpperBound)));
    }

    [Theory]
    [InlineData(ConfidenceIntervalMethod.PercentileBootstrap)]
    [InlineData(ConfidenceIntervalMethod.BasicBootstrap)]
    public void EnergyIntervals_MatchScalarResamplingReference(ConfidenceIntervalMethod method)
    {
        double[] errors = { 10010, 20030, 30000, 40090, 50070 };
        const int samples = 41;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(31);
        var means = new double[samples];
        for (int b = 0; b < samples; b++)
        {
            double sum = 0;
            for (int i = 0; i < errors.Length; i++) sum += errors[random.Next(errors.Length)];
            means[b] = sum / errors.Length;
        }
        Array.Sort(means);
        double low = means[4], high = means[35];
        if (method == ConfidenceIntervalMethod.BasicBootstrap)
            (low, high) = (Math.Max(0, 2 * errors.Average() - high), Math.Max(0, 2 * errors.Average() - low));
        var actual = new EnergyScoreMetric<double>().ComputeWithCI(errors, new double[errors.Length], method,
            confidenceLevel: 0.8, bootstrapSamples: samples, randomSeed: 31);
        Assert.Equal(low, Assert.IsType<double>(actual.LowerBound), 8);
        Assert.Equal(high, Assert.IsType<double>(actual.UpperBound), 8);
    }

    [Fact]
    public void EnergyStudentizedInterval_MatchesScalarNestedBootstrap()
    {
        double[] errors = { 30000, 35000, 45000, 50000 };
        const int count = 32, innerCount = 25;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        // The public method draws the initial outer bootstrap before its Studentized resamples.
        for (int b = 0; b < count; b++)
            for (int i = 0; i < errors.Length; i++) _ = random.Next(errors.Length);
        double theta = errors.Average();
        var jack = Enumerable.Range(0, errors.Length)
            .Select(leave => errors.Where((_, index) => index != leave).Average()).ToArray();
        double jackMean = jack.Average();
        double seHat = Math.Sqrt((errors.Length - 1.0) / errors.Length * jack.Sum(x => (x - jackMean) * (x - jackMean)));
        var t = new double[count];
        for (int b = 0; b < count; b++)
        {
            var outer = Enumerable.Range(0, errors.Length).Select(_ => errors[random.Next(errors.Length)]).ToArray();
            var inner = new double[innerCount];
            for (int ib = 0; ib < innerCount; ib++)
            {
                double sum = 0;
                for (int i = 0; i < errors.Length; i++) sum += outer[random.Next(errors.Length)];
                inner[ib] = sum / errors.Length;
            }
            double mean = inner.Average();
            double se = Math.Max(1e-30, Math.Sqrt(inner.Sum(x => (x - mean) * (x - mean)) / innerCount));
            t[b] = (outer.Average() - theta) / se;
        }
        Array.Sort(t);
        var actual = new EnergyScoreMetric<double>().ComputeWithCI(errors, new double[errors.Length],
            ConfidenceIntervalMethod.StudentizedBootstrap, confidenceLevel: 0.8, bootstrapSamples: count, randomSeed: 42);
        Assert.Equal(Math.Max(0, theta - t[27] * seHat), Assert.IsType<double>(actual.LowerBound), 8);
        Assert.Equal(Math.Max(0, theta - t[3] * seHat), Assert.IsType<double>(actual.UpperBound), 8);
    }

    private sealed class EnergyBootstrapEngine : AiDotNet.Tensors.Engines.CpuEngine
    {
        internal int Products;
        public override Tensor<T> TensorMultiply<T>(Tensor<T> left, Tensor<T> right)
        {
            Products++;
            Assert.Equal(1, left.Shape.Length);
            Assert.Equal(1, right.Shape.Length);
            return base.TensorMultiply(left, right);
        }
    }

    [Fact]
    public void EnergyStudentizedInterval_ReducesOneInnerReplicateBatchAtATime()
    {
        var previous = AiDotNetEngine.Current;
        var engine = new EnergyBootstrapEngine();
        try
        {
            AiDotNetEngine.Current = engine;
            var actual = new EnergyScoreMetric<double>().ComputeWithCI(
                new[] { 10.0, 20, 30, 40, 50 }, new double[5], ConfidenceIntervalMethod.StudentizedBootstrap,
                confidenceLevel: 0.8, bootstrapSamples: 64, randomSeed: 17);
            Assert.True(Assert.IsType<double>(actual.LowerBound) <= Assert.IsType<double>(actual.UpperBound));
            Assert.True(engine.Products > 0);
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void EnergyIntervals_RejectNaNConfidence()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EnergyScoreMetric<double>().ComputeWithCI(
            new[] { 1.0 }, new double[1], confidenceLevel: double.NaN));
    }


    [Fact]
    public void HalfValidationSummaries_KeepLargeCountsAndSampleVarianceWide()
    {
        const System.Reflection.BindingFlags flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic;
        var cv = typeof(CrossValidationEngine<Half>).GetMethod("Summarize", flags) ?? throw new InvalidOperationException("Reflected method not found.");
        var curve = typeof(LearningCurveEngine<Half>).GetMethod("MeanAndStandardDeviation", flags) ?? throw new InvalidOperationException("Reflected method not found.");
        var values = Enumerable.Repeat((Half)60000, 70000).ToList();
        Assert.Equal((60000.0, 0.0, 0.0), Assert.IsType<(double, double, double)>(cv.Invoke(null, new object[] { values })));
        Assert.Equal((60000.0, 0.0), Assert.IsType<(double, double)>(curve.Invoke(null, new object[] { values })));
        values = new List<Half> { (Half)(-60000), (Half)60000 };
        var (mean, std, margin) = Assert.IsType<(double, double, double)>(cv.Invoke(null, new object[] { values }));
        Assert.Equal(0.0, mean);
        Assert.Equal(Math.Sqrt(7200000000.0), std, 8);
        Assert.Equal(60000 * 12.706204736432095, margin, 2);
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
        var method = typeof(DeLongTest<double>).GetMethod("ComputeComponents", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic) ?? throw new InvalidOperationException("Reflected method not found.");
        var (auc, rows, columns) = Assert.IsType<(double, Tensor<double>, Tensor<double>)>(method.Invoke(null, new object[] { scores, positiveMask, negativeMask }));
        var rowSums = new double[positives];
        var columnSums = new double[negatives];
        double total = 0;
        for (int i = 0; i < positives; i++)
        for (int j = 0; j < negatives; j++)
        {
            double left = scores[i], right = scores[positives + j];
            double kernel = left > right ? 1 : (left == right || Math.Abs(left - right) < 1e-10) ? 0.5 : 0;
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
        Assert.Equal((group + 1.0) / 2, ranks.GetFlat(0));
        Assert.Equal((group + 1.0 + size) / 2, ranks.GetFlat(size - 1));
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
            double ascendingRank = (start + 1.0 + end) / 2;
            double descendingRank = (size - end + 1.0 + size - start) / 2;
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


    [Fact]
    public void HalfRepeatedDatasetRanks_StayFiniteThroughLargeAverages()
    {
        const int count = 70000;
        var samples = new[]
        {
            Enumerable.Range(0, count).Select(i => i % 2 == 0 ? (Half)3 : (Half)1).ToArray(),
            Enumerable.Repeat((Half)2, count).ToArray(),
            Enumerable.Range(0, count).Select(i => i % 2 == 0 ? (Half)1 : (Half)3).ToArray()
        };
        var friedman = new FriedmanTest<Half>().Test(samples);
        Assert.Equal((Half)0, friedman.Statistic);
        Assert.Equal((Half)1, friedman.PValue);
        var nemenyi = new NemenyiPostHocTest<Half>().Test(samples);
        Assert.Equal(new[] { 2.0, 2.0, 2.0 }, nemenyi.AverageRanks);
        Assert.Empty(nemenyi.GetSignificantPairs());
    }

    [Fact]
    public void HalfKruskalAndDunn_KeepLargeGroupSizesAndTiedRanksWide()
    {
        var values = Enumerable.Repeat((Half)1, 70000).ToArray();
        var groups = new[] { values, values };
        var test = new KruskalWallisTest<Half>();
        var result = test.Test(groups);
        Assert.Equal((Half)0, result.Statistic);
        Assert.Equal((Half)1, result.PValue);
        Assert.Equal(1.0, test.DunnPostHoc(groups)[(0, 1)]);
        var ordered = test.Test(new[]
        {
            new[] { (Half)1, (Half)2, (Half)3 },
            new[] { (Half)4, (Half)5, (Half)6 }
        });
        Assert.Equal((Half)(27.0 / 7), ordered.Statistic);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void HalfLevene_PreservesCentersSizesAndVariance(int center)
    {
        var test = new LeveneTest<Half>((LeveneTest<Half>.CenterType)center);
        Half[][] groups =
        {
            new[] { (Half)(-60000), (Half)(-30000), (Half)0, (Half)30000, (Half)60000 },
            new[] { (Half)(-30000), (Half)(-15000), (Half)0, (Half)15000, (Half)30000 }
        };
        Assert.Equal((Half)(72.0 / 35), test.Test(groups).Statistic);
        var constant = Enumerable.Repeat((Half)60000, 70000).ToArray();
        var identical = test.Test(new[] { constant, constant });
        Assert.Equal((Half)0, identical.Statistic);
        Assert.Equal((Half)1, identical.PValue);
        groups = new[]
        {
            new[] { (Half)59968, (Half)60000, (Half)60032, (Half)60064 },
            new[] { (Half)59904, (Half)60000, (Half)60096, (Half)60192 }
        };
        var expected = new LeveneTest<double>((LeveneTest<double>.CenterType)center)
            .Test(groups.Select(g => g.Select(x => (double)x).ToArray()).ToArray());
        var actual = test.Test(groups);
        Assert.Equal((Half)expected.Statistic, actual.Statistic);
        Assert.Equal((Half)expected.PValue, actual.PValue);
    }

    [Fact]
    public void HalfWilcoxon_PreservesWideDifferencesAndRankSums()
    {
        var test = new WilcoxonSignedRankTest<Half>();
        var positive = Enumerable.Repeat((Half)1, 512).ToArray();
        positive[0] = (Half)(-1);
        var ranked = test.Test(positive, new Half[positive.Length]);
        Assert.Equal((Half)256.5, ranked.Statistic);
        Assert.Equal((Half)(510.0 / 512), ranked.EffectSize);
        Assert.Equal((Half)0, ranked.PValue);
        var extremes = test.Test(
            new[] { (Half)60000, (Half)60000, (Half)60000, (Half)(-60000), (Half)(-60000) },
            new[] { (Half)(-60000), (Half)(-60000), (Half)(-60000), (Half)60000, (Half)60000 });
        Assert.Equal((Half)6, extremes.Statistic);
        Assert.Equal((Half)0.2, extremes.EffectSize);
    }

    [Fact]
    public void HalfBootstrap_KeepsLargeCountsAndDoubleConfidenceEndpoints()
    {
        const int count = 70000;
        var first = Enumerable.Repeat((Half)60000, count).ToArray();
        var second = new Half[count];
        var test = new BootstrapTest<Half>(16, 42);
        var result = test.Test(first, second);
        Assert.Equal((Half)60000, result.Statistic);
        Assert.Equal((Half)(1.0 / 17), result.PValue);
        Assert.Equal((Half)0, result.EffectSize);
        second = Enumerable.Repeat((Half)(-60000), count).ToArray();
        Assert.Equal((120000.0, 120000.0), test.ComputeCI(first, second));
    }

    [Fact]
    public void HalfBootstrap_ZeroObservedDifferenceIncludesAllFinitePermutations()
    {
        var test = new BootstrapTest<Half>(64, 17);
        var result = test.Test(
            new[] { (Half)(-60000), (Half)(-30000), (Half)0, (Half)30000, (Half)60000 },
            new[] { (Half)60000, (Half)0, (Half)0, (Half)0, (Half)(-60000) });
        Assert.Equal((Half)0, result.Statistic);
        Assert.Equal((Half)1, result.PValue);
        Assert.Equal((Half)0, result.EffectSize);
    }


    [Fact]
    public void HalfIsotonicInterpolation_KeepsSubnormalProductsWide()
    {
        Half end = (Half)(2 * (double)Half.Epsilon);
        var calibrator = new ProbabilityCalibrator<Half>(new ProbabilityCalibratorOptions
            { CalibratorMethod = ProbabilityCalibrationMethod.IsotonicRegression });
        calibrator.Fit(new Vector<Half>(new[] { (Half)0, end, end }),
            new Vector<Half>(new[] { (Half)0, (Half)1, (Half)0 }));
        Assert.Equal((Half)0.25, calibrator.Transform(new Vector<Half>(new[] { Half.Epsilon }))[0]);
    }

    [Fact]
    public void DecimalIsotonicInterpolation_PreservesCloseLargeCoordinates()
    {
        const decimal start = 10000000000000000m;
        var calibrator = new ProbabilityCalibrator<decimal>(new ProbabilityCalibratorOptions
            { CalibratorMethod = ProbabilityCalibrationMethod.IsotonicRegression });
        calibrator.Fit(new Vector<decimal>(new[] { start, start + 2 }),
            new Vector<decimal>(new[] { 0m, 1m }));
        Assert.Equal(0.5m, calibrator.Transform(new Vector<decimal>(new[] { start + 1 }))[0]);
    }

    [Fact]
    public void FloatHistogram_FitAndLookupUseTheSameUnroundedEdges()
    {
        var calibrator = new ProbabilityCalibrator<float>(new ProbabilityCalibratorOptions
            { CalibratorMethod = ProbabilityCalibrationMethod.HistogramBinning, NumBins = 10 });
        var fitted = calibrator.FitTransform(new Vector<float>(new[] { 0.7f, 0.75f }),
            new Vector<float>(new[] { 1f, 0f }));
        Assert.Equal(new[] { 1f, 0f }, fitted.ToArray());
        Assert.Equal(new[] { 1f, 0f }, calibrator.Transform(new Vector<float>(new[] { 0.7f, 0.7001f })).ToArray());
    }

    [Fact]
    public void ErrorRate_CountsMismatchesOutsideTheBinaryLabels()
    {
        var metric = new ErrorRateMetric<double>();
        Assert.Equal(1.0, metric.Compute(new[] { 2.0 }, new[] { 3.0 }));
        Assert.Equal(2.0 / 3, metric.Compute(new[] { 2.0, 3.0, 4.0 }, new[] { 3.0, 3.0, 5.0 }));
        Assert.Equal(1.0 / 3, metric.Compute(new[] { 0.0, 1.0, 1.0 }, new[] { 0.0, 0.0, 1.0 }));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(0.50001)]
    public void LogLoss_RejectsInvalidClippingIntervals(double epsilon)
    {
        Assert.Equal("epsilon", Assert.Throws<ArgumentOutOfRangeException>(() => new LogLossMetric<double>(epsilon)).ParamName);
    }

    [Fact]
    public void LogLoss_AllowsTheDegenerateHalfProbabilityInterval()
    {
        Assert.Equal(Math.Log(2), new LogLossMetric<double>(0.5)
            .Compute(new[] { 0.0, 1.0 }, new[] { 0.0, 1.0 }), 12);
    }

    [Fact]
    public void HalfPointEnergyAndAlternateMape_ReduceWideAndRetainFloors()
    {
        var energy = new EnergyScoreMetric<Half>();
        Assert.Equal((Half)100, energy.Compute(Enumerable.Repeat((Half)100, 1000).ToArray(), new Half[1000]));
        Assert.Equal((Half)40000, energy.Compute(new[] { (Half)60000, (Half)0, (Half)0 },
            new[] { (Half)(-60000), (Half)0, (Half)0 }));
        var mape = new MeanAbsolutePercentageErrorMetric<Half>();
        Assert.Equal((Half)0, mape.Compute(new Half[1], new Half[1]));
        Assert.Equal((Half)(100 * (double)Half.Epsilon / 1e-10), mape.Compute(new[] { Half.Epsilon }, new Half[1]));
        Assert.Equal((Half)50, mape.Compute(Enumerable.Repeat((Half)1, 70000).ToArray(),
            Enumerable.Repeat((Half)2, 70000).ToArray()));
    }

    [Fact]
    public void HalfMultiClassAucRoc_AveragesManyClassScoresWide()
    {
        // Each class has two positives: one scoring 1 against all-zero negatives (AUC 1) and one tied with them
        // (AUC 1/2), so every one-vs-rest AUC is 3/4 and so is the macro average (sklearn ovr, macro).
        const int classes = 1400;
        int rows = 2 * classes;
        var actuals = new Half[rows];
        var probabilities = new Half[rows * classes];
        for (int r = 0; r < rows; r++)
        {
            int label = r / 2;
            actuals[r] = (Half)label;
            if (r % 2 == 0) probabilities[r * classes + label] = (Half)1;
        }

        Assert.Equal((Half)0.75, new AUCROCMetric<Half>().Compute(probabilities, actuals, classes));
    }

    [Fact]
    public void Levene_RejectsInvalidGroupsAtThePublicBoundary()
    {
        var test = new LeveneTest<Half>();
        Half[][]? missingGroups = null;
        Half[]?[] missingGroup = { null, new[] { (Half)1 } };
        // The nulls are deliberate: this checks the runtime guard that nullable annotations cannot enforce.
#pragma warning disable CS8604, CS8620
        Assert.Equal("groups", Assert.Throws<ArgumentNullException>(() => test.Test(missingGroups)).ParamName);
        Assert.Equal("groups", Assert.Throws<ArgumentException>(() =>
            test.Test(new[] { Array.Empty<Half>(), new[] { (Half)1 } })).ParamName);
        Assert.Equal("groups", Assert.Throws<ArgumentException>(() => test.Test(missingGroup)).ParamName);
#pragma warning restore CS8604, CS8620
    }

    [Fact]
    public void DeLong_IdenticalInfinitiesContributeHalfAPair()
    {
        double[] scores = { double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, double.NegativeInfinity };
        var positives = new Tensor<double>(new[] { 1.0, 1.0, 0.0, 0.0 }, new[] { 4 });
        var negatives = new Tensor<double>(new[] { 0.0, 0.0, 1.0, 1.0 }, new[] { 4 });
        var method = typeof(DeLongTest<double>).GetMethod("ComputeComponents",
            System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic) ?? throw new InvalidOperationException("Reflected method not found.");
        var (auc, rows, columns) = Assert.IsType<(double, Tensor<double>, Tensor<double>)>(
            method.Invoke(null, new object[] { scores, positives, negatives }));
        Assert.Equal(0.5, auc);
        Assert.Equal(new[] { 0.75, 0.25 }, rows.ToArray());
        Assert.Equal(new[] { 0.25, 0.75 }, columns.ToArray());
    }

    [Theory]
    [InlineData(1.0, 2.0)]
    [InlineData(2.0, 0.0)]
    public void DecimalTweedie_DoesNotEvaluateAnInvalidMaskedLogarithm(double power, double expected)
    {
        Assert.Equal((decimal)expected, new TweedieLossMetric<decimal>(power).Compute(new[] { 1m }, new[] { 0m }));
    }

}
#endif
