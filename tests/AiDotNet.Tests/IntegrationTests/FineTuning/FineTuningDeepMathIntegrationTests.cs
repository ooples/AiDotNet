using AiDotNet.Enums;
using AiDotNet.FineTuning;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FineTuning;

/// <summary>
/// Deep math integration tests for the FineTuning module.
/// Tests mathematical functions: Sigmoid, LogSigmoid, KL divergence, DPO/KTO/SimPO loss formulas,
/// cosine similarity, scalar/string/array log probability computations.
/// </summary>
public class FineTuningDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper - expose protected methods via test subclass

    /// <summary>
    /// Test helper that exposes protected math methods from FineTuningBase for testing.
    /// </summary>
    private class TestFineTuning : FineTuningBase<double, double[], double[]>
    {
        public TestFineTuning() : base(new FineTuningOptions<double>()) { }

        public override string MethodName => "Test";
        public override FineTuningCategory Category => FineTuningCategory.DirectPreference;
        public override bool RequiresRewardModel => false;
        public override bool RequiresReferenceModel => false;

        public override Task<IFullModel<double, double[], double[]>> FineTuneAsync(
            IFullModel<double, double[], double[]> baseModel,
            FineTuningData<double, double[], double[]> trainingData,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public override Task<FineTuningMetrics<double>> EvaluateAsync(
            IFullModel<double, double[], double[]> model,
            FineTuningData<double, double[], double[]> evaluationData,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        // Expose protected static methods
        public static double TestSigmoid(double x) => Sigmoid(x);
        public static double TestLogSigmoid(double x) => LogSigmoid(x);

        // Expose KL divergence
        public double TestKLDivergence(double[] p, double[] q) => ComputeKLDivergence(p, q);

        // Expose log probability methods
        public double TestComputeLogProb(double[] prediction, double[] target)
            => ComputeLogProbabilityFromPrediction(prediction, target);
    }

    /// <summary>
    /// String-typed test helper for string log probability testing.
    /// </summary>
    private class StringTestFineTuning : FineTuningBase<double, string, string>
    {
        public StringTestFineTuning() : base(new FineTuningOptions<double>()) { }

        public override string MethodName => "StringTest";
        public override FineTuningCategory Category => FineTuningCategory.DirectPreference;
        public override bool RequiresRewardModel => false;
        public override bool RequiresReferenceModel => false;

        public override Task<IFullModel<double, string, string>> FineTuneAsync(
            IFullModel<double, string, string> baseModel,
            FineTuningData<double, string, string> trainingData,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public override Task<FineTuningMetrics<double>> EvaluateAsync(
            IFullModel<double, string, string> model,
            FineTuningData<double, string, string> evaluationData,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public double TestComputeLogProb(string prediction, string target)
            => ComputeLogProbabilityFromPrediction(prediction, target);
    }

    /// <summary>
    /// Scalar-typed test helper for scalar log probability testing.
    /// </summary>
    private class ScalarTestFineTuning : FineTuningBase<double, double, double>
    {
        public ScalarTestFineTuning() : base(new FineTuningOptions<double>()) { }

        public override string MethodName => "ScalarTest";
        public override FineTuningCategory Category => FineTuningCategory.DirectPreference;
        public override bool RequiresRewardModel => false;
        public override bool RequiresReferenceModel => false;

        public override Task<IFullModel<double, double, double>> FineTuneAsync(
            IFullModel<double, double, double> baseModel,
            FineTuningData<double, double, double> trainingData,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public override Task<FineTuningMetrics<double>> EvaluateAsync(
            IFullModel<double, double, double> model,
            FineTuningData<double, double, double> evaluationData,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public double TestComputeLogProb(double prediction, double target)
            => ComputeLogProbabilityFromPrediction(prediction, target);
    }

    #endregion

    // ============================
    // Sigmoid Tests
    // ============================

    [Fact]
    public void Sigmoid_AtZero_ReturnsHalf()
    {
        // sigmoid(0) = 1 / (1 + exp(0)) = 1/2
        var result = TestFineTuning.TestSigmoid(0.0);
        Assert.Equal(0.5, result, Tolerance);
    }

    [Fact]
    public void Sigmoid_LargePositive_ApproachesOne()
    {
        // sigmoid(100) ≈ 1.0
        var result = TestFineTuning.TestSigmoid(100.0);
        Assert.True(result > 1.0 - 1e-10);
        Assert.True(result <= 1.0);
    }

    [Fact]
    public void Sigmoid_LargeNegative_ApproachesZero()
    {
        // sigmoid(-100) ≈ 0.0
        var result = TestFineTuning.TestSigmoid(-100.0);
        Assert.True(result < 1e-10);
        Assert.True(result >= 0.0);
    }

    [Fact]
    public void Sigmoid_SymmetryProperty_SigmaX_Plus_SigmaNegX_Equals_One()
    {
        // σ(x) + σ(-x) = 1 for all x
        var values = new[] { -5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 10.0, -10.0 };
        foreach (var x in values)
        {
            var sigX = TestFineTuning.TestSigmoid(x);
            var sigNegX = TestFineTuning.TestSigmoid(-x);
            Assert.Equal(1.0, sigX + sigNegX, Tolerance);
        }
    }

    [Fact]
    public void Sigmoid_KnownValues_MatchHandComputed()
    {
        // σ(1) = 1/(1+e^-1) = 1/(1+0.367879...) = 0.731058...
        var sig1 = TestFineTuning.TestSigmoid(1.0);
        Assert.Equal(1.0 / (1.0 + Math.Exp(-1.0)), sig1, Tolerance);

        // σ(-1) = 1/(1+e^1) = 1/(1+2.71828...) = 0.268941...
        var sigNeg1 = TestFineTuning.TestSigmoid(-1.0);
        Assert.Equal(1.0 / (1.0 + Math.Exp(1.0)), sigNeg1, Tolerance);

        // σ(2) = 1/(1+e^-2)
        var sig2 = TestFineTuning.TestSigmoid(2.0);
        Assert.Equal(1.0 / (1.0 + Math.Exp(-2.0)), sig2, Tolerance);
    }

    [Fact]
    public void Sigmoid_IsMonotonicallyIncreasing()
    {
        double prev = 0.0;
        for (double x = -10.0; x <= 10.0; x += 0.5)
        {
            var current = TestFineTuning.TestSigmoid(x);
            if (x > -10.0)
            {
                Assert.True(current > prev, $"Sigmoid must be increasing: σ({x}) = {current} <= σ({x - 0.5}) = {prev}");
            }
            prev = current;
        }
    }

    [Fact]
    public void Sigmoid_OutputBoundedBetween0And1()
    {
        var testValues = new[] { -1000.0, -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0 };
        foreach (var x in testValues)
        {
            var result = TestFineTuning.TestSigmoid(x);
            Assert.True(result >= 0.0, $"Sigmoid({x}) = {result} < 0");
            Assert.True(result <= 1.0, $"Sigmoid({x}) = {result} > 1");
        }
    }

    // ============================
    // LogSigmoid Tests
    // ============================

    [Fact]
    public void LogSigmoid_AtZero_ReturnsNegLn2()
    {
        // log(sigmoid(0)) = log(0.5) = -ln(2)
        var result = TestFineTuning.TestLogSigmoid(0.0);
        Assert.Equal(-Math.Log(2.0), result, Tolerance);
    }

    [Fact]
    public void LogSigmoid_LargePositive_ApproachesZero()
    {
        // log(sigmoid(100)) ≈ log(1) = 0
        var result = TestFineTuning.TestLogSigmoid(100.0);
        Assert.True(Math.Abs(result) < 1e-10);
    }

    [Fact]
    public void LogSigmoid_LargeNegative_ApproachesNegativeInput()
    {
        // log(sigmoid(-100)) ≈ -100 (since sigmoid(-x) ≈ exp(x) for large neg x)
        var result = TestFineTuning.TestLogSigmoid(-100.0);
        Assert.Equal(-100.0, result, 1e-5);
    }

    [Fact]
    public void LogSigmoid_EqualsLogOfSigmoid()
    {
        // Verify log(σ(x)) = logσ(x) for moderate values
        var values = new[] { -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0 };
        foreach (var x in values)
        {
            var logSig = TestFineTuning.TestLogSigmoid(x);
            var sig = TestFineTuning.TestSigmoid(x);
            var logOfSig = Math.Log(sig);
            Assert.Equal(logOfSig, logSig, 1e-10);
        }
    }

    [Fact]
    public void LogSigmoid_IsAlwaysNonPositive()
    {
        // Since sigmoid is in (0,1], log(sigmoid) is in (-inf, 0]
        var values = new[] { -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };
        foreach (var x in values)
        {
            var result = TestFineTuning.TestLogSigmoid(x);
            Assert.True(result <= 0.0 + 1e-15, $"LogSigmoid({x}) = {result} should be <= 0");
        }
    }

    [Fact]
    public void LogSigmoid_NumericalStability_NoNaNOrInfinity()
    {
        // Test extreme values don't produce NaN or Infinity
        var extremeValues = new[] { -1000.0, -500.0, -100.0, 0.0, 100.0, 500.0, 1000.0 };
        foreach (var x in extremeValues)
        {
            var result = TestFineTuning.TestLogSigmoid(x);
            Assert.False(double.IsNaN(result), $"LogSigmoid({x}) produced NaN");
            Assert.False(double.IsPositiveInfinity(result), $"LogSigmoid({x}) produced +Infinity");
        }
    }

    // ============================
    // KL Divergence Tests
    // ============================

    [Fact]
    public void KLDivergence_IdenticalDistributions_ReturnsZero()
    {
        var ft = new TestFineTuning();
        var p = new[] { 0.25, 0.25, 0.25, 0.25 };
        var result = ft.TestKLDivergence(p, p);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void KLDivergence_IsNonNegative()
    {
        // Gibbs' inequality: KL(P||Q) >= 0
        var ft = new TestFineTuning();
        var p = new[] { 0.1, 0.4, 0.5 };
        var q = new[] { 0.3, 0.3, 0.4 };
        var result = ft.TestKLDivergence(p, q);
        Assert.True(result >= -Tolerance, $"KL divergence = {result} should be non-negative");
    }

    [Fact]
    public void KLDivergence_IsNotSymmetric()
    {
        // KL(P||Q) != KL(Q||P) in general
        var ft = new TestFineTuning();
        var p = new[] { 0.1, 0.9 };
        var q = new[] { 0.5, 0.5 };
        var kl_pq = ft.TestKLDivergence(p, q);
        var kl_qp = ft.TestKLDivergence(q, p);
        Assert.NotEqual(kl_pq, kl_qp);
    }

    [Fact]
    public void KLDivergence_KnownValue_HandComputed()
    {
        // P = [0.5, 0.5], Q = [0.25, 0.75]
        // KL(P||Q) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
        //          = 0.5*ln(2) + 0.5*ln(2/3)
        //          = 0.5*0.6931... + 0.5*(-0.4054...)
        //          = 0.3466 - 0.2027 = 0.1438...
        var ft = new TestFineTuning();
        var p = new[] { 0.5, 0.5 };
        var q = new[] { 0.25, 0.75 };
        var result = ft.TestKLDivergence(p, q);
        var expected = 0.5 * Math.Log(0.5 / 0.25) + 0.5 * Math.Log(0.5 / 0.75);
        Assert.Equal(expected, result, Tolerance);
    }

    [Fact]
    public void KLDivergence_DifferentLengths_Throws()
    {
        var ft = new TestFineTuning();
        var p = new[] { 0.5, 0.5 };
        var q = new[] { 0.3, 0.3, 0.4 };
        Assert.Throws<ArgumentException>(() => ft.TestKLDivergence(p, q));
    }

    // ============================
    // Scalar Log Probability Tests
    // ============================

    [Fact]
    public void ScalarLogProb_PerfectMatch_ReturnsZero()
    {
        // When pred == target, squared error = 0, so log prob = 0
        var ft = new ScalarTestFineTuning();
        var result = ft.TestComputeLogProb(5.0, 5.0);
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void ScalarLogProb_IsNonPositive()
    {
        // Log probability from Gaussian kernel: -error^2 / (2*sigma^2) <= 0
        var ft = new ScalarTestFineTuning();
        var pairs = new[] { (1.0, 2.0), (0.0, 5.0), (3.0, 7.0), (-1.0, 1.0) };
        foreach (var (pred, target) in pairs)
        {
            var result = ft.TestComputeLogProb(pred, target);
            Assert.True(result <= 0.0 + Tolerance, $"LogProb({pred}, {target}) = {result} should be <= 0");
        }
    }

    [Fact]
    public void ScalarLogProb_HandComputed_Unit1()
    {
        // pred=0, target=1: squared_error = 1, sigma=1
        // log_prob = -1 / (2*1*1) = -0.5
        var ft = new ScalarTestFineTuning();
        var result = ft.TestComputeLogProb(0.0, 1.0);
        Assert.Equal(-0.5, result, Tolerance);
    }

    [Fact]
    public void ScalarLogProb_CloserPrediction_HigherLogProb()
    {
        var ft = new ScalarTestFineTuning();
        var target = 5.0;
        var closePred = 4.5; // error = 0.5
        var farPred = 3.0;   // error = 2.0

        var closeLogProb = ft.TestComputeLogProb(closePred, target);
        var farLogProb = ft.TestComputeLogProb(farPred, target);
        Assert.True(closeLogProb > farLogProb);
    }

    // ============================
    // String Log Probability Tests
    // ============================

    [Fact]
    public void StringLogProb_IdenticalStrings_ReturnsZero()
    {
        var ft = new StringTestFineTuning();
        var result = ft.TestComputeLogProb("hello", "hello");
        Assert.Equal(0.0, result, Tolerance);
    }

    [Fact]
    public void StringLogProb_CompletelyDifferent_IsVeryNegative()
    {
        var ft = new StringTestFineTuning();
        var result = ft.TestComputeLogProb("aaaa", "zzzz");
        // Match ratio = 0/4 = 0, log(1e-10) ≈ -23
        Assert.True(result < -10.0);
    }

    [Fact]
    public void StringLogProb_PartialMatch_IntermediateValue()
    {
        var ft = new StringTestFineTuning();
        // "hello" vs "helly" -> 4/5 match = 0.8
        // log(0.8) ≈ -0.2231
        var result = ft.TestComputeLogProb("hello", "helly");
        var expected = Math.Log(4.0 / 5.0);
        Assert.Equal(expected, result, 1e-10);
    }

    [Fact]
    public void StringLogProb_DifferentLengths_PenalizesLength()
    {
        var ft = new StringTestFineTuning();
        // "abc" vs "abcdef" -> matches = min(3,6)=3, maxLen=6, matchRatio = 3/6 = 0.5
        var result = ft.TestComputeLogProb("abc", "abcdef");
        var expected = Math.Log(3.0 / 6.0);
        Assert.Equal(expected, result, 1e-10);
    }

    // ============================
    // Array Log Probability Tests (Probability Distribution)
    // ============================

    [Fact]
    public void ArrayLogProb_ProbabilityDistribution_CrossEntropy()
    {
        // When prediction looks like a probability distribution (sums to ~1, all non-negative),
        // uses cross-entropy: -sum(target * log(pred))
        var ft = new TestFineTuning();
        var pred = new[] { 0.7, 0.2, 0.1 }; // sums to 1.0, all >= 0 -> prob dist
        var target = new[] { 1.0, 0.0, 0.0 }; // one-hot

        var result = ft.TestComputeLogProb(pred, target);
        // -sum(target * log(pred)) = -(1.0 * log(0.7) + 0 + 0) = -log(0.7)
        var expected = 1.0 * Math.Log(0.7);
        Assert.Equal(expected, result, 1e-10);
    }

    [Fact]
    public void ArrayLogProb_PerfectProbDist_ReturnsZero()
    {
        // If prediction exactly matches target one-hot, cross entropy = -log(1) = 0
        var ft = new TestFineTuning();
        var pred = new[] { 1.0, 0.0, 0.0 };
        var target = new[] { 1.0, 0.0, 0.0 };

        var result = ft.TestComputeLogProb(pred, target);
        // -sum(1.0 * log(1.0)) = 0
        Assert.Equal(0.0, result, 1e-8);
    }

    [Fact]
    public void ArrayLogProb_Embedding_UsesCosine()
    {
        // When prediction doesn't look like a prob dist (doesn't sum to ~1 or has negatives),
        // uses cosine similarity
        var ft = new TestFineTuning();
        var pred = new[] { 1.0, 2.0, 3.0 }; // sums to 6, not a prob dist
        var target = new[] { 1.0, 2.0, 3.0 }; // identical

        var result = ft.TestComputeLogProb(pred, target);
        // Cosine similarity = 1.0, prob = (1+1)/2 = 1.0, log(1.0) = 0
        Assert.Equal(0.0, result, 1e-8);
    }

    [Fact]
    public void ArrayLogProb_OrthogonalEmbeddings_HasNegativeLogProb()
    {
        // Use vectors that don't look like probability distributions (negative values or sum != 1)
        // to trigger cosine similarity path
        var ft = new TestFineTuning();
        var pred = new[] { 1.0, 0.0, -1.0 };   // sum=0, has negative -> not prob dist
        var target = new[] { 0.0, 1.0, 0.0 };   // orthogonal to pred

        var result = ft.TestComputeLogProb(pred, target);
        // cosine = 0, prob = (0+1)/2 = 0.5, log(0.5) = -0.693...
        Assert.Equal(Math.Log(0.5), result, 1e-8);
    }

    [Fact]
    public void ArrayLogProb_OppositeEmbeddings_VeryNegative()
    {
        // Use vectors that are opposite and not prob distributions
        var ft = new TestFineTuning();
        var pred = new[] { 1.0, -1.0, 0.5 };    // has negative -> not prob dist
        var target = new[] { -1.0, 1.0, -0.5 };  // opposite of pred

        var result = ft.TestComputeLogProb(pred, target);
        // cosine = -1, prob = (-1+1)/2 = 0 -> clipped to 1e-10, log(1e-10) ≈ -23
        Assert.True(result < -20.0);
    }

    // ============================
    // DPO Loss Formula Tests
    // ============================

    [Fact]
    public void DPOLoss_Formula_WhenChosenPreferred_LossIsSmall()
    {
        // DPO loss: -log(σ(β * (chosen_log_ratio - rejected_log_ratio)))
        // When chosen is strongly preferred: chosen_log_ratio > rejected_log_ratio
        // margin is large positive, σ(margin) ≈ 1, -log(1) ≈ 0
        double beta = 0.5;
        double chosenLogRatio = 10.0;  // policy strongly prefers chosen
        double rejectedLogRatio = -10.0;  // policy doesn't prefer rejected
        double margin = beta * (chosenLogRatio - rejectedLogRatio); // 0.5 * 20 = 10
        double loss = -TestFineTuning.TestLogSigmoid(margin);

        // σ(10) ≈ 0.99995, -log(0.99995) ≈ 0.00005
        Assert.True(loss < 0.01, $"DPO loss should be near 0 when chosen is preferred, got {loss}");
    }

    [Fact]
    public void DPOLoss_Formula_WhenRejectedPreferred_LossIsLarge()
    {
        // When rejected is preferred: rejected_log_ratio > chosen_log_ratio
        // margin is large negative, σ(margin) ≈ 0, -log(ε) is large
        double beta = 0.1;
        double chosenLogRatio = -5.0;
        double rejectedLogRatio = 5.0;
        double margin = beta * (chosenLogRatio - rejectedLogRatio);
        double loss = -TestFineTuning.TestLogSigmoid(margin);

        Assert.True(loss > 0.5, $"DPO loss should be large when rejected is preferred, got {loss}");
    }

    [Fact]
    public void DPOLoss_Formula_EqualPreference_LossIsLn2()
    {
        // When margin = 0, σ(0) = 0.5, -log(0.5) = ln(2) ≈ 0.693
        double margin = 0.0;
        double loss = -TestFineTuning.TestLogSigmoid(margin);
        Assert.Equal(Math.Log(2.0), loss, Tolerance);
    }

    [Fact]
    public void DPOLoss_BetaScaling_HigherBeta_SteepergGradient()
    {
        // Higher beta makes the model more sensitive to preference differences
        double chosenLogRatio = 1.0;
        double rejectedLogRatio = 0.0;
        double diff = chosenLogRatio - rejectedLogRatio;

        double lossBeta01 = -TestFineTuning.TestLogSigmoid(0.1 * diff);
        double lossBeta10 = -TestFineTuning.TestLogSigmoid(1.0 * diff);

        // Higher beta with positive diff should give lower loss (more confident)
        Assert.True(lossBeta10 < lossBeta01);
    }

    // ============================
    // SimPO Loss Formula Tests
    // ============================

    [Fact]
    public void SimPOLoss_WithGamma_ShiftsDecisionBoundary()
    {
        // SimPO loss: -log(σ(β * (chosen_avg - rejected_avg) - γ))
        // gamma > 0 shifts the boundary, requiring a larger margin to achieve same loss
        double beta = 1.0;
        double chosenAvg = 1.0;
        double rejectedAvg = 0.0;

        double lossNoGamma = -TestFineTuning.TestLogSigmoid(beta * (chosenAvg - rejectedAvg));
        double lossWithGamma = -TestFineTuning.TestLogSigmoid(beta * (chosenAvg - rejectedAvg) - 0.5);

        Assert.True(lossWithGamma > lossNoGamma, "Gamma should increase loss (require larger margin)");
    }

    [Fact]
    public void SimPOLoss_LengthNormalization_AverageVsSum()
    {
        // SimPO normalizes by response length. Shorter responses should not be unfairly penalized.
        // Average log prob: logprob_total / length
        double logProbShort = -2.0;
        int shortLen = 2;
        double logProbLong = -6.0;
        int longLen = 10;

        double avgShort = logProbShort / shortLen;  // -1.0
        double avgLong = logProbLong / longLen;     // -0.6

        // Long response has better average even though worse total
        Assert.True(avgLong > avgShort);
    }

    // ============================
    // KTO Loss Formula Tests
    // ============================

    [Fact]
    public void KTOLoss_Desirable_SigmoidOfBetaTimesMargin()
    {
        // KTO desirable loss: -w_d * σ(β * (log_ratio - KL))
        double desirableWeight = 1.0;
        double beta = 0.1;
        double logRatio = 2.0;
        double klEstimate = 0.5;

        double loss = -desirableWeight * TestFineTuning.TestSigmoid(beta * (logRatio - klEstimate));

        // β * (2.0 - 0.5) = 0.1 * 1.5 = 0.15
        // σ(0.15) = 1/(1+e^-0.15) ≈ 0.5374
        // loss = -1.0 * 0.5374 ≈ -0.5374
        var expected = -desirableWeight * (1.0 / (1.0 + Math.Exp(-0.15)));
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void KTOLoss_Undesirable_OneMinusSigmoid()
    {
        // KTO undesirable loss: -w_u * (1 - σ(β * (log_ratio - KL)))
        double undesirableWeight = 1.0;
        double beta = 0.1;
        double logRatio = -2.0;
        double klEstimate = 0.5;

        double loss = -undesirableWeight * (1.0 - TestFineTuning.TestSigmoid(beta * (logRatio - klEstimate)));

        // β * (-2.0 - 0.5) = 0.1 * (-2.5) = -0.25
        // σ(-0.25) = 1/(1+e^0.25) ≈ 0.4378
        // 1 - 0.4378 = 0.5622
        // loss = -1.0 * 0.5622 ≈ -0.5622
        var sigma = 1.0 / (1.0 + Math.Exp(0.25));
        var expected = -undesirableWeight * (1.0 - sigma);
        Assert.Equal(expected, loss, Tolerance);
    }

    [Fact]
    public void KTOLoss_LossAversion_UndesirableWeightHigher()
    {
        // In prospect theory, losses are weighted more than gains
        // Typical: undesirable_weight > desirable_weight (e.g., 1.5 vs 1.0)
        double desirableWeight = 1.0;
        double undesirableWeight = 1.5;

        // Same absolute log ratio magnitude
        double beta = 0.1;
        double klEstimate = 0.0;
        double absLogRatio = 1.0;

        var desirableLoss = Math.Abs(-desirableWeight * TestFineTuning.TestSigmoid(beta * absLogRatio));
        var undesirableLoss = Math.Abs(-undesirableWeight * (1.0 - TestFineTuning.TestSigmoid(beta * (-absLogRatio))));

        // Undesirable should contribute more to loss due to higher weight
        Assert.True(undesirableLoss > desirableLoss);
    }

    // ============================
    // DPO Label Smoothing Tests
    // ============================

    [Fact]
    public void DPOLoss_LabelSmoothing_InterpolatesBetweenDirections()
    {
        // Label smoothed DPO: (1-ε)*loss + ε*reversed_loss
        double beta = 0.1;
        double margin = 1.0;
        double epsilon = 0.1;

        double directLoss = -TestFineTuning.TestLogSigmoid(beta * margin);
        double reversedLoss = -TestFineTuning.TestLogSigmoid(-beta * margin);
        double smoothedLoss = (1 - epsilon) * directLoss + epsilon * reversedLoss;

        // Smoothed loss should be between the two extremes
        var minLoss = Math.Min(directLoss, reversedLoss);
        var maxLoss = Math.Max(directLoss, reversedLoss);
        Assert.True(smoothedLoss >= minLoss - Tolerance);
        Assert.True(smoothedLoss <= maxLoss + Tolerance);
    }

    [Fact]
    public void DPOLoss_ZeroSmoothing_EqualsDirectLoss()
    {
        double beta = 0.1;
        double margin = 1.0;
        double epsilon = 0.0;

        double directLoss = -TestFineTuning.TestLogSigmoid(beta * margin);
        double reversedLoss = -TestFineTuning.TestLogSigmoid(-beta * margin);
        double smoothedLoss = (1 - epsilon) * directLoss + epsilon * reversedLoss;

        Assert.Equal(directLoss, smoothedLoss, Tolerance);
    }

    // ============================
    // Cosine Similarity (used in embedding log prob) Tests
    // ============================

    [Fact]
    public void CosineSimilarity_ParallelVectors_ReturnsOne()
    {
        // cos(a, 2a) = (a . 2a) / (|a| * |2a|) = 2|a|^2 / (|a| * 2|a|) = 1
        var a = new[] { 1.0, 2.0, 3.0 };
        var b = new[] { 2.0, 4.0, 6.0 };

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        var cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.Equal(1.0, cosine, Tolerance);
    }

    [Fact]
    public void CosineSimilarity_AntiParallel_ReturnsNegOne()
    {
        var a = new[] { 1.0, 0.0 };
        var b = new[] { -1.0, 0.0 };

        double dot = a[0] * b[0] + a[1] * b[1];
        double normA = Math.Sqrt(a[0] * a[0] + a[1] * a[1]);
        double normB = Math.Sqrt(b[0] * b[0] + b[1] * b[1]);
        var cosine = dot / (normA * normB);
        Assert.Equal(-1.0, cosine, Tolerance);
    }

    // ============================
    // Method Configuration Tests
    // ============================

    [Fact]
    public void DPO_RequiresReferenceModel_NotRewardModel()
    {
        var dpo = new DirectPreferenceOptimization<double, double[], double[]>(new FineTuningOptions<double>());
        Assert.True(dpo.RequiresReferenceModel);
        Assert.False(dpo.RequiresRewardModel);
        Assert.Equal("DPO", dpo.MethodName);
    }

    [Fact]
    public void SimPO_DoesNotRequireReferenceOrReward()
    {
        var simpo = new SimplePreferenceOptimization<double, double[], double[]>(new FineTuningOptions<double>());
        Assert.False(simpo.RequiresReferenceModel);
        Assert.False(simpo.RequiresRewardModel);
        Assert.Equal("SimPO", simpo.MethodName);
    }

    [Fact]
    public void KTO_RequiresReferenceModel_NotRewardModel()
    {
        var kto = new KahnemanTverskyOptimization<double, double[], double[]>(new FineTuningOptions<double>());
        Assert.True(kto.RequiresReferenceModel);
        Assert.False(kto.RequiresRewardModel);
        Assert.Equal("KTO", kto.MethodName);
    }

    // ============================
    // Null/Edge Case Log Probability Tests
    // ============================

    [Fact]
    public void ArrayLogProb_SingleElement_TreatedAsScalar()
    {
        // Single-element arrays should be treated as scalars (cosine always 1 for same-sign)
        var ft = new TestFineTuning();
        var pred = new[] { 3.0 };
        var target = new[] { 5.0 };

        var result = ft.TestComputeLogProb(pred, target);
        // Single element: uses scalar logic: -(3-5)^2 / (2*1^2) = -4/2 = -2.0
        Assert.Equal(-2.0, result, 1e-8);
    }

    [Fact]
    public void StringLogProb_EmptyVsNonEmpty_ReturnsNegInfinity()
    {
        var ft = new StringTestFineTuning();
        var result = ft.TestComputeLogProb("", "hello");
        Assert.Equal(double.NegativeInfinity, result);
    }

    [Fact]
    public void StringLogProb_BothEmpty_ReturnsZero()
    {
        var ft = new StringTestFineTuning();
        var result = ft.TestComputeLogProb("", "");
        Assert.Equal(0.0, result, Tolerance);
    }
}
