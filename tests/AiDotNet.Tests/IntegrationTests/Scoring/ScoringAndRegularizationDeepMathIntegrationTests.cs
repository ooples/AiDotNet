using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Regularization;
using AiDotNet.Scoring;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Scoring;

/// <summary>
/// Deep math integration tests for Scoring Rules (CRPS, LogScore) and
/// Regularization (L1, L2, ElasticNet) with hand-computed expected values.
/// </summary>
public class ScoringAndRegularizationDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    // ============================
    // CRPS Score - Normal Distribution (Closed-Form)
    // ============================

    [Fact]
    public void CRPSNormal_ObservationAtMean_Formula()
    {
        // CRPS for N(0,1) at observation=0: z=0, phi(0)=1/sqrt(2*pi), Phi(0)=0.5
        // CRPS = sigma * [z*(2*Phi-1) + 2*phi - 1/sqrt(pi)]
        //      = 1.0 * [0*(2*0.5-1) + 2*(1/sqrt(2*pi)) - 1/sqrt(pi)]
        //      = 2/sqrt(2*pi) - 1/sqrt(pi)
        //      = sqrt(2/pi) - 1/sqrt(pi)
        //      = 1/sqrt(pi) * (sqrt(2) - 1)
        var dist = new NormalDistribution<double>(0.0, 1.0); // mean=0, variance=1
        var crps = new CRPSScore<double>();
        double score = crps.Score(dist, 0.0);

        double expected = (Math.Sqrt(2.0) - 1.0) / Math.Sqrt(Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void CRPSNormal_ObservationAwayFromMean_IncreasesScore()
    {
        var dist = new NormalDistribution<double>(0.0, 1.0);
        var crps = new CRPSScore<double>();

        double scoreAtMean = crps.Score(dist, 0.0);
        double scoreAway = crps.Score(dist, 3.0);

        // CRPS increases with distance from mean
        Assert.True(scoreAway > scoreAtMean,
            $"CRPS at 3.0 ({scoreAway}) should be > CRPS at 0.0 ({scoreAtMean})");
    }

    [Fact]
    public void CRPSNormal_LargerVariance_IncreasesScore()
    {
        var narrow = new NormalDistribution<double>(0.0, 1.0);   // variance=1
        var wide = new NormalDistribution<double>(0.0, 100.0);   // variance=100 (sigma=10)
        var crps = new CRPSScore<double>();

        double scoreNarrow = crps.Score(narrow, 0.0);
        double scoreWide = crps.Score(wide, 0.0);

        // Wider distribution = higher CRPS (less confident)
        Assert.True(scoreWide > scoreNarrow,
            $"Wide CRPS ({scoreWide}) should be > narrow CRPS ({scoreNarrow})");
    }

    [Fact]
    public void CRPSNormal_ScalesWithSigma()
    {
        // CRPS at mean for N(mu, sigma^2) = sigma * (sqrt(2) - 1) / sqrt(pi)
        // So CRPS is proportional to sigma
        var dist1 = new NormalDistribution<double>(0.0, 1.0);   // sigma=1
        var dist2 = new NormalDistribution<double>(0.0, 4.0);   // sigma=2
        var crps = new CRPSScore<double>();

        double score1 = crps.Score(dist1, 0.0);
        double score2 = crps.Score(dist2, 0.0);

        // score2 should be exactly 2 * score1
        Assert.Equal(2.0 * score1, score2, Tolerance);
    }

    [Fact]
    public void CRPSNormal_IsNonNegative()
    {
        var dist = new NormalDistribution<double>(5.0, 4.0);
        var crps = new CRPSScore<double>();

        double score = crps.Score(dist, 3.0);
        Assert.True(score >= 0, $"CRPS should be non-negative, got {score}");
    }

    [Fact]
    public void CRPSNormal_HandComputed_SpecificCase()
    {
        // N(10, 4): mu=10, sigma=2
        // Observation y=12: z=(12-10)/2 = 1
        // phi(1) = exp(-0.5)/sqrt(2*pi) ≈ 0.241971
        // Phi(1) ≈ 0.841345
        // CRPS = 2 * [1*(2*0.841345-1) + 2*0.241971 - 1/sqrt(pi)]
        //      = 2 * [0.682689 + 0.483942 - 0.564190]
        //      = 2 * 0.602441
        //      = 1.204882
        var dist = new NormalDistribution<double>(10.0, 4.0);
        var crps = new CRPSScore<double>();

        double score = crps.Score(dist, 12.0);

        double z = 1.0;
        double phi = Math.Exp(-0.5) / Math.Sqrt(2 * Math.PI);
        double phiCdf = 0.5 * (1 + Erf(1.0 / Math.Sqrt(2)));
        double expected = 2.0 * (z * (2 * phiCdf - 1) + 2 * phi - 1 / Math.Sqrt(Math.PI));

        Assert.Equal(expected, score, Tolerance);
    }

    // ============================
    // CRPS Score - Laplace Distribution (Closed-Form)
    // ============================

    [Fact]
    public void CRPSLaplace_ObservationAtLocation_Formula()
    {
        // CRPS for Laplace(mu, b) at y=mu:
        // |y-mu| = 0, exp(0) = 1
        // CRPS = 0 + b*1 - 3b/4 = b/4
        var dist = new LaplaceDistribution<double>(0.0, 2.0); // location=0, scale=2
        var crps = new CRPSScore<double>();

        double score = crps.Score(dist, 0.0);
        Assert.Equal(0.5, score, Tolerance); // b/4 = 2/4 = 0.5
    }

    [Fact]
    public void CRPSLaplace_HandComputed_AwayFromLocation()
    {
        // Laplace(mu=0, b=1), observation y=2
        // |y-mu| = 2, exp(-2/1) = exp(-2)
        // CRPS = 2 + 1*exp(-2) - 3*1/4 = 2 + exp(-2) - 0.75
        var dist = new LaplaceDistribution<double>(0.0, 1.0);
        var crps = new CRPSScore<double>();

        double score = crps.Score(dist, 2.0);
        double expected = 2.0 + Math.Exp(-2.0) - 0.75;

        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void CRPSLaplace_IsNonNegative()
    {
        var dist = new LaplaceDistribution<double>(5.0, 3.0);
        var crps = new CRPSScore<double>();

        double score = crps.Score(dist, -1.0);
        Assert.True(score >= 0, $"CRPS should be non-negative, got {score}");
    }

    // ============================
    // LogScore Tests
    // ============================

    [Fact]
    public void LogScore_IsNegLogPdf()
    {
        // LogScore = -log(pdf(y))
        // For N(0,1), pdf(0) = 1/sqrt(2*pi), log(pdf(0)) = -0.5*log(2*pi)
        // LogScore = 0.5*log(2*pi)
        var dist = new NormalDistribution<double>(0.0, 1.0);
        var logScore = new LogScore<double>();

        double score = logScore.Score(dist, 0.0);
        double expected = 0.5 * Math.Log(2 * Math.PI);

        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void LogScore_FartherObservation_HigherScore()
    {
        var dist = new NormalDistribution<double>(0.0, 1.0);
        var logScore = new LogScore<double>();

        double scoreClose = logScore.Score(dist, 0.0);
        double scoreFar = logScore.Score(dist, 5.0);

        Assert.True(scoreFar > scoreClose,
            $"LogScore far ({scoreFar}) should be > close ({scoreClose})");
    }

    [Fact]
    public void LogScore_NormalAtMean_HandComputed()
    {
        // N(5, 9): mu=5, variance=9, sigma=3
        // pdf(5) = 1/(3*sqrt(2*pi))
        // log(pdf(5)) = -log(3) - 0.5*log(2*pi)
        // LogScore = log(3) + 0.5*log(2*pi)
        var dist = new NormalDistribution<double>(5.0, 9.0);
        var logScore = new LogScore<double>();

        double score = logScore.Score(dist, 5.0);
        double expected = Math.Log(3.0) + 0.5 * Math.Log(2 * Math.PI);

        Assert.Equal(expected, score, Tolerance);
    }

    // ============================
    // CRPS Gradient Tests
    // ============================

    [Fact]
    public void CRPSNormalGradient_AtMean_MeanGradientIsZero()
    {
        // d(CRPS)/d(mu) = -(2*Phi(z) - 1), at z=0: -(2*0.5-1) = 0
        var dist = new NormalDistribution<double>(0.0, 1.0);
        var crps = new CRPSScore<double>();

        var gradient = crps.ScoreGradient(dist, 0.0);
        Assert.Equal(0.0, gradient[0], Tolerance); // gradient w.r.t. mean
    }

    [Fact]
    public void CRPSNormalGradient_ObservationAboveMean_NegativeMeanGrad()
    {
        // When y > mu, z > 0, Phi(z) > 0.5, so 2*Phi-1 > 0, grad_mu = -(2*Phi-1) < 0
        // This means increasing mu would decrease CRPS (move closer to observation)
        var dist = new NormalDistribution<double>(0.0, 1.0);
        var crps = new CRPSScore<double>();

        var gradient = crps.ScoreGradient(dist, 3.0);
        Assert.True(gradient[0] < 0,
            $"Mean gradient ({gradient[0]}) should be negative when observation is above mean");
    }

    // ============================
    // L1 Regularization (Lasso) Tests
    // ============================

    [Fact]
    public void L1_SoftThresholding_BelowThreshold_IsZero()
    {
        // L1 with strength=0.5: values with |x| < 0.5 → 0
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.5, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var data = new Vector<double>(new double[] { 0.3, -0.2, 0.1, -0.4 });
        var result = l1.Regularize(data);

        // All values have |x| < 0.5, so all should be 0
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(0.0, result[i], Tolerance);
        }
    }

    [Fact]
    public void L1_SoftThresholding_AboveThreshold_ShrunkByStrength()
    {
        // L1 with strength=0.1: sign(x) * max(0, |x| - 0.1)
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var data = new Vector<double>(new double[] { 0.5, -0.8, 1.0, -0.05 });
        var result = l1.Regularize(data);

        // 0.5: sign(0.5)*max(0, 0.5-0.1) = 1*0.4 = 0.4
        Assert.Equal(0.4, result[0], Tolerance);
        // -0.8: sign(-0.8)*max(0, 0.8-0.1) = -1*0.7 = -0.7
        Assert.Equal(-0.7, result[1], Tolerance);
        // 1.0: sign(1)*max(0, 1.0-0.1) = 0.9
        Assert.Equal(0.9, result[2], Tolerance);
        // -0.05: |x| < 0.1 → 0
        Assert.Equal(0.0, result[3], Tolerance);
    }

    [Fact]
    public void L1_SparsityInducing_SmallValuesGoToZero()
    {
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.3, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var data = new Vector<double>(new double[] { 0.1, 0.2, 0.5, 1.0, 0.01 });
        var result = l1.Regularize(data);

        // Values below 0.3 should be exactly zero
        Assert.Equal(0.0, result[0], Tolerance); // 0.1 < 0.3
        Assert.Equal(0.0, result[1], Tolerance); // 0.2 < 0.3
        Assert.True(result[2] > 0); // 0.5 > 0.3, shrunk to 0.2
        Assert.True(result[3] > 0); // 1.0 > 0.3, shrunk to 0.7
        Assert.Equal(0.0, result[4], Tolerance); // 0.01 < 0.3
    }

    [Fact]
    public void L1_GradientRegularize_AddsSubdifferential()
    {
        // Gradient regularization: gradient + strength * sign(coefficients)
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var gradient = new Vector<double>(new double[] { 1.0, -2.0, 0.5 });
        var coefficients = new Vector<double>(new double[] { 3.0, -4.0, 0.0 });

        var result = l1.Regularize(gradient, coefficients);

        // result[0] = 1.0 + 0.1 * sign(3.0) = 1.0 + 0.1 = 1.1
        Assert.Equal(1.1, result[0], Tolerance);
        // result[1] = -2.0 + 0.1 * sign(-4.0) = -2.0 + 0.1*(-1) = -2.1
        Assert.Equal(-2.1, result[1], Tolerance);
        // result[2] = 0.5 + 0.1 * sign(0.0) = 0.5 + 0 = 0.5
        Assert.Equal(0.5, result[2], Tolerance);
    }

    // ============================
    // L2 Regularization (Ridge) Tests
    // ============================

    [Fact]
    public void L2_UniformShrinkage_Formula()
    {
        // L2 Regularize(vector): x * (1 - strength)
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.01, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var data = new Vector<double>(new double[] { 10.0, -5.0, 2.0, -1.0 });
        var result = l2.Regularize(data);

        double shrinkage = 1.0 - 0.01;
        Assert.Equal(10.0 * shrinkage, result[0], Tolerance);
        Assert.Equal(-5.0 * shrinkage, result[1], Tolerance);
        Assert.Equal(2.0 * shrinkage, result[2], Tolerance);
        Assert.Equal(-1.0 * shrinkage, result[3], Tolerance);
    }

    [Fact]
    public void L2_NeverProducesExactZeros()
    {
        // L2 shrinks but never sets to exactly zero (unlike L1)
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.5, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var data = new Vector<double>(new double[] { 0.001, -0.001, 0.5, -0.5 });
        var result = l2.Regularize(data);

        for (int i = 0; i < result.Length; i++)
        {
            // All non-zero inputs should produce non-zero outputs
            Assert.NotEqual(0.0, result[i]);
        }
    }

    [Fact]
    public void L2_GradientRegularize_AddsProportionalPenalty()
    {
        // Gradient regularization: gradient + strength * coefficients
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.01, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var gradient = new Vector<double>(new double[] { 1.0, -2.0 });
        var coefficients = new Vector<double>(new double[] { 10.0, -20.0 });

        var result = l2.Regularize(gradient, coefficients);

        // result[0] = 1.0 + 0.01 * 10.0 = 1.1
        Assert.Equal(1.1, result[0], Tolerance);
        // result[1] = -2.0 + 0.01 * (-20.0) = -2.2
        Assert.Equal(-2.2, result[1], Tolerance);
    }

    [Fact]
    public void L2_PreservesSign()
    {
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var data = new Vector<double>(new double[] { 5.0, -3.0, 0.0 });
        var result = l2.Regularize(data);

        Assert.True(result[0] > 0, "Positive input should stay positive");
        Assert.True(result[1] < 0, "Negative input should stay negative");
        Assert.Equal(0.0, result[2], Tolerance); // Zero stays zero
    }

    [Fact]
    public void L2_MatrixRegularize_UniformShrinkage()
    {
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var data = new Matrix<double>(2, 2);
        data[0, 0] = 1.0; data[0, 1] = 2.0;
        data[1, 0] = 3.0; data[1, 1] = 4.0;

        var result = l2.Regularize(data);

        double f = 0.9; // 1 - 0.1
        Assert.Equal(1.0 * f, result[0, 0], Tolerance);
        Assert.Equal(2.0 * f, result[0, 1], Tolerance);
        Assert.Equal(3.0 * f, result[1, 0], Tolerance);
        Assert.Equal(4.0 * f, result[1, 1], Tolerance);
    }

    // ============================
    // ElasticNet Regularization Tests
    // ============================

    [Fact]
    public void ElasticNet_L1RatioZero_PureL2()
    {
        // L1Ratio=0 means pure L2
        var elastic = new ElasticNetRegularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 0.0, Type = RegularizationType.ElasticNet });
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var data = new Vector<double>(new double[] { 1.0, -2.0, 0.05, -0.5 });
        var elasticResult = elastic.Regularize(data);
        var l2Result = l2.Regularize(data);

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(l2Result[i], elasticResult[i], Tolerance);
        }
    }

    [Fact]
    public void ElasticNet_L1RatioOne_PureL1()
    {
        // L1Ratio=1 means pure L1
        var elastic = new ElasticNetRegularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 1.0, Type = RegularizationType.ElasticNet });
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var data = new Vector<double>(new double[] { 1.0, -2.0, 0.05, -0.5 });
        var elasticResult = elastic.Regularize(data);
        var l1Result = l1.Regularize(data);

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(l1Result[i], elasticResult[i], Tolerance);
        }
    }

    [Fact]
    public void ElasticNet_HandComputed_MixedPenalty()
    {
        // ElasticNet with strength=0.2, L1Ratio=0.5:
        // L1 threshold = strength * L1Ratio = 0.2 * 0.5 = 0.1
        // L2 shrinkage = 1 - strength * (1 - L1Ratio) = 1 - 0.2*0.5 = 0.9
        // Step 1: L1 soft threshold with 0.1
        // Step 2: Multiply by 0.9
        var elastic = new ElasticNetRegularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.2, L1Ratio = 0.5, Type = RegularizationType.ElasticNet });

        var data = new Vector<double>(new double[] { 0.5, -0.05, 1.0 });
        var result = elastic.Regularize(data);

        // 0.5: L1→ sign(0.5)*max(0, 0.5-0.1) = 0.4, then L2→ 0.4*0.9 = 0.36
        Assert.Equal(0.36, result[0], Tolerance);
        // -0.05: |x| < 0.1 → 0, then L2→ 0
        Assert.Equal(0.0, result[1], Tolerance);
        // 1.0: L1→ sign(1)*max(0, 1.0-0.1) = 0.9, then L2→ 0.9*0.9 = 0.81
        Assert.Equal(0.81, result[2], Tolerance);
    }

    [Fact]
    public void ElasticNet_GradientRegularize_CombinesL1AndL2()
    {
        // gradient + strength * [L1Ratio * sign(coeff) + (1-L1Ratio) * coeff]
        var elastic = new ElasticNetRegularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 0.5, Type = RegularizationType.ElasticNet });

        var gradient = new Vector<double>(new double[] { 1.0, -1.0 });
        var coefficients = new Vector<double>(new double[] { 4.0, -6.0 });

        var result = elastic.Regularize(gradient, coefficients);

        // result[0] = 1.0 + 0.1 * [0.5 * sign(4) + 0.5 * 4] = 1.0 + 0.1 * (0.5 + 2.0) = 1.0 + 0.25 = 1.25
        Assert.Equal(1.25, result[0], Tolerance);
        // result[1] = -1.0 + 0.1 * [0.5 * sign(-6) + 0.5 * (-6)] = -1.0 + 0.1 * (-0.5 - 3.0) = -1.0 - 0.35 = -1.35
        Assert.Equal(-1.35, result[1], Tolerance);
    }

    [Fact]
    public void ElasticNet_MoreSparseWithHigherL1Ratio()
    {
        var data = new Vector<double>(new double[] { 0.15, -0.15, 0.3, -0.3 });

        var lowL1 = new ElasticNetRegularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.2, L1Ratio = 0.2, Type = RegularizationType.ElasticNet });
        var highL1 = new ElasticNetRegularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.2, L1Ratio = 0.8, Type = RegularizationType.ElasticNet });

        var lowResult = lowL1.Regularize(data);
        var highResult = highL1.Regularize(data);

        // Count zeros: higher L1 ratio should produce more zeros
        int lowZeros = Enumerable.Range(0, data.Length).Count(i => Math.Abs(lowResult[i]) < 1e-10);
        int highZeros = Enumerable.Range(0, data.Length).Count(i => Math.Abs(highResult[i]) < 1e-10);

        Assert.True(highZeros >= lowZeros,
            $"Higher L1 ratio zeros ({highZeros}) should be >= low L1 ratio zeros ({lowZeros})");
    }

    // ============================
    // Regularization Properties
    // ============================

    [Fact]
    public void L1_Matrix_SoftThresholding_HandComputed()
    {
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.2, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var data = new Matrix<double>(2, 2);
        data[0, 0] = 0.5; data[0, 1] = -0.1;
        data[1, 0] = -0.3; data[1, 1] = 1.0;

        var result = l1.Regularize(data);

        // [0,0]: sign(0.5)*max(0, 0.5-0.2) = 0.3
        Assert.Equal(0.3, result[0, 0], Tolerance);
        // [0,1]: |0.1| < 0.2 → 0
        Assert.Equal(0.0, result[0, 1], Tolerance);
        // [1,0]: sign(-0.3)*max(0, 0.3-0.2) = -0.1
        Assert.Equal(-0.1, result[1, 0], Tolerance);
        // [1,1]: sign(1.0)*max(0, 1.0-0.2) = 0.8
        Assert.Equal(0.8, result[1, 1], Tolerance);
    }

    [Fact]
    public void Regularization_ZeroInputs_StayZero()
    {
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 1.0, Type = RegularizationType.L1 });
        var l2 = new L2Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.1, L1Ratio = 0.0, Type = RegularizationType.L2 });

        var zeros = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

        var l1Result = l1.Regularize(zeros);
        var l2Result = l2.Regularize(zeros);

        for (int i = 0; i < zeros.Length; i++)
        {
            Assert.Equal(0.0, l1Result[i], Tolerance);
            Assert.Equal(0.0, l2Result[i], Tolerance);
        }
    }

    [Fact]
    public void L1_SoftThresholding_IsIdempotentAtFixedPoint()
    {
        // Applying L1 regularization repeatedly to the same data:
        // After first application, values are shrunk. Applying again shrinks further.
        // At a fixed point (all zeros for small values), it stabilizes.
        var l1 = new L1Regularization<double, double[], Vector<double>>(
            new RegularizationOptions { Strength = 0.5, L1Ratio = 1.0, Type = RegularizationType.L1 });

        var data = new Vector<double>(new double[] { 0.3, -0.2 }); // All below threshold
        var result1 = l1.Regularize(data); // All become 0
        var result2 = l1.Regularize(result1); // Still 0

        for (int i = 0; i < data.Length; i++)
        {
            Assert.Equal(result1[i], result2[i], Tolerance);
        }
    }

    // ============================
    // Scoring Properties
    // ============================

    [Fact]
    public void CRPSScore_IsMinimized()
    {
        var crps = new CRPSScore<double>();
        Assert.True(crps.IsMinimized);
    }

    [Fact]
    public void LogScore_IsMinimized()
    {
        var logScore = new LogScore<double>();
        Assert.True(logScore.IsMinimized);
    }

    [Fact]
    public void CRPS_MinIntegrationPoints_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CRPSScore<double>(5)); // min is 10
    }

    // ============================
    // Helper: Error Function (for verification)
    // ============================

    private static double Erf(double x)
    {
        double sign = x < 0 ? -1.0 : 1.0;
        x = Math.Abs(x);

        const double a1 = 0.254829592;
        const double a2 = -0.284496736;
        const double a3 = 1.421413741;
        const double a4 = -1.453152027;
        const double a5 = 1.061405429;
        const double p = 0.3275911;

        double t = 1.0 / (1.0 + p * x);
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;

        double y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * Math.Exp(-x * x);
        return sign * y;
    }
}
