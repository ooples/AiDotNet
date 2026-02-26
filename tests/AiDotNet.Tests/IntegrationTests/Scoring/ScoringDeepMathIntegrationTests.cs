using AiDotNet.Distributions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Scoring;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Scoring;

/// <summary>
/// Deep mathematical integration tests for scoring rules (CRPS, LogScore).
/// Tests hand-verified closed-form values, properness, gradient correctness,
/// non-negativity, unit invariance, and cross-method consistency.
/// </summary>
public class ScoringDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    // ================================================================
    // CRPS - Standard Normal closed-form hand calculations
    // ================================================================

    [Fact]
    public void CRPS_StandardNormal_AtMean_HandValue()
    {
        // CRPS(N(0,1), y=0) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
        // z = (0-0)/1 = 0, Phi(0) = 0.5, phi(0) = 1/sqrt(2*pi)
        // = 1 * [0*(2*0.5 - 1) + 2/sqrt(2*pi) - 1/sqrt(pi)]
        // = 2/sqrt(2*pi) - 1/sqrt(pi)
        // = sqrt(2/pi) - 1/sqrt(pi)
        // = 1/sqrt(pi) * (sqrt(2) - 1)
        // = (sqrt(2) - 1) / sqrt(pi) ~ 0.23369...
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double score = crps.Score(normal, 0.0);
        double expected = (Math.Sqrt(2) - 1) / Math.Sqrt(Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void CRPS_StandardNormal_At1Sigma_HandValue()
    {
        // z = 1.0, Phi(1) ~ 0.84134, phi(1) ~ 0.24197
        // CRPS = 1 * [1*(2*0.84134 - 1) + 2*0.24197 - 1/sqrt(pi)]
        //      = [1*0.68269 + 0.48394 - 0.56419]
        //      = 0.60244
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double score = crps.Score(normal, 1.0);

        double z = 1.0;
        double phi_z = Math.Exp(-0.5 * z * z) / Math.Sqrt(2 * Math.PI);
        double Phi_z = 0.5 * (1 + Erf(z / Math.Sqrt(2)));
        double expected = z * (2 * Phi_z - 1) + 2 * phi_z - 1 / Math.Sqrt(Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void CRPS_Normal_ScaledByVariance_HandValue()
    {
        // N(mu=3, var=4) => sigma=2
        // Observation y=5, z = (5-3)/2 = 1.0
        // CRPS = 2 * [1*(2*Phi(1)-1) + 2*phi(1) - 1/sqrt(pi)]
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(3.0, 4.0);
        double score = crps.Score(normal, 5.0);

        double z = 1.0;
        double sigma = 2.0;
        double phi_z = Math.Exp(-0.5 * z * z) / Math.Sqrt(2 * Math.PI);
        double Phi_z = 0.5 * (1 + Erf(z / Math.Sqrt(2)));
        double expected = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / Math.Sqrt(Math.PI));
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void CRPS_Normal_SymmetricAroundMean()
    {
        // CRPS(N(mu, sigma^2), mu+d) == CRPS(N(mu, sigma^2), mu-d)
        // because the formula depends on |z| through Phi and phi
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(5.0, 2.0);
        double scoreAbove = crps.Score(normal, 7.0);
        double scoreBelow = crps.Score(normal, 3.0);
        Assert.Equal(scoreAbove, scoreBelow, Tolerance);
    }

    [Fact]
    public void CRPS_Normal_NonNegative()
    {
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double[] observations = { -3, -2, -1, 0, 1, 2, 3 };
        foreach (var y in observations)
        {
            double score = crps.Score(normal, y);
            Assert.True(score >= 0, $"CRPS should be non-negative, got {score} for y={y}");
        }
    }

    [Fact]
    public void CRPS_Normal_IncreasesWithDistance()
    {
        // CRPS should increase as observation moves away from mean
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double prev = crps.Score(normal, 0.0);
        for (double y = 0.5; y <= 5.0; y += 0.5)
        {
            double score = crps.Score(normal, y);
            Assert.True(score > prev, $"CRPS should increase with distance: score({y})={score} <= score({y - 0.5})={prev}");
            prev = score;
        }
    }

    [Fact]
    public void CRPS_Normal_LargerVariance_LargerScore()
    {
        // For observation at mean, CRPS = sigma * (sqrt(2) - 1) / sqrt(pi)
        // So larger sigma means larger CRPS
        var crps = new CRPSScore<double>();
        var narrow = new NormalDistribution<double>(0.0, 1.0);
        var wide = new NormalDistribution<double>(0.0, 4.0);

        double scoreNarrow = crps.Score(narrow, 0.0);
        double scoreWide = crps.Score(wide, 0.0);
        Assert.True(scoreWide > scoreNarrow,
            $"Wider distribution should have larger CRPS at mean: {scoreWide} <= {scoreNarrow}");
    }

    [Fact]
    public void CRPS_Normal_SigmaScaling_LinearInSigma()
    {
        // CRPS(N(mu, k^2*sigma^2), mu) = k * CRPS(N(mu, sigma^2), mu)
        // This follows from CRPS = sigma * constant when y=mu
        var crps = new CRPSScore<double>();
        double score1 = crps.Score(new NormalDistribution<double>(0.0, 1.0), 0.0);
        double score4 = crps.Score(new NormalDistribution<double>(0.0, 4.0), 0.0);
        // sigma=1 vs sigma=2, so score4 should be 2 * score1
        Assert.Equal(2.0 * score1, score4, Tolerance);
    }

    // ================================================================
    // CRPS - Laplace closed-form hand calculations
    // ================================================================

    [Fact]
    public void CRPS_Laplace_AtLocation_HandValue()
    {
        // CRPS(Laplace(mu, b), y=mu) = |mu-mu| + b*exp(0) - 3b/4
        //                             = 0 + b - 3b/4 = b/4
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        double score = crps.Score(laplace, 0.0);
        Assert.Equal(0.25, score, Tolerance);  // b/4 = 1/4 = 0.25
    }

    [Fact]
    public void CRPS_Laplace_ScaleParameter_AtLocation()
    {
        // CRPS(Laplace(0, b), 0) = b/4
        var crps = new CRPSScore<double>();
        double[] scales = { 0.5, 1.0, 2.0, 5.0 };
        foreach (var b in scales)
        {
            var laplace = new LaplaceDistribution<double>(0.0, b);
            double score = crps.Score(laplace, 0.0);
            Assert.Equal(b / 4.0, score, Tolerance);
        }
    }

    [Fact]
    public void CRPS_Laplace_AwayFromLocation_HandValue()
    {
        // CRPS(Laplace(0, 1), y=2) = |2| + 1*exp(-2) - 3/4
        //                          = 2 + exp(-2) - 0.75
        //                          = 2 + 0.13534 - 0.75 = 1.38534
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        double score = crps.Score(laplace, 2.0);
        double expected = 2.0 + Math.Exp(-2.0) - 0.75;
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void CRPS_Laplace_SymmetricAroundLocation()
    {
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(3.0, 2.0);
        double scoreAbove = crps.Score(laplace, 5.0);
        double scoreBelow = crps.Score(laplace, 1.0);
        Assert.Equal(scoreAbove, scoreBelow, Tolerance);
    }

    [Fact]
    public void CRPS_Laplace_NonNegative()
    {
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        double[] observations = { -5, -2, -1, 0, 1, 2, 5 };
        foreach (var y in observations)
        {
            double score = crps.Score(laplace, y);
            Assert.True(score >= 0, $"CRPS should be non-negative, got {score} for y={y}");
        }
    }

    [Fact]
    public void CRPS_Laplace_IncreasesWithDistance()
    {
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        double prev = crps.Score(laplace, 0.0);
        for (double y = 0.5; y <= 5.0; y += 0.5)
        {
            double score = crps.Score(laplace, y);
            Assert.True(score > prev, $"CRPS should increase with distance: score({y})={score} <= prev={prev}");
            prev = score;
        }
    }

    // ================================================================
    // LogScore - Normal distribution hand calculations
    // ================================================================

    [Fact]
    public void LogScore_StandardNormal_AtMean_HandValue()
    {
        // LogScore = -log(pdf(0)) = -log(1/sqrt(2*pi)) = 0.5*log(2*pi) ~ 0.91894
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double score = log.Score(normal, 0.0);
        double expected = 0.5 * Math.Log(2 * Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void LogScore_StandardNormal_At1Sigma_HandValue()
    {
        // pdf(1) = exp(-0.5) / sqrt(2*pi)
        // LogScore = -log(exp(-0.5)/sqrt(2*pi)) = 0.5 + 0.5*log(2*pi)
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double score = log.Score(normal, 1.0);
        double expected = 0.5 + 0.5 * Math.Log(2 * Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void LogScore_Normal_GeneralFormula_HandValue()
    {
        // For N(mu, sigma^2), LogScore = (y-mu)^2/(2*sigma^2) + 0.5*log(2*pi*sigma^2)
        // N(3, 4), y=5: (5-3)^2/8 + 0.5*log(8*pi) = 0.5 + 0.5*log(8*pi)
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(3.0, 4.0);
        double score = log.Score(normal, 5.0);
        double expected = 4.0 / 8.0 + 0.5 * Math.Log(8 * Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void LogScore_Normal_MinimizedAtMean()
    {
        // LogScore = (y-mu)^2/(2*sigma^2) + const
        // Minimized when y = mu
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(2.0, 1.0);
        double scoreAtMean = log.Score(normal, 2.0);
        double scoreAway = log.Score(normal, 4.0);
        Assert.True(scoreAtMean < scoreAway,
            $"LogScore should be minimized at mean: {scoreAtMean} >= {scoreAway}");
    }

    [Fact]
    public void LogScore_Normal_SymmetricAroundMean()
    {
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double scorePos = log.Score(normal, 2.0);
        double scoreNeg = log.Score(normal, -2.0);
        Assert.Equal(scorePos, scoreNeg, Tolerance);
    }

    [Fact]
    public void LogScore_Normal_QuadraticInObservation()
    {
        // LogScore increases quadratically with distance from mean
        // Score(y+d) - Score(y) should be approximately linear in d for small d
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);

        double s0 = log.Score(normal, 0.0);
        double s1 = log.Score(normal, 1.0);
        double s2 = log.Score(normal, 2.0);

        // s(y) = y^2/2 + const, so s(1) - s(0) = 0.5, s(2) - s(0) = 2.0
        Assert.Equal(0.5, s1 - s0, Tolerance);
        Assert.Equal(2.0, s2 - s0, Tolerance);
    }

    [Fact]
    public void LogScore_Normal_LargerVariance_LargerAtMean()
    {
        // At mean: LogScore = 0.5*log(2*pi*sigma^2), which increases with sigma
        var log = new LogScore<double>();
        var narrow = new NormalDistribution<double>(0.0, 1.0);
        var wide = new NormalDistribution<double>(0.0, 4.0);

        double scoreNarrow = log.Score(narrow, 0.0);
        double scoreWide = log.Score(wide, 0.0);
        Assert.True(scoreWide > scoreNarrow,
            $"LogScore at mean should increase with variance: {scoreWide} <= {scoreNarrow}");
    }

    // ================================================================
    // LogScore - Laplace distribution
    // ================================================================

    [Fact]
    public void LogScore_Laplace_AtLocation_HandValue()
    {
        // Laplace pdf at location: f(mu) = 1/(2b) * exp(0) = 1/(2b)
        // LogScore = -log(1/(2b)) = log(2b)
        // For b=1: LogScore = log(2) ~ 0.69315
        var log = new LogScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        double score = log.Score(laplace, 0.0);
        Assert.Equal(Math.Log(2), score, Tolerance);
    }

    [Fact]
    public void LogScore_Laplace_AwayFromLocation_HandValue()
    {
        // Laplace pdf: f(y) = 1/(2b) * exp(-|y-mu|/b)
        // LogScore = log(2b) + |y-mu|/b
        // For mu=0, b=1, y=2: log(2) + 2 ~ 2.69315
        var log = new LogScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        double score = log.Score(laplace, 2.0);
        double expected = Math.Log(2) + 2.0;
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void LogScore_Laplace_LinearInDistance()
    {
        // LogScore for Laplace: log(2b) + |y-mu|/b
        // The distance term is linear, unlike Normal which is quadratic
        var log = new LogScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);

        double s0 = log.Score(laplace, 0.0);
        double s1 = log.Score(laplace, 1.0);
        double s2 = log.Score(laplace, 2.0);
        double s3 = log.Score(laplace, 3.0);

        // Differences should be constant (= 1/b = 1)
        Assert.Equal(1.0, s1 - s0, Tolerance);
        Assert.Equal(1.0, s2 - s1, Tolerance);
        Assert.Equal(1.0, s3 - s2, Tolerance);
    }

    // ================================================================
    // MeanScore - Averaging property
    // ================================================================

    [Fact]
    public void CRPS_MeanScore_IsAverageOfScores()
    {
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var distributions = new IParametricDistribution<double>[] { normal, normal, normal };
        var observations = new Vector<double>(new[] { -1.0, 0.0, 1.0 });

        double meanScore = crps.MeanScore(distributions, observations);

        double s1 = crps.Score(normal, -1.0);
        double s2 = crps.Score(normal, 0.0);
        double s3 = crps.Score(normal, 1.0);
        double expectedMean = (s1 + s2 + s3) / 3.0;

        Assert.Equal(expectedMean, meanScore, Tolerance);
    }

    [Fact]
    public void LogScore_MeanScore_IsAverageOfScores()
    {
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var distributions = new IParametricDistribution<double>[] { normal, normal };
        var observations = new Vector<double>(new[] { 0.0, 1.0 });

        double meanScore = log.MeanScore(distributions, observations);

        double s1 = log.Score(normal, 0.0);
        double s2 = log.Score(normal, 1.0);
        double expectedMean = (s1 + s2) / 2.0;

        Assert.Equal(expectedMean, meanScore, Tolerance);
    }

    [Fact]
    public void MeanScore_MismatchedLengths_Throws()
    {
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var distributions = new IParametricDistribution<double>[] { normal };
        var observations = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Throws<ArgumentException>(() => crps.MeanScore(distributions, observations));
    }

    [Fact]
    public void MeanScore_EmptyArrays_Throws()
    {
        var crps = new CRPSScore<double>();
        var distributions = Array.Empty<IParametricDistribution<double>>();
        var observations = new Vector<double>(Array.Empty<double>());

        Assert.Throws<ArgumentException>(() => crps.MeanScore(distributions, observations));
    }

    // ================================================================
    // CRPS Gradient - Normal distribution
    // ================================================================

    [Fact]
    public void CRPS_Gradient_Normal_AtMean_MeanGradientIsZero()
    {
        // d(CRPS)/d(mu) = -(2*Phi(z)-1)
        // At z=0: -(2*0.5 - 1) = 0
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var grad = crps.ScoreGradient(normal, 0.0);

        Assert.Equal(0.0, grad[0], Tolerance); // d/dmu = 0 at mean
    }

    [Fact]
    public void CRPS_Gradient_Normal_MeanGradient_HandValue()
    {
        // d(CRPS)/d(mu) = -(2*Phi(z)-1)
        // z = (y-mu)/sigma = (2-0)/1 = 2
        // Phi(2) ~ 0.97725
        // grad = -(2*0.97725 - 1) = -0.95450
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var grad = crps.ScoreGradient(normal, 2.0);

        double z = 2.0;
        double Phi_z = 0.5 * (1 + Erf(z / Math.Sqrt(2)));
        double expected = -(2 * Phi_z - 1);
        Assert.Equal(expected, grad[0], Tolerance);
    }

    [Fact]
    public void CRPS_Gradient_Normal_MatchesNumericalDifferentiation()
    {
        var crps = new CRPSScore<double>();
        double mu = 2.0, var_ = 3.0, y = 4.0;
        double h = 1e-5;

        // Numerical gradient with respect to mean
        double scorePlus = crps.Score(new NormalDistribution<double>(mu + h, var_), y);
        double scoreMinus = crps.Score(new NormalDistribution<double>(mu - h, var_), y);
        double numericalGradMean = (scorePlus - scoreMinus) / (2 * h);

        // Numerical gradient with respect to variance
        double scoreVarPlus = crps.Score(new NormalDistribution<double>(mu, var_ + h), y);
        double scoreVarMinus = crps.Score(new NormalDistribution<double>(mu, var_ - h), y);
        double numericalGradVar = (scoreVarPlus - scoreVarMinus) / (2 * h);

        var analyticalGrad = crps.ScoreGradient(new NormalDistribution<double>(mu, var_), y);

        Assert.Equal(numericalGradMean, analyticalGrad[0], LooseTolerance);
        Assert.Equal(numericalGradVar, analyticalGrad[1], LooseTolerance);
    }

    // ================================================================
    // LogScore Gradient - Normal distribution
    // ================================================================

    [Fact]
    public void LogScore_Gradient_Normal_MeanGrad_HandValue()
    {
        // LogScore = (y-mu)^2/(2*sigma^2) + 0.5*log(2*pi*sigma^2)
        // d/dmu = -(y-mu)/sigma^2
        // For N(0,1), y=2: d/dmu = -2
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var grad = log.ScoreGradient(normal, 2.0);

        Assert.Equal(-2.0, grad[0], Tolerance);
    }

    [Fact]
    public void LogScore_Gradient_Normal_VarianceGrad_HandValue()
    {
        // d/d(sigma^2) = (y-mu)^2/(2*sigma^4) - 1/(2*sigma^2)
        // For N(0,1), y=2: (4/2) - 0.5 = 2.0 - 0.5 = 1.5
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var grad = log.ScoreGradient(normal, 2.0);

        // Wait, this is d(-logpdf)/d(variance)
        // -log pdf = (y-mu)^2/(2v) + 0.5*log(2*pi*v)
        // d/dv = -(y-mu)^2/(2v^2) + 1/(2v)
        // Actually: for variance v=1, y=2, mu=0:
        // d/dv = -4/2 + 0.5 = -1.5... wait
        // Let me recalculate:
        // log pdf = -0.5*log(2*pi*v) - (y-mu)^2/(2v)
        // d(log pdf)/dv = -1/(2v) + (y-mu)^2/(2v^2)
        // -d(log pdf)/dv = 1/(2v) - (y-mu)^2/(2v^2)
        // For v=1, (y-mu)^2=4: 0.5 - 2.0 = -1.5
        Assert.Equal(-1.5, grad[1], Tolerance);
    }

    [Fact]
    public void LogScore_Gradient_Normal_AtMean_MeanGradIsZero()
    {
        // d(-logpdf)/d(mu) = -(y-mu)/sigma^2
        // At y=mu: gradient = 0
        var log = new LogScore<double>();
        var normal = new NormalDistribution<double>(5.0, 2.0);
        var grad = log.ScoreGradient(normal, 5.0);
        Assert.Equal(0.0, grad[0], Tolerance);
    }

    // ================================================================
    // Scoring rule properties
    // ================================================================

    [Fact]
    public void CRPS_Name_IsCRPS()
    {
        var crps = new CRPSScore<double>();
        Assert.Equal("CRPS", crps.Name);
    }

    [Fact]
    public void CRPS_IsMinimized_True()
    {
        var crps = new CRPSScore<double>();
        Assert.True(crps.IsMinimized);
    }

    [Fact]
    public void LogScore_Name_IsLogScore()
    {
        var log = new LogScore<double>();
        Assert.Equal("LogScore", log.Name);
    }

    [Fact]
    public void LogScore_IsMinimized_True()
    {
        var log = new LogScore<double>();
        Assert.True(log.IsMinimized);
    }

    [Fact]
    public void CRPS_InvalidIntegrationPoints_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CRPSScore<double>(numIntegrationPoints: 9));
    }

    [Fact]
    public void CRPS_MinimumIntegrationPoints_Accepted()
    {
        var crps = new CRPSScore<double>(numIntegrationPoints: 10);
        var normal = new NormalDistribution<double>(0.0, 1.0);
        double score = crps.Score(normal, 0.0);
        Assert.True(score >= 0);
    }

    // ================================================================
    // Properness property: score at true distribution <= wrong distribution
    // ================================================================

    [Fact]
    public void CRPS_Properness_TrueDistribution_BetterThanWrong()
    {
        // Generate observations from N(0,1)
        // CRPS should be lower for N(0,1) than for N(5,1) (wrong mean)
        var crps = new CRPSScore<double>();
        var trueNormal = new NormalDistribution<double>(0.0, 1.0);
        var wrongNormal = new NormalDistribution<double>(5.0, 1.0);

        double[] observations = { -1.5, -0.5, 0.2, 0.8, 1.3 };
        double trueTotal = 0, wrongTotal = 0;
        foreach (var y in observations)
        {
            trueTotal += crps.Score(trueNormal, y);
            wrongTotal += crps.Score(wrongNormal, y);
        }

        Assert.True(trueTotal < wrongTotal,
            $"True distribution should have lower CRPS: {trueTotal} >= {wrongTotal}");
    }

    [Fact]
    public void LogScore_Properness_TrueDistribution_BetterThanWrong()
    {
        var log = new LogScore<double>();
        var trueNormal = new NormalDistribution<double>(0.0, 1.0);
        var wrongNormal = new NormalDistribution<double>(5.0, 1.0);

        double[] observations = { -1.5, -0.5, 0.2, 0.8, 1.3 };
        double trueTotal = 0, wrongTotal = 0;
        foreach (var y in observations)
        {
            trueTotal += log.Score(trueNormal, y);
            wrongTotal += log.Score(wrongNormal, y);
        }

        Assert.True(trueTotal < wrongTotal,
            $"True distribution should have lower LogScore: {trueTotal} >= {wrongTotal}");
    }

    // ================================================================
    // CRPS vs LogScore comparison
    // ================================================================

    [Fact]
    public void CRPS_VsLogScore_BothMinimizedAtSameDistribution()
    {
        // Both CRPS and LogScore should prefer the same distribution when it's the true one
        var crps = new CRPSScore<double>();
        var log = new LogScore<double>();

        var dist1 = new NormalDistribution<double>(0.0, 1.0);
        var dist2 = new NormalDistribution<double>(3.0, 1.0);

        double[] observations = { -0.5, 0.3, 0.7, -0.2, 0.1 };

        double crps1 = 0, crps2 = 0, log1 = 0, log2 = 0;
        foreach (var y in observations)
        {
            crps1 += crps.Score(dist1, y);
            crps2 += crps.Score(dist2, y);
            log1 += log.Score(dist1, y);
            log2 += log.Score(dist2, y);
        }

        // Both should prefer dist1 (closer to the observations)
        Assert.True(crps1 < crps2, "CRPS should prefer dist1");
        Assert.True(log1 < log2, "LogScore should prefer dist1");
    }

    // ================================================================
    // Helper: Erf approximation (same as used in CRPSScore)
    // ================================================================

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
