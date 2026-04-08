using AiDotNet.Distributions;
using AiDotNet.Scoring;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Scoring;

/// <summary>
/// Integration tests for scoring rules (CRPSScore, LogScore).
/// Tests proper scoring rule properties, known values, and gradient computation.
/// </summary>
public class ScoringRulesIntegrationTests
{
    private const double Tolerance = 1e-4;
    private const double LooseTolerance = 1e-2;

    #region CRPSScore - Construction and Properties

    [Fact]
    public void CRPSScore_DefaultConstruction_NameAndProperties()
    {
        var crps = new CRPSScore<double>();
        Assert.Equal("CRPS", crps.Name);
        Assert.True(crps.IsMinimized);
    }

    [Fact]
    public void CRPSScore_InvalidIntegrationPoints_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CRPSScore<double>(numIntegrationPoints: 5));
    }

    #endregion

    #region CRPSScore - Normal Distribution

    [Fact]
    public void CRPSScore_NormalDistribution_AtMean_IsNonNegative()
    {
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var score = crps.Score(normal, 0.0);
        Assert.True(score >= 0, $"CRPS should be non-negative, got {score}");
    }

    [Fact]
    public void CRPSScore_NormalDistribution_KnownValue()
    {
        // For standard normal with observation at mean:
        // CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
        // At z=0: Phi(0)=0.5, phi(0)=1/sqrt(2*pi)
        // CRPS = 1 * (0 * 0 + 2/sqrt(2*pi) - 1/sqrt(pi))
        // CRPS = 2/sqrt(2*pi) - 1/sqrt(pi) = sqrt(2/pi) - 1/sqrt(pi) = 1/sqrt(pi) * (sqrt(2) - 1)
        // ≈ 0.23370
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var score = crps.Score(normal, 0.0);
        Assert.Equal(1.0 / Math.Sqrt(Math.PI) * (Math.Sqrt(2) - 1), score, LooseTolerance);
    }

    [Fact]
    public void CRPSScore_NormalDistribution_FarFromMean_HigherScore()
    {
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var scoreAtMean = crps.Score(normal, 0.0);
        var scoreFar = crps.Score(normal, 5.0);
        Assert.True(scoreFar > scoreAtMean, "CRPS should be larger for observations far from mean");
    }

    [Fact]
    public void CRPSScore_NormalDistribution_NarrowDistribution_LowerScoreAtMean()
    {
        var crps = new CRPSScore<double>();
        var narrow = new NormalDistribution<double>(0.0, 0.01); // variance = 0.01, sigma = 0.1
        var wide = new NormalDistribution<double>(0.0, 100.0);  // variance = 100, sigma = 10
        var scoreNarrow = crps.Score(narrow, 0.0);
        var scoreWide = crps.Score(wide, 0.0);
        Assert.True(scoreNarrow < scoreWide, "Narrow distribution should have lower CRPS at mean");
    }

    [Fact]
    public void CRPSScore_NormalDistribution_ScoresScaleWithSigma()
    {
        // CRPS for Normal at the mean = sigma * (sqrt(2/pi) - 1/sqrt(pi))
        // So CRPS should scale linearly with sigma
        var crps = new CRPSScore<double>();
        var normal1 = new NormalDistribution<double>(0.0, 1.0);  // sigma=1
        var normal4 = new NormalDistribution<double>(0.0, 4.0);  // sigma=2
        var score1 = crps.Score(normal1, 0.0);
        var score4 = crps.Score(normal4, 0.0);
        Assert.Equal(2.0, score4 / score1, LooseTolerance);
    }

    #endregion

    #region CRPSScore - Laplace Distribution

    [Fact]
    public void CRPSScore_LaplaceDistribution_AtMean_IsNonNegative()
    {
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        var score = crps.Score(laplace, 0.0);
        Assert.True(score >= 0, $"CRPS should be non-negative, got {score}");
    }

    [Fact]
    public void CRPSScore_LaplaceDistribution_KnownValue()
    {
        // For Laplace at mean: CRPS = |y-mu| + b*exp(-|y-mu|/b) - 3b/4
        // At y=mu: CRPS = 0 + b*1 - 3b/4 = b/4
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        var score = crps.Score(laplace, 0.0);
        Assert.Equal(0.25, score, Tolerance);
    }

    [Fact]
    public void CRPSScore_LaplaceDistribution_FarFromMean_HigherScore()
    {
        var crps = new CRPSScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        var scoreAtMean = crps.Score(laplace, 0.0);
        var scoreFar = crps.Score(laplace, 10.0);
        Assert.True(scoreFar > scoreAtMean);
    }

    #endregion

    #region CRPSScore - MeanScore

    [Fact]
    public void CRPSScore_MeanScore_AveragesCorrectly()
    {
        var crps = new CRPSScore<double>();
        var dist = new NormalDistribution<double>(0.0, 1.0);
        var distributions = new IParametricDistribution<double>[] { dist, dist, dist };
        var observations = new Vector<double>([0.0, 0.0, 0.0]);
        var meanScore = crps.MeanScore(distributions, observations);
        var singleScore = crps.Score(dist, 0.0);
        Assert.Equal(singleScore, meanScore, Tolerance);
    }

    [Fact]
    public void CRPSScore_MeanScore_MismatchedLengths_Throws()
    {
        var crps = new CRPSScore<double>();
        var distributions = new IParametricDistribution<double>[] { new NormalDistribution<double>() };
        var observations = new Vector<double>([0.0, 1.0]);
        Assert.Throws<ArgumentException>(() => crps.MeanScore(distributions, observations));
    }

    [Fact]
    public void CRPSScore_MeanScore_Empty_Throws()
    {
        var crps = new CRPSScore<double>();
        var distributions = Array.Empty<IParametricDistribution<double>>();
        var observations = new Vector<double>(0);
        Assert.Throws<ArgumentException>(() => crps.MeanScore(distributions, observations));
    }

    #endregion

    #region CRPSScore - Gradient

    [Fact]
    public void CRPSScore_ScoreGradient_Normal_ReturnsCorrectDimension()
    {
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var gradient = crps.ScoreGradient(normal, 0.5);
        Assert.Equal(2, gradient.Length); // [d/dMean, d/dVariance]
    }

    [Fact]
    public void CRPSScore_ScoreGradient_Normal_MeanGradientSign()
    {
        // When observation > mean, gradient w.r.t. mean should be negative
        // (increasing mean would decrease CRPS)
        var crps = new CRPSScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var gradient = crps.ScoreGradient(normal, 2.0);
        Assert.True(gradient[0] < 0, "Gradient w.r.t. mean should be negative when obs > mean");
    }

    #endregion

    #region LogScore - Construction and Properties

    [Fact]
    public void LogScore_DefaultConstruction_NameAndProperties()
    {
        var logScore = new LogScore<double>();
        Assert.Equal("LogScore", logScore.Name);
        Assert.True(logScore.IsMinimized);
    }

    #endregion

    #region LogScore - Normal Distribution

    [Fact]
    public void LogScore_NormalDistribution_AtMean_IsNonNegative()
    {
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var score = logScore.Score(normal, 0.0);
        // -log(pdf(mean)) = -log(1/sqrt(2*pi)) = 0.5*log(2*pi) ≈ 0.9189
        Assert.True(score > 0, $"LogScore should be positive (negative log likelihood), got {score}");
    }

    [Fact]
    public void LogScore_NormalDistribution_KnownValue_AtMean()
    {
        // For standard normal at mean: -log(pdf(0)) = -log(1/sqrt(2*pi)) = 0.5*ln(2*pi)
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var score = logScore.Score(normal, 0.0);
        double expected = 0.5 * Math.Log(2 * Math.PI);
        Assert.Equal(expected, score, Tolerance);
    }

    [Fact]
    public void LogScore_NormalDistribution_FarFromMean_HigherScore()
    {
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var scoreAtMean = logScore.Score(normal, 0.0);
        var scoreFar = logScore.Score(normal, 5.0);
        Assert.True(scoreFar > scoreAtMean, "LogScore should be higher for unlikely observations");
    }

    [Fact]
    public void LogScore_NormalDistribution_SymmetricAroundMean()
    {
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var scorePositive = logScore.Score(normal, 2.0);
        var scoreNegative = logScore.Score(normal, -2.0);
        Assert.Equal(scorePositive, scoreNegative, Tolerance);
    }

    #endregion

    #region LogScore - Gradient

    [Fact]
    public void LogScore_ScoreGradient_Normal_ReturnsCorrectDimension()
    {
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var gradient = logScore.ScoreGradient(normal, 0.5);
        Assert.Equal(2, gradient.Length); // [d/dMean, d/dVariance]
    }

    [Fact]
    public void LogScore_ScoreGradient_Normal_MeanGradientSign()
    {
        // When observation > mean, gradient w.r.t. mean should be negative
        // (increasing mean would move distribution toward observation, reducing NLL)
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var gradient = logScore.ScoreGradient(normal, 2.0);
        Assert.True(gradient[0] < 0, "Gradient w.r.t. mean should be negative when obs > mean");
    }

    [Fact]
    public void LogScore_ScoreGradient_Normal_ZeroAtMean()
    {
        // At the mean, the gradient w.r.t. mean should be zero
        var logScore = new LogScore<double>();
        var normal = new NormalDistribution<double>(0.0, 1.0);
        var gradient = logScore.ScoreGradient(normal, 0.0);
        Assert.Equal(0.0, gradient[0], Tolerance);
    }

    #endregion

    #region LogScore - Laplace Distribution

    [Fact]
    public void LogScore_LaplaceDistribution_AtMean()
    {
        // For Laplace(0,1) at mean: -log(1/(2b)) = log(2b) = log(2)
        var logScore = new LogScore<double>();
        var laplace = new LaplaceDistribution<double>(0.0, 1.0);
        var score = logScore.Score(laplace, 0.0);
        Assert.Equal(Math.Log(2.0), score, Tolerance);
    }

    #endregion

    #region Proper Scoring Rule Property

    [Fact]
    public void CRPSScore_ProperScoringRule_TrueDistributionMinimizes()
    {
        // A proper scoring rule should give a better (lower for minimized) score
        // when the predicted distribution matches the true distribution
        var crps = new CRPSScore<double>();
        var trueDistribution = new NormalDistribution<double>(5.0, 1.0);
        var wrongDistribution = new NormalDistribution<double>(0.0, 1.0);

        // Generate "observations" at the true mean
        var observation = 5.0;
        var scoreTrue = crps.Score(trueDistribution, observation);
        var scoreWrong = crps.Score(wrongDistribution, observation);
        Assert.True(scoreTrue < scoreWrong, "Proper scoring rule should favor the true distribution");
    }

    [Fact]
    public void LogScore_ProperScoringRule_TrueDistributionMinimizes()
    {
        var logScore = new LogScore<double>();
        var trueDistribution = new NormalDistribution<double>(5.0, 1.0);
        var wrongDistribution = new NormalDistribution<double>(0.0, 1.0);

        var observation = 5.0;
        var scoreTrue = logScore.Score(trueDistribution, observation);
        var scoreWrong = logScore.Score(wrongDistribution, observation);
        Assert.True(scoreTrue < scoreWrong, "Proper scoring rule should favor the true distribution");
    }

    #endregion
}
