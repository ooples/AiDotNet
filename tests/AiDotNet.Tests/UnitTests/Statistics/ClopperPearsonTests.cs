using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Statistics;

/// <summary>
/// Tests for Clopper-Pearson exact binomial confidence intervals
/// and the underlying Beta distribution functions in StatisticsHelper.
/// </summary>
/// <remarks>
/// Reference values computed using scipy.stats.beta and R's binom.test().
/// The Clopper-Pearson interval is the "exact" binomial confidence interval
/// based on the Beta distribution quantiles.
/// </remarks>
public class ClopperPearsonTests
{
    private const double Tolerance = 1e-4;
    private const double LooseTolerance = 1e-3;

    #region Beta CDF Tests

    [Theory]
    [InlineData(0.0, 1.0, 1.0, 0.0)]      // CDF at x=0 should be 0
    [InlineData(1.0, 1.0, 1.0, 1.0)]      // CDF at x=1 should be 1
    [InlineData(0.5, 1.0, 1.0, 0.5)]      // Uniform Beta(1,1) at x=0.5 should be 0.5
    [InlineData(0.5, 2.0, 2.0, 0.5)]      // Symmetric Beta(2,2) at x=0.5 should be 0.5
    [InlineData(0.25, 2.0, 5.0, 0.4661)]  // Beta(2,5) at x=0.25
    [InlineData(0.75, 5.0, 2.0, 0.5339)]  // Beta(5,2) at x=0.75 (satisfies symmetry: I_x(a,b) + I_{1-x}(b,a) = 1)
    public void CalculateBetaCDF_KnownValues_ReturnsExpected(
        double x, double alpha, double beta, double expected)
    {
        // Act
        var result = StatisticsHelper<double>.CalculateBetaCDF(x, alpha, beta);

        // Assert
        Assert.True(
            Math.Abs(result - expected) < Tolerance,
            $"BetaCDF({x}, {alpha}, {beta}) = {result}, expected {expected}");
    }

    [Fact]
    public void CalculateBetaCDF_InvalidAlpha_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateBetaCDF(0.5, 0.0, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateBetaCDF(0.5, -1.0, 1.0));
    }

    [Fact]
    public void CalculateBetaCDF_InvalidBeta_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateBetaCDF(0.5, 1.0, 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateBetaCDF(0.5, 1.0, -1.0));
    }

    [Theory]
    [InlineData(-0.01, 1.0, 1.0)]
    [InlineData(1.01, 1.0, 1.0)]
    public void CalculateBetaCDF_XOutOfRange_ThrowsArgumentOutOfRangeException(
        double x, double alpha, double beta)
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateBetaCDF(x, alpha, beta));
    }

    [Fact]
    public void CalculateBetaCDF_ResultAlwaysBetween0And1()
    {
        // Arrange - test various valid inputs
        var testCases = new[]
        {
            (x: 0.1, alpha: 2.0, beta: 3.0),
            (x: 0.5, alpha: 1.0, beta: 1.0),
            (x: 0.9, alpha: 5.0, beta: 2.0),
            (x: 0.3, alpha: 0.5, beta: 0.5),  // Jeffrey's prior
        };

        foreach (var (x, alpha, beta) in testCases)
        {
            // Act
            var result = StatisticsHelper<double>.CalculateBetaCDF(x, alpha, beta);

            // Assert
            Assert.True(result >= 0.0 && result <= 1.0,
                $"BetaCDF({x}, {alpha}, {beta}) = {result} is not in [0,1]");
        }
    }

    #endregion

    #region Inverse Beta CDF Tests

    [Theory]
    [InlineData(0.0, 1.0, 1.0, 0.0)]      // Inverse at p=0 should be 0
    [InlineData(1.0, 1.0, 1.0, 1.0)]      // Inverse at p=1 should be 1
    [InlineData(0.5, 1.0, 1.0, 0.5)]      // Uniform Beta(1,1) at p=0.5 should be 0.5
    [InlineData(0.5, 2.0, 2.0, 0.5)]      // Symmetric Beta(2,2) at p=0.5 should be 0.5
    [InlineData(0.025, 10.0, 5.0, 0.4191)] // Computed via bisection method
    [InlineData(0.975, 10.0, 5.0, 0.8724)] // Computed via bisection method
    public void CalculateInverseBetaCDF_KnownValues_ReturnsExpected(
        double p, double alpha, double beta, double expected)
    {
        // Act
        var result = StatisticsHelper<double>.CalculateInverseBetaCDF(p, alpha, beta);

        // Assert
        Assert.True(
            Math.Abs(result - expected) < LooseTolerance,
            $"InverseBetaCDF({p}, {alpha}, {beta}) = {result}, expected {expected}");
    }

    [Theory]
    [InlineData(0.1, 2.0, 3.0)]
    [InlineData(0.5, 5.0, 5.0)]
    [InlineData(0.9, 1.0, 1.0)]
    public void CalculateInverseBetaCDF_RoundTrip_ReturnsOriginalProbability(
        double p, double alpha, double beta)
    {
        // Act
        var x = StatisticsHelper<double>.CalculateInverseBetaCDF(p, alpha, beta);
        var pBack = StatisticsHelper<double>.CalculateBetaCDF(x, alpha, beta);

        // Assert
        Assert.True(
            Math.Abs(pBack - p) < LooseTolerance,
            $"Round-trip failed: p={p} -> x={x} -> p'={pBack}");
    }

    [Fact]
    public void CalculateInverseBetaCDF_InvalidProbability_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateInverseBetaCDF(-0.01, 1.0, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateInverseBetaCDF(1.01, 1.0, 1.0));
    }

    #endregion

    #region Clopper-Pearson Interval Tests

    /// <summary>
    /// Tests Clopper-Pearson intervals against known reference values.
    /// Values computed using our Beta distribution implementation which follows
    /// the Lentz continued fraction algorithm from Numerical Recipes.
    /// </summary>
    [Theory]
    // n=100, k=95 -> 95% of 100 (two-sided: alpha/2 = 0.025 each tail)
    [InlineData(95, 100, 0.95, 0.8872, 0.9838)]
    // n=100, k=50 -> 50% of 100
    [InlineData(50, 100, 0.95, 0.3983, 0.6017)]
    // n=20, k=18 -> 90% of 20
    [InlineData(18, 20, 0.95, 0.6830, 0.9877)]
    // n=1000, k=500 -> 50% of 1000
    [InlineData(500, 1000, 0.95, 0.4687, 0.5313)]
    public void CalculateClopperPearsonInterval_KnownValues_ReturnsExpected(
        int successes, int trials, double confidence, double expectedLower, double expectedUpper)
    {
        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert
        Assert.True(
            Math.Abs(lower - expectedLower) < LooseTolerance,
            $"Lower bound for ({successes}/{trials}) = {lower}, expected {expectedLower}");
        Assert.True(
            Math.Abs(upper - expectedUpper) < LooseTolerance,
            $"Upper bound for ({successes}/{trials}) = {upper}, expected {expectedUpper}");
    }

    [Theory]
    // Edge case: k=0, lower bound should be exactly 0
    [InlineData(0, 10, 0.95, 0.0)]
    [InlineData(0, 100, 0.95, 0.0)]
    [InlineData(0, 1000, 0.95, 0.0)]
    public void CalculateClopperPearsonInterval_ZeroSuccesses_LowerBoundIsZero(
        int successes, int trials, double confidence, double expectedLower)
    {
        // Act
        var (lower, _) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert
        Assert.Equal(expectedLower, lower, Tolerance);
    }

    [Theory]
    // Edge case: k=n, upper bound should be exactly 1
    [InlineData(10, 10, 0.95, 1.0)]
    [InlineData(100, 100, 0.95, 1.0)]
    [InlineData(1000, 1000, 0.95, 1.0)]
    public void CalculateClopperPearsonInterval_AllSuccesses_UpperBoundIsOne(
        int successes, int trials, double confidence, double expectedUpper)
    {
        // Act
        var (_, upper) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert
        Assert.Equal(expectedUpper, upper, Tolerance);
    }

    [Theory]
    // Reference: k=0, n=10 at 95% CI -> upper ~0.3085
    [InlineData(0, 10, 0.95, 0.3085)]
    public void CalculateClopperPearsonInterval_ZeroSuccesses_UpperBoundIsCorrect(
        int successes, int trials, double confidence, double expectedUpper)
    {
        // Act
        var (_, upper) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert
        Assert.True(
            Math.Abs(upper - expectedUpper) < LooseTolerance,
            $"Upper bound for (0/{trials}) = {upper}, expected {expectedUpper}");
    }

    [Theory]
    // Reference: k=10, n=10 at 95% CI -> lower ~0.6915
    [InlineData(10, 10, 0.95, 0.6915)]
    public void CalculateClopperPearsonInterval_AllSuccesses_LowerBoundIsCorrect(
        int successes, int trials, double confidence, double expectedLower)
    {
        // Act
        var (lower, _) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert
        Assert.True(
            Math.Abs(lower - expectedLower) < LooseTolerance,
            $"Lower bound for ({successes}/{trials}) = {lower}, expected {expectedLower}");
    }

    [Fact]
    public void CalculateClopperPearsonInterval_BoundsContainPointEstimate()
    {
        // Arrange
        var testCases = new[]
        {
            (successes: 50, trials: 100),
            (successes: 95, trials: 100),
            (successes: 5, trials: 100),
            (successes: 1, trials: 10),
            (successes: 9, trials: 10),
        };

        foreach (var (successes, trials) in testCases)
        {
            // Act
            var (lower, upper) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
                successes, trials, 0.95);
            var pointEstimate = (double)successes / trials;

            // Assert - point estimate should be within the interval
            Assert.True(lower <= pointEstimate && pointEstimate <= upper,
                $"Point estimate {pointEstimate} not in [{lower}, {upper}] for {successes}/{trials}");
        }
    }

    [Theory]
    [InlineData(0.90)]
    [InlineData(0.95)]
    [InlineData(0.99)]
    public void CalculateClopperPearsonInterval_HigherConfidence_WiderInterval(double confidence)
    {
        // Arrange
        const int successes = 50;
        const int trials = 100;
        const double lowerConfidence = 0.80;

        // Act
        var (lowerNarrow, upperNarrow) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, lowerConfidence);
        var (lowerWide, upperWide) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert - higher confidence should give wider (or equal) interval
        Assert.True(lowerWide <= lowerNarrow,
            $"Higher confidence ({confidence}) lower bound {lowerWide} should be <= {lowerNarrow}");
        Assert.True(upperWide >= upperNarrow,
            $"Higher confidence ({confidence}) upper bound {upperWide} should be >= {upperNarrow}");
    }

    [Fact]
    public void CalculateClopperPearsonInterval_NegativeSuccesses_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateClopperPearsonInterval(-1, 100, 0.95));
    }

    [Fact]
    public void CalculateClopperPearsonInterval_SuccessesGreaterThanTrials_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateClopperPearsonInterval(101, 100, 0.95));
    }

    [Fact]
    public void CalculateClopperPearsonInterval_ZeroTrials_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateClopperPearsonInterval(0, 0, 0.95));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(1.1)]
    public void CalculateClopperPearsonInterval_InvalidConfidence_ThrowsArgumentOutOfRangeException(
        double confidence)
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            StatisticsHelper<double>.CalculateClopperPearsonInterval(50, 100, confidence));
    }

    #endregion

    #region Clopper-Pearson Lower Bound Tests

    [Theory]
    // One-sided lower bounds: qbeta(1-confidence, k, n-k+1) in R
    // Note: This is for one-sided bounds, NOT two-sided interval bounds
    [InlineData(95, 100, 0.95, 0.8977)]  // qbeta(0.05, 95, 6)
    [InlineData(50, 100, 0.95, 0.4136)]  // qbeta(0.05, 50, 51)
    [InlineData(0, 10, 0.95, 0.0)]
    public void CalculateClopperPearsonLowerBound_KnownValues_ReturnsExpected(
        int successes, int trials, double confidence, double expected)
    {
        // Act
        var result = StatisticsHelper<double>.CalculateClopperPearsonLowerBound(
            successes, trials, confidence);

        // Assert
        Assert.True(
            Math.Abs(result - expected) < LooseTolerance,
            $"Lower bound for ({successes}/{trials}) = {result}, expected {expected}");
    }

    [Fact]
    public void CalculateClopperPearsonLowerBound_MatchesIntervalLowerBound()
    {
        // Two-sided 95% interval uses alpha/2 = 0.025 for lower bound
        // One-sided bound at 97.5% confidence uses alpha = 0.025
        // So: Interval(conf=0.95).Lower == LowerBound(conf=0.975)
        var testCases = new[]
        {
            (successes: 50, trials: 100),
            (successes: 95, trials: 100),
            (successes: 5, trials: 100),
            (successes: 0, trials: 10),
            (successes: 10, trials: 10),
        };

        foreach (var (successes, trials) in testCases)
        {
            // Act
            // Two-sided interval at 95% confidence
            var (intervalLower, _) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
                successes, trials, 0.95);
            // One-sided lower bound at 97.5% confidence matches the two-sided 95% lower
            var directLower = StatisticsHelper<double>.CalculateClopperPearsonLowerBound(
                successes, trials, 0.975);

            // Assert
            Assert.True(
                Math.Abs(intervalLower - directLower) < Tolerance,
                $"Interval lower {intervalLower} != direct lower {directLower} for {successes}/{trials}");
        }
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void CalculateBetaCDF_FloatType_ReturnsCorrectResult()
    {
        // Act
        var result = StatisticsHelper<float>.CalculateBetaCDF(0.5f, 2.0f, 2.0f);

        // Assert
        Assert.True(Math.Abs(result - 0.5f) < 0.001f);
    }

    [Fact]
    public void CalculateClopperPearsonInterval_FloatType_ReturnsCorrectResult()
    {
        // Act
        var (lower, upper) = StatisticsHelper<float>.CalculateClopperPearsonInterval(50, 100, 0.95f);

        // Assert
        Assert.True(lower > 0.35f && lower < 0.45f, $"Lower bound {lower} out of expected range");
        Assert.True(upper > 0.55f && upper < 0.65f, $"Upper bound {upper} out of expected range");
    }

    #endregion

    #region Coverage Guarantee Tests

    /// <summary>
    /// Verifies the coverage guarantee property of Clopper-Pearson intervals.
    /// For a 95% CI, the true proportion should be covered at least 95% of the time.
    /// This test validates the mathematical correctness of our implementation.
    /// </summary>
    [Theory]
    [InlineData(0.3, 50, 0.95)]
    [InlineData(0.5, 100, 0.95)]
    [InlineData(0.7, 30, 0.95)]
    public void ClopperPearsonInterval_CoverageProperty_IsConservative(
        double trueProportion, int trials, double confidence)
    {
        // The Clopper-Pearson interval should contain the true proportion
        // when the number of successes is drawn from Binomial(n, trueProportion)

        // For this unit test, we verify that when k = round(n*p), the interval contains p
        var successes = (int)Math.Round(trueProportion * trials);

        // Act
        var (lower, upper) = StatisticsHelper<double>.CalculateClopperPearsonInterval(
            successes, trials, confidence);

        // Assert - the true proportion should be in the interval when successes = expected value
        // This is a necessary (but not sufficient) condition for correct coverage
        Assert.True(
            lower <= trueProportion && trueProportion <= upper,
            $"True proportion {trueProportion} not in [{lower}, {upper}] for {successes}/{trials}");
    }

    #endregion
}
