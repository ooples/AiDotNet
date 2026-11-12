using AiDotNet.SelfSupervisedLearning.Evaluation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.SelfSupervisedLearning;

/// <summary>
/// Reference values for <see cref="SelfSupervisedLearningMetrics{T}"/>, derived by hand.
/// </summary>
public class SelfSupervisedLearningMetricsTests
{
    private const double Tolerance = 1e-10;

    /// <summary>
    /// Effective rank (Roy and Vetterli, 2007) is exp of the entropy of the normalized eigenvalue
    /// spectrum of the covariance matrix.
    /// </summary>
    /// <remarks>
    /// The rows (2, 1), (-1, 0), (-1, -1) have zero mean, so the unbiased covariance is
    /// X^T X / 2 = [[3, 1.5], [1.5, 1]]. Its trace is 4 and its determinant is 0.75, so the
    /// eigenvalues are 2 +/- sqrt(4 - 0.75) = 2 +/- sqrt(13) / 2. The metric adds 1e-8 to each
    /// |lambda| before normalizing. The unequal diagonal forces a real rotation, which a
    /// wrongly signed Jacobi sweep gets wrong (it returned 1.31 and 2.69).
    /// </remarks>
    [Fact]
    public void ComputeEffectiveRank_MatchesHandDerivedSpectrum()
    {
        var representations = new Tensor<double>(new[] { 2.0, 1.0, -1.0, 0.0, -1.0, -1.0 }, new[] { 3, 2 });

        double root = Math.Sqrt(13.0) / 2.0;
        double m1 = 2.0 + root + 1e-8;
        double m2 = 2.0 - root + 1e-8;
        double p1 = m1 / (m1 + m2);
        double p2 = m2 / (m1 + m2);
        double expected = Math.Exp(-(p1 * Math.Log(p1) + p2 * Math.Log(p2)));

        var metrics = new SelfSupervisedLearningMetrics<double>();

        Assert.Equal(expected, metrics.ComputeEffectiveRank(representations), Tolerance);
    }

    /// <summary>
    /// Rows (1, 1), (-1, 1), (1, -1), (-1, -1) have zero mean and covariance (4/3) I, so both
    /// eigenvalues are 4/3, the spectrum is uniform and the effective rank is exactly 2.
    /// </summary>
    [Fact]
    public void ComputeEffectiveRank_IsotropicCovariance_IsFullRank()
    {
        var representations = new Tensor<double>(
            new[] { 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0 }, new[] { 4, 2 });

        var metrics = new SelfSupervisedLearningMetrics<double>();

        Assert.Equal(2.0, metrics.ComputeEffectiveRank(representations), Tolerance);
    }
}
