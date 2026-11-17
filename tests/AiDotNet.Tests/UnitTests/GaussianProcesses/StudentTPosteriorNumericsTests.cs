using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.GaussianProcesses;

[Collection("EngineCurrentGlobalState")]
public sealed class StudentTPosteriorNumericsTests
{
    private static void OnCpu(Action action)
    {
        var previous = AiDotNetEngine.Current;
        try { AiDotNetEngine.Current = new CpuEngine(); action(); }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(0, 0, 0.0182770113056)]
    [InlineData(1, 0.981424073800, 0.0192022829205)]
    [InlineData(10, 0.534485581272, 1.061496773323)]
    [InlineData(100, 0.0500298385471, 1.000500895929)]
    [InlineData(1000, 0.00500002980039, 1.000005000089)]
    public void IndependentDensityOracle_MatchesReferenceAndRefinement(double target, double mean, double variance)
    {
        var coarse = GaussianProcessPosteriorContract.IntegrateSingleObservation(target, 1, 4, 0.1);
        var refined = GaussianProcessPosteriorContract.IntegrateSingleObservation(target, 1, 4, 0.1, 80000, 14);
        GaussianProcessPosteriorContract.AssertClose(mean, coarse.Mean, 1e-10, "reference mean");
        GaussianProcessPosteriorContract.AssertClose(variance, coarse.Variance, 1e-10, "reference variance");
        GaussianProcessPosteriorContract.AssertClose(refined.Mean, coarse.Mean, 1e-10, "refined mean");
        GaussianProcessPosteriorContract.AssertClose(refined.Variance, coarse.Variance, 1e-10, "refined variance");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(100)]
    [InlineData(1000)]
    public void SingleObservation_MatchesIndependentPosteriorMeanAndVariance(double target)
        => OnCpu(() =>
        {
            using var model = new StudentTGaussianProcess<double>(new GaussianKernel<double>(), tolerance: 1e-9);
            var x = new Matrix<double>(1, 1);
            var point = new Vector<double>(1);
            const double prior = 1.000001;
            var oracle = GaussianProcessPosteriorContract.IntegrateSingleObservation(target, prior, 4, 0.1);
            model.Fit(x, new Vector<double>(new[] { target }));
            var actual = model.Predict(point);
            double projection = 1 / prior;
            GaussianProcessPosteriorContract.AssertClose(projection * oracle.Mean, actual.mean, 2e-8, "posterior mean");
            double expectedVariance = 1 - projection + projection * projection * oracle.Variance + 0.02;
            GaussianProcessPosteriorContract.AssertClose(expectedVariance, actual.variance, 2e-8, "predictive variance");
            if (target == 10) Assert.True(actual.variance > 1.02, "A valid negative site must increase variance above the prior.");

            // Fixed zero-mean prior implies odd means and even variances.
            model.Fit(x, new Vector<double>(new[] { -target }));
            var reflected = model.Predict(point);
            GaussianProcessPosteriorContract.AssertClose(-actual.mean, reflected.mean, 2e-8, "reflected mean");
            GaussianProcessPosteriorContract.AssertClose(actual.variance, reflected.variance, 2e-8, "reflected variance");
            var far = model.Predict(new Vector<double>(new[] { 100.0 }));
            GaussianProcessPosteriorContract.AssertClose(0, far.mean, 1e-12, "far prior mean");
            GaussianProcessPosteriorContract.AssertClose(1.02, far.variance, 1e-12, "far prior variance");
        });

    [Theory]
    [InlineData(2.5, 0.05)]
    [InlineData(8, 0.5)]
    public void LikelihoodScaleAndDegreesOfFreedom_AgreeWithIndependentDensity(double nu, double scale)
        => OnCpu(() =>
        {
            using var model = new StudentTGaussianProcess<double>(new GaussianKernel<double>(), nu, scale, tolerance: 1e-9);
            Assert.True(GaussianProcessPosteriorContract.VerifyFixedPrior(model, 1, 0, 1, 10));
        });

    [Theory]
    [InlineData(0.5, 1)]
    [InlineData(0, 10)]
    public void MultipleObservations_MatchIndependentEpFixedPoint_NotExactJointPosterior(double y0, double y1)
        => OnCpu(() =>
        {
            var kernel = new GaussianKernel<double>();
            using var model = new StudentTGaussianProcess<double>(kernel, maxIterations: 500, tolerance: 1e-9);
            var x = new Matrix<double>(2, 1);
            x[1, 0] = 1;
            double diagonal = kernel.Calculate(x.GetRow(0), x.GetRow(0)) + 1e-6;
            double offDiagonal = kernel.Calculate(x.GetRow(0), x.GetRow(1));
            var expected = ReferenceEp(diagonal, offDiagonal, new[] { y0, y1 });
            model.Fit(x, new Vector<double>(new[] { y0, y1 }));
            double determinant = diagonal * diagonal - offDiagonal * offDiagonal;
            for (int i = 0; i < 2; i++)
            {
                double k0 = kernel.Calculate(x.GetRow(i), x.GetRow(0));
                double k1 = kernel.Calculate(x.GetRow(i), x.GetRow(1));
                double a0 = (diagonal * k0 - offDiagonal * k1) / determinant;
                double a1 = (diagonal * k1 - offDiagonal * k0) / determinant;
                double mean = a0 * expected.Mean[0] + a1 * expected.Mean[1];
                double variance = 1 - k0 * a0 - k1 * a1 + a0 * a0 * expected.V00 +
                    2 * a0 * a1 * expected.V01 + a1 * a1 * expected.V11 + 0.02;
                var actual = model.Predict(x.GetRow(i));
                GaussianProcessPosteriorContract.AssertClose(mean, actual.mean, 2e-7, "EP reference mean");
                GaussianProcessPosteriorContract.AssertClose(variance, actual.variance, 2e-7, "EP reference variance");
            }
        });

    [Theory]
    [InlineData(0.5)]
    [InlineData(1)]
    [InlineData(2)]
    public void InfiniteStudentTNoiseVariance_IsReportedExplicitly(double nu)
        => OnCpu(() =>
        {
            using var model = new StudentTGaussianProcess<double>(new GaussianKernel<double>(), nu: nu, tolerance: 1e-9);
            model.Fit(new Matrix<double>(1, 1), new Vector<double>(new[] { 0.0 }));
            var prediction = model.Predict(new Vector<double>(1));
            GaussianProcessPosteriorContract.AssertClose(0, prediction.mean, 1e-9, "finite symmetric latent mean");
            Assert.Equal(double.PositiveInfinity, prediction.variance);
        });

    [Theory]
    [InlineData(1e20)]
    [InlineData(1e100)]
    public void LargeDegreesOfFreedom_MatchesIndependentGaussianConjugateLimit(double nu)
        => OnCpu(() =>
        {
            const double prior = 1.000001, noise = .01, observation = 1;
            using var model = new StudentTGaussianProcess<double>(new GaussianKernel<double>(),
                nu: nu, scale: .1, tolerance: 1e-9);
            model.Fit(new Matrix<double>(1, 1), new Vector<double>(new[] { observation }));
            var actual = model.Predict(new Vector<double>(1));
            // Normal-normal conjugacy, then conditional-prior projection to
            // the noise-free query (training covariance includes jitter).
            double latentMean = prior * observation / (prior + noise);
            double latentVariance = prior * noise / (prior + noise);
            double projection = 1 / prior;
            GaussianProcessPosteriorContract.AssertClose(projection * latentMean, actual.mean, 2e-8,
                "Gaussian-limit predictive mean");
            GaussianProcessPosteriorContract.AssertClose(1 - projection + projection * projection * latentVariance + noise,
                actual.variance, 2e-8, "Gaussian-limit predictive variance");
        });

    [Theory]
    [InlineData(1e-12, 4)]
    [InlineData(.5, 1)]
    public void InsufficientFixedPointProgress_ReportsNonconvergenceInsteadOfPublishingPrior(double damping, int iterations)
        => OnCpu(() =>
        {
            using var model = new StudentTGaussianProcess<double>(new GaussianKernel<double>(),
                damping: damping, maxIterations: iterations);
            var error = Assert.Throws<InvalidOperationException>(() =>
                model.Fit(new Matrix<double>(1, 1), new Vector<double>(new[] { 1.0 })));
            Assert.Contains("did not converge", error.Message);
            Assert.Throws<InvalidOperationException>(() => model.Predict(new Vector<double>(1)));
        });

    private static (double[] Mean, double V00, double V01, double V11) ReferenceEp(double diagonal, double offDiagonal, double[] targets)
    {
        // Independent scalar 2x2 precision algebra plus direct density
        // integration. EP matches tilted marginal moments, not the exact
        // multi-observation posterior of the Student-t likelihood.
        double detK = diagonal * diagonal - offDiagonal * offDiagonal;
        double p = diagonal / detK, q = -offDiagonal / detK;
        var tau = new double[2]; var eta = new double[2]; var mean = new double[2];
        double v00 = diagonal, v01 = offDiagonal, v11 = diagonal;
        for (int iteration = 0; iteration < 500; iteration++)
        {
            double change = 0;
            var nextTau = new double[2]; var nextEta = new double[2];
            for (int i = 0; i < 2; i++)
            {
                double cavityPrecision = 1 / (i == 0 ? v00 : v11) - tau[i];
                Assert.True(cavityPrecision > 0);
                double cavityMean = (mean[i] / (i == 0 ? v00 : v11) - eta[i]) / cavityPrecision;
                var tilted = GaussianProcessPosteriorContract.IntegrateSingleObservation(
                    targets[i] - cavityMean, 1 / cavityPrecision, 4, 0.1);
                double tiltedMean = tilted.Mean + cavityMean;
                nextTau[i] = 0.5 * tau[i] + 0.5 * (1 / tilted.Variance - cavityPrecision);
                nextEta[i] = 0.5 * eta[i] + 0.5 * (tiltedMean / tilted.Variance - cavityMean * cavityPrecision);
                change = Math.Max(change, Math.Abs(nextTau[i] - tau[i]) / (1 + Math.Abs(nextTau[i])) +
                    Math.Abs(nextEta[i] - eta[i]) / (1 + Math.Abs(nextEta[i])));
            }
            tau = nextTau; eta = nextEta;
            double determinant = (p + tau[0]) * (p + tau[1]) - q * q;
            Assert.True(determinant > 0 && p + tau[0] > 0 && p + tau[1] > 0);
            v00 = (p + tau[1]) / determinant; v11 = (p + tau[0]) / determinant; v01 = -q / determinant;
            mean[0] = v00 * eta[0] + v01 * eta[1]; mean[1] = v01 * eta[0] + v11 * eta[1];
            if (change < 1e-10) return (mean, v00, v01, v11);
        }
        throw new InvalidOperationException("Independent EP reference did not converge.");
    }
}
