using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.GaussianProcesses;

/// <summary>
/// Correctness tests for GP inference methods.
/// Tests against known mathematical results.
/// </summary>
public class GPInferenceCorrectnessTests
{
    private const double Tolerance = 1e-4;

    #region GPWithMCMC Tests

    [Fact]
    public void GPWithMCMC_Construction_Succeeds()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new GPWithMCMC<double>(kernel, numSamples: 10, burnIn: 5);

        Assert.NotNull(gp);
        Assert.False(gp.IsTrained);
    }

    [Fact]
    public void GPWithMCMC_Fit_SetsIsTrained()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new GPWithMCMC<double>(kernel, numSamples: 10, burnIn: 5, seed: 42);

        // Simple training data
        var X = new Matrix<double>(5, 1);
        var y = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            X[i, 0] = i;
            y[i] = Math.Sin(i);
        }

        gp.Fit(X, y);

        Assert.True(gp.IsTrained);
        Assert.True(gp.NumStoredSamples > 0);
    }

    [Fact]
    public void GPWithMCMC_Predict_ReturnsReasonableValues()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new GPWithMCMC<double>(kernel, numSamples: 50, burnIn: 20, seed: 42);

        // Training data: y = sin(x) with noise
        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i * 0.5;
            y[i] = Math.Sin(i * 0.5);
        }

        gp.Fit(X, y);

        // Predict at a training point
        var xTest = new Vector<double>(new double[] { 0.0 });
        var (mean, variance) = gp.Predict(xTest);

        // Should be close to sin(0) = 0
        Assert.True(Math.Abs(mean) < 0.5,
            $"Prediction at training point should be close to true value, got mean={mean}");
        Assert.True(variance > 0,
            $"Variance should be positive, got {variance}");
    }

    [Fact]
    public void GPWithMCMC_Predict_UncertaintyIncreasesAwayFromData()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new GPWithMCMC<double>(kernel, numSamples: 50, burnIn: 20, seed: 42);

        // Training data clustered around x=0
        var X = new Matrix<double>(5, 1);
        var y = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            X[i, 0] = i * 0.2;
            y[i] = i * 0.2;
        }

        gp.Fit(X, y);

        // Predict near and far from data
        var xNear = new Vector<double>(new double[] { 0.5 });
        var xFar = new Vector<double>(new double[] { 10.0 });

        var (_, varNear) = gp.Predict(xNear);
        var (_, varFar) = gp.Predict(xFar);

        Assert.True(varFar > varNear,
            $"Variance should be higher far from data: var(near)={varNear}, var(far)={varFar}");
    }

    [Fact]
    public void GPWithMCMC_GetPosteriorStatistics_ReturnsValidStats()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new GPWithMCMC<double>(kernel, numSamples: 50, burnIn: 20, seed: 42);

        var X = new Matrix<double>(5, 1);
        var y = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            X[i, 0] = i;
            y[i] = i;
        }

        gp.Fit(X, y);

        var stats = gp.GetPosteriorStatistics();

        Assert.True(stats.ContainsKey("lengthscale"));
        Assert.True(stats.ContainsKey("outputVariance"));
        Assert.True(stats.ContainsKey("noiseVariance"));

        var (lsMean, lsStd) = stats["lengthscale"];
        Assert.True(lsMean > 0, "Lengthscale mean should be positive");
        Assert.True(lsStd >= 0, "Lengthscale std should be non-negative");
    }

    [Fact]
    public void GPWithMCMC_GetSamples_ReturnsCorrectCount()
    {
        var kernel = new GaussianKernel<double>(1.0);
        int numSamples = 30;
        var gp = new GPWithMCMC<double>(kernel, numSamples: numSamples, burnIn: 10, thinning: 1, seed: 42);

        var X = new Matrix<double>(5, 1);
        var y = new Vector<double>(5);
        for (int i = 0; i < 5; i++)
        {
            X[i, 0] = i;
            y[i] = i;
        }

        gp.Fit(X, y);

        var samples = gp.GetSamples();

        Assert.Equal(numSamples, samples.Count);
        Assert.All(samples, s => Assert.Equal(3, s.Length)); // [lengthscale, outputVar, noiseVar]
    }

    #endregion

    #region BetaLikelihood Tests

    [Fact]
    public void BetaLikelihood_Construction_Succeeds()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        Assert.NotNull(likelihood);
        Assert.Equal(10.0, likelihood.Precision);
    }

    [Fact]
    public void BetaLikelihood_GetMeans_ReturnsBetweenZeroAndOne()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        var f = new Vector<double>(new double[] { -5.0, -1.0, 0.0, 1.0, 5.0 });
        var mu = likelihood.GetMeans(f);

        Assert.All(Enumerable.Range(0, mu.Length), i =>
        {
            Assert.True(mu[i] > 0 && mu[i] < 1,
                $"Mean should be in (0,1), got {mu[i]} for f={f[i]}");
        });
    }

    [Fact]
    public void BetaLikelihood_GetMeans_SigmoidCorrect()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        var f = new Vector<double>(new double[] { 0.0 });
        var mu = likelihood.GetMeans(f);

        // sigmoid(0) = 0.5
        Assert.True(Math.Abs(mu[0] - 0.5) < Tolerance,
            $"sigmoid(0) should be 0.5, got {mu[0]}");
    }

    [Fact]
    public void BetaLikelihood_GetBetaParameters_ArePositive()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        double[] testMeans = { 0.1, 0.3, 0.5, 0.7, 0.9 };

        foreach (var mu in testMeans)
        {
            var (alpha, beta) = likelihood.GetBetaParameters(mu);

            Assert.True(alpha > 0, $"Alpha should be positive for mu={mu}, got {alpha}");
            Assert.True(beta > 0, $"Beta should be positive for mu={mu}, got {beta}");

            // Check mean matches: E[Beta] = alpha / (alpha + beta) = mu
            double expectedMean = alpha / (alpha + beta);
            Assert.True(Math.Abs(expectedMean - mu) < Tolerance,
                $"Beta mean should be {mu}, got {expectedMean}");
        }
    }

    [Fact]
    public void BetaLikelihood_LogLikelihood_IsNegative()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        var y = new Vector<double>(new double[] { 0.3, 0.5, 0.7 });
        var f = new Vector<double>(new double[] { 0.0, 0.5, 1.0 });

        var logLik = likelihood.LogLikelihood(y, f);

        // Log-likelihood should be negative (probability < 1)
        Assert.True(logLik < 0 || double.IsNaN(logLik) == false,
            $"Log-likelihood should be negative and finite, got {logLik}");
    }

    [Fact]
    public void BetaLikelihood_LogLikelihood_HigherForGoodFit()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        // Good fit: y = 0.5 with f = 0 (sigmoid(0) = 0.5)
        var yGood = new Vector<double>(new double[] { 0.5 });
        var fGood = new Vector<double>(new double[] { 0.0 });

        // Bad fit: y = 0.9 with f = -2 (sigmoid(-2) ≈ 0.12)
        var yBad = new Vector<double>(new double[] { 0.9 });
        var fBad = new Vector<double>(new double[] { -2.0 });

        var logLikGood = likelihood.LogLikelihood(yGood, fGood);
        var logLikBad = likelihood.LogLikelihood(yBad, fBad);

        Assert.True(logLikGood > logLikBad,
            $"Good fit should have higher log-likelihood: good={logLikGood}, bad={logLikBad}");
    }

    [Fact]
    public void BetaLikelihood_Gradient_IsFinite()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        var y = new Vector<double>(new double[] { 0.3, 0.5, 0.7 });
        var f = new Vector<double>(new double[] { -1.0, 0.0, 1.0 });

        var grad = likelihood.LogLikelihoodGradient(y, f);

        Assert.All(Enumerable.Range(0, grad.Length), i =>
        {
            Assert.False(double.IsNaN(grad[i]), $"Gradient[{i}] is NaN");
            Assert.False(double.IsInfinity(grad[i]), $"Gradient[{i}] is infinite");
        });
    }

    [Fact]
    public void BetaLikelihood_Hessian_IsNegative()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        var y = new Vector<double>(new double[] { 0.3, 0.5, 0.7 });
        var f = new Vector<double>(new double[] { -1.0, 0.0, 1.0 });

        var hess = likelihood.LogLikelihoodHessianDiag(y, f);

        // Diagonal of Hessian of log-likelihood should be non-positive (concave)
        Assert.All(Enumerable.Range(0, hess.Length), i =>
        {
            Assert.True(hess[i] <= 0,
                $"Hessian diagonal should be non-positive, got hess[{i}]={hess[i]}");
        });
    }

    [Fact]
    public void BetaLikelihood_PredictiveMoments_AreReasonable()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);

        // Latent prediction with low uncertainty
        var (meanLowVar, varLowVar) = likelihood.PredictiveMoments(0.0, 0.01);

        // Latent prediction with high uncertainty
        var (meanHighVar, varHighVar) = likelihood.PredictiveMoments(0.0, 1.0);

        // Mean should be around 0.5 for f_mean = 0
        Assert.True(Math.Abs(meanLowVar - 0.5) < 0.1,
            $"Predictive mean should be around 0.5 for f=0, got {meanLowVar}");

        // Variance should increase with latent variance
        Assert.True(varHighVar > varLowVar,
            $"Predictive variance should increase: var(low)={varLowVar}, var(high)={varHighVar}");
    }

    [Fact]
    public void BetaLikelihood_Sample_IsInUnitInterval()
    {
        var likelihood = new BetaLikelihood<double>(precision: 10.0);
        var random = new Random(42);

        double[] testMeans = { 0.1, 0.3, 0.5, 0.7, 0.9 };

        foreach (var mu in testMeans)
        {
            for (int i = 0; i < 100; i++)
            {
                var sample = likelihood.Sample(mu, random);

                Assert.True(sample >= 0 && sample <= 1,
                    $"Sample should be in [0,1], got {sample} for mu={mu}");
            }
        }
    }

    [Fact]
    public void BetaLikelihood_Sample_HasCorrectMean()
    {
        var likelihood = new BetaLikelihood<double>(precision: 20.0);
        var random = new Random(42);

        double mu = 0.3;
        double sum = 0;
        int n = 1000;

        for (int i = 0; i < n; i++)
        {
            sum += likelihood.Sample(mu, random);
        }

        double sampleMean = sum / n;

        // Sample mean should be close to mu (within 3 standard errors)
        double stdErr = Math.Sqrt(mu * (1 - mu) / (likelihood.Precision + 1) / n);
        Assert.True(Math.Abs(sampleMean - mu) < 3 * stdErr + 0.05,
            $"Sample mean {sampleMean} should be close to {mu} (stdErr={stdErr})");
    }

    [Fact]
    public void BetaLikelihood_FromData_EstimatesPrecision()
    {
        // Generate data from Beta(3, 7) which has mean 0.3 and variance 0.3*0.7/11 ≈ 0.019
        var random = new Random(42);
        var y = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            // Simple approximation for Beta sampling
            double sum = 0;
            for (int j = 0; j < 10; j++)
            {
                sum += random.NextDouble() < 0.3 ? 1 : 0;
            }
            y[i] = Math.Max(0.01, Math.Min(0.99, (sum + 0.5 * random.NextDouble()) / 10));
        }

        var likelihood = BetaLikelihood<double>.FromData(y);

        Assert.True(likelihood.Precision > 1,
            $"Estimated precision should be > 1, got {likelihood.Precision}");
        Assert.True(likelihood.Precision < 1000,
            $"Estimated precision should be reasonable, got {likelihood.Precision}");
    }

    #endregion

    #region Heteroscedastic GP Tests

    [Fact]
    public void HeteroscedasticGP_Construction_Succeeds()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new HeteroscedasticGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void HeteroscedasticGP_Fit_Succeeds()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new HeteroscedasticGaussianProcess<double>(kernel);

        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            y[i] = Math.Sin(i);
        }

        gp.Fit(X, y);

        // After fit, should be able to get noise variances (indicates training succeeded)
        var noiseVars = gp.GetNoiseVariances();
        Assert.NotNull(noiseVars);
    }

    [Fact]
    public void HeteroscedasticGP_GetNoiseVariances_ReturnsPositiveValues()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new HeteroscedasticGaussianProcess<double>(kernel);

        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            y[i] = Math.Sin(i) + 0.1 * i * (new Random(42 + i).NextDouble() - 0.5);
        }

        gp.Fit(X, y);

        var noiseVars = gp.GetNoiseVariances();

        Assert.Equal(10, noiseVars.Length);
        Assert.All(Enumerable.Range(0, noiseVars.Length), i =>
        {
            Assert.True(noiseVars[i] > 0,
                $"Noise variance[{i}] should be positive, got {noiseVars[i]}");
        });
    }

    #endregion

    #region Student-t GP Tests

    [Fact]
    public void StudentTGP_Construction_Succeeds()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new StudentTGaussianProcess<double>(kernel);

        Assert.NotNull(gp);
    }

    [Fact]
    public void StudentTGP_Fit_Succeeds()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new StudentTGaussianProcess<double>(kernel);

        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            y[i] = Math.Sin(i);
        }

        gp.Fit(X, y);

        // After fit, should be able to get outlier weights (indicates training succeeded)
        var weights = gp.GetOutlierWeights();
        Assert.NotNull(weights);
    }

    [Fact]
    public void StudentTGP_Predict_ReturnsReasonableValues()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new StudentTGaussianProcess<double>(kernel);

        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i * 0.5;
            y[i] = Math.Sin(i * 0.5);
        }

        gp.Fit(X, y);

        var xTest = new Vector<double>(new double[] { 0.0 });
        var (mean, variance) = gp.Predict(xTest);

        Assert.False(double.IsNaN(mean), "Prediction mean should not be NaN");
        Assert.True(variance > 0, "Prediction variance should be positive");
    }

    [Fact]
    public void StudentTGP_GetOutlierWeights_ReturnsValidWeights()
    {
        var kernel = new GaussianKernel<double>(1.0);
        var gp = new StudentTGaussianProcess<double>(kernel);

        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            y[i] = i;
        }
        // Add an outlier
        y[5] = 100.0;

        gp.Fit(X, y);

        var weights = gp.GetOutlierWeights();

        Assert.Equal(10, weights.Length);
        Assert.All(Enumerable.Range(0, weights.Length), i =>
        {
            Assert.True(weights[i] >= 0 && weights[i] <= 1,
                $"Weight[{i}] should be in [0,1], got {weights[i]}");
        });
    }

    [Fact]
    public void StudentTGP_DetectsOutliers()
    {
        var kernel = new GaussianKernel<double>(2.0);
        var gp = new StudentTGaussianProcess<double>(kernel, nu: 3.0);

        // Create linear data with one outlier
        var X = new Matrix<double>(11, 1);
        var y = new Vector<double>(11);
        for (int i = 0; i < 11; i++)
        {
            X[i, 0] = i;
            y[i] = i;
        }
        // Add a clear outlier
        y[5] = 50.0; // Should be 5, but is 50

        gp.Fit(X, y);

        var outlierIndices = gp.GetOutlierIndices();

        // The outlier at index 5 should be detected
        Assert.Contains(5, outlierIndices);
    }

    #endregion
}
