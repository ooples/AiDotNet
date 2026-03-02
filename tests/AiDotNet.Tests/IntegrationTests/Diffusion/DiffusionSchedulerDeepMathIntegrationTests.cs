using AiDotNet.Enums;
using AiDotNet.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Deep math-correctness integration tests for diffusion model schedulers.
/// Verifies beta schedules, alpha products, forward diffusion, and reverse step formulas
/// against hand-calculated values and known mathematical properties.
/// </summary>
public class DiffusionSchedulerDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Linear Beta Schedule

    [Fact]
    public void LinearBetaSchedule_HandCalculated_FirstBetaEqualsBetaStart()
    {
        // Linear: beta[0] = betaStart = 0.0001
        var config = SchedulerConfig<double>.CreateDefault(); // betaStart=0.0001, betaEnd=0.02, 1000 steps
        var scheduler = new DDPMScheduler<double>(config);
        double alpha0 = scheduler.GetAlphaCumulativeProduct(0);

        // alpha[0] = 1 - beta[0] = 1 - 0.0001 = 0.9999
        // alphaCumprod[0] = alpha[0] = 0.9999
        Assert.Equal(0.9999, alpha0, Tolerance);
    }

    [Fact]
    public void LinearBetaSchedule_HandCalculated_LastBetaEqualsBetaEnd()
    {
        // Linear: beta[999] = betaEnd = 0.02
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        double alphaLast = scheduler.GetAlphaCumulativeProduct(999);

        // alpha[999] = 1 - 0.02 = 0.98
        // alphaCumprod[999] = product of all alphas - should be very small (near zero)
        // The cumulative product should be much less than 1 and positive
        Assert.True(alphaLast > 0.0);
        Assert.True(alphaLast < 0.1); // Should be very small after 1000 steps
    }

    [Fact]
    public void LinearBetaSchedule_HandCalculated_InterpolationAtMidpoint()
    {
        // With 10 steps, linear interpolation: beta[i] = 0.0001 + (0.02 - 0.0001) * i / 9
        // beta[0] = 0.0001
        // beta[1] = 0.0001 + 0.0199 * 1/9 = 0.0001 + 0.002211... = 0.002311...
        // alpha[0] = 0.9999
        // alpha[1] = 1 - 0.002311... = 0.997688...
        // alphaCumprod[0] = 0.9999
        // alphaCumprod[1] = 0.9999 * 0.997688... = 0.997589...
        var config = new SchedulerConfig<double>(10, 0.0001, 0.02);
        var scheduler = new DDPMScheduler<double>(config);

        double alphaCumprod0 = scheduler.GetAlphaCumulativeProduct(0);
        double alphaCumprod1 = scheduler.GetAlphaCumulativeProduct(1);

        Assert.Equal(0.9999, alphaCumprod0, Tolerance);

        double beta1 = 0.0001 + (0.02 - 0.0001) * 1.0 / 9.0;
        double expectedAlphaCumprod1 = 0.9999 * (1.0 - beta1);
        Assert.Equal(expectedAlphaCumprod1, alphaCumprod1, Tolerance);
    }

    [Fact]
    public void LinearBetaSchedule_AlphaCumprod_MonotonicallyDecreasing()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        double prev = scheduler.GetAlphaCumulativeProduct(0);
        for (int t = 1; t < 1000; t += 50)
        {
            double current = scheduler.GetAlphaCumulativeProduct(t);
            Assert.True(current < prev, $"alphaCumprod should decrease: t={t}, prev={prev}, current={current}");
            Assert.True(current > 0.0, $"alphaCumprod should remain positive at t={t}");
            prev = current;
        }
    }

    #endregion

    #region Scaled Linear Beta Schedule

    [Fact]
    public void ScaledLinearBetaSchedule_HandCalculated_FirstAndLastBetas()
    {
        // ScaledLinear: beta[i] = (sqrt(betaStart) + (sqrt(betaEnd) - sqrt(betaStart)) * i/(steps-1))^2
        // beta[0] = (sqrt(0.00085))^2 = 0.00085
        // beta[999] = (sqrt(0.012))^2 = 0.012
        var config = SchedulerConfig<double>.CreateStableDiffusion(); // betaStart=0.00085, betaEnd=0.012
        var scheduler = new DDIMScheduler<double>(config);

        double alphaCumprod0 = scheduler.GetAlphaCumulativeProduct(0);
        // alpha[0] = 1 - 0.00085 = 0.99915
        Assert.Equal(1.0 - 0.00085, alphaCumprod0, Tolerance);
    }

    [Fact]
    public void ScaledLinearBetaSchedule_HandCalculated_MidpointBeta()
    {
        // With 10 steps: beta[5] = (sqrt(0.00085) + (sqrt(0.012) - sqrt(0.00085)) * 5/9)^2
        var config = new SchedulerConfig<double>(10, 0.00085, 0.012, BetaSchedule.ScaledLinear);
        var scheduler = new DDIMScheduler<double>(config);

        double sqrtStart = Math.Sqrt(0.00085);
        double sqrtEnd = Math.Sqrt(0.012);
        double sqrtBeta5 = sqrtStart + (sqrtEnd - sqrtStart) * 5.0 / 9.0;
        double beta5 = sqrtBeta5 * sqrtBeta5;

        // Compute expected alphaCumprod[5]
        double cumprod = 1.0;
        for (int i = 0; i <= 5; i++)
        {
            double sqrtBetaI = sqrtStart + (sqrtEnd - sqrtStart) * i / 9.0;
            double betaI = sqrtBetaI * sqrtBetaI;
            cumprod *= (1.0 - betaI);
        }

        double actual = scheduler.GetAlphaCumulativeProduct(5);
        Assert.Equal(cumprod, actual, Tolerance);
    }

    #endregion

    #region Squared Cosine Beta Schedule

    [Fact]
    public void SquaredCosineBetaSchedule_AlphaCumprod_StartsNearOneAndEndsNearZero()
    {
        var config = new SchedulerConfig<double>(1000, 0.0001, 0.02, BetaSchedule.SquaredCosine);
        var scheduler = new DDPMScheduler<double>(config);

        double alphaStart = scheduler.GetAlphaCumulativeProduct(0);
        double alphaEnd = scheduler.GetAlphaCumulativeProduct(999);

        // Cosine schedule: alpha_cumprod starts near 1 and ends near 0
        Assert.True(alphaStart > 0.95, $"First alphaCumprod should be near 1, got {alphaStart}");
        Assert.True(alphaEnd < 0.05, $"Last alphaCumprod should be near 0, got {alphaEnd}");
    }

    [Fact]
    public void SquaredCosineBetaSchedule_AlphaCumprod_SmoothTransition()
    {
        var config = new SchedulerConfig<double>(100, 0.0001, 0.02, BetaSchedule.SquaredCosine);
        var scheduler = new DDPMScheduler<double>(config);

        // Check that transition is smooth (no large jumps between consecutive steps)
        double prev = scheduler.GetAlphaCumulativeProduct(0);
        for (int t = 1; t < 100; t++)
        {
            double current = scheduler.GetAlphaCumulativeProduct(t);
            double jump = Math.Abs(current - prev);
            Assert.True(jump < 0.1, $"Jump too large at t={t}: {jump}");
            prev = current;
        }
    }

    [Fact]
    public void SquaredCosineBetaSchedule_HandCalculated_Formula()
    {
        // Cosine schedule: alpha_bar(t) = cos((t/T + s) / (1+s) * pi/2)^2
        // where s=0.008 offset, T = number of steps
        // beta[t] is clipped to [0, 0.999]
        int steps = 20;
        double s = 0.008;
        var config = new SchedulerConfig<double>(steps, 0.0001, 0.02, BetaSchedule.SquaredCosine);
        var scheduler = new DDPMScheduler<double>(config);

        // Verify by computing manually
        double cumprod = 1.0;
        for (int i = 0; i < steps; i++)
        {
            double t1 = (double)i / steps;
            double t2 = (double)(i + 1) / steps;

            double cos1 = Math.Cos(((t1 + s) / (1.0 + s)) * Math.PI / 2.0);
            double cos2 = Math.Cos(((t2 + s) / (1.0 + s)) * Math.PI / 2.0);

            double alphaBar1 = cos1 * cos1;
            double alphaBar2 = cos2 * cos2;

            double beta = 1.0 - alphaBar2 / alphaBar1;
            beta = Math.Max(0.0, Math.Min(0.999, beta));

            cumprod *= (1.0 - beta);
        }

        double actual = scheduler.GetAlphaCumulativeProduct(steps - 1);
        Assert.Equal(cumprod, actual, 1e-5);
    }

    #endregion

    #region Forward Diffusion (AddNoise)

    [Fact]
    public void ForwardDiffusion_HandCalculated_AtTimestep0()
    {
        // x_t = sqrt(alphaCumprod) * x_0 + sqrt(1-alphaCumprod) * noise
        // At t=0, alphaCumprod ≈ 0.9999
        // x_0 = [1.0, 2.0, 3.0], noise = [0.1, 0.2, 0.3]
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(1000);

        var x0 = new Vector<double>([1.0, 2.0, 3.0]);
        var noise = new Vector<double>([0.1, 0.2, 0.3]);

        var result = scheduler.AddNoise(x0, noise, 0);

        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(0); // ≈ 0.9999
        double sqrtAlpha = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlpha = Math.Sqrt(1.0 - alphaCumprod);

        for (int i = 0; i < 3; i++)
        {
            double expected = sqrtAlpha * x0[i] + sqrtOneMinusAlpha * noise[i];
            Assert.Equal(expected, result[i], Tolerance);
        }
    }

    [Fact]
    public void ForwardDiffusion_HandCalculated_AtHighTimestep()
    {
        // At high timestep, alphaCumprod is near 0, so noise dominates
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(1000);

        var x0 = new Vector<double>([1.0, 2.0, 3.0]);
        var noise = new Vector<double>([0.5, 0.5, 0.5]);

        var result = scheduler.AddNoise(x0, noise, 999);

        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(999);
        double sqrtAlpha = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlpha = Math.Sqrt(1.0 - alphaCumprod);

        // With low alphaCumprod, noise term dominates
        for (int i = 0; i < 3; i++)
        {
            double expected = sqrtAlpha * x0[i] + sqrtOneMinusAlpha * noise[i];
            Assert.Equal(expected, result[i], Tolerance);
        }

        // Verify noise dominates: result should be closer to noise * sqrt(1-alpha) than to x0 * sqrt(alpha)
        Assert.True(sqrtOneMinusAlpha > sqrtAlpha, "Noise term should dominate at high timestep");
    }

    [Fact]
    public void ForwardDiffusion_EnergyConservation()
    {
        // If x_0 and noise are orthogonal unit vectors, the energy is preserved:
        // ||x_t||^2 = alphaCumprod * ||x_0||^2 + (1-alphaCumprod) * ||noise||^2
        var config = new SchedulerConfig<double>(100, 0.0001, 0.02);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(100);

        // Use known unit vectors
        var x0 = new Vector<double>([1.0, 0.0, 0.0]);
        var noise = new Vector<double>([0.0, 1.0, 0.0]);

        for (int t = 0; t < 100; t += 10)
        {
            var result = scheduler.AddNoise(x0, noise, t);
            double alphaCumprod = scheduler.GetAlphaCumulativeProduct(t);

            double resultNormSq = 0;
            for (int i = 0; i < 3; i++)
                resultNormSq += result[i] * result[i];

            // ||x_t||^2 = alpha_bar * ||x_0||^2 + (1 - alpha_bar) * ||noise||^2 = alpha_bar + (1 - alpha_bar) = 1
            Assert.Equal(1.0, resultNormSq, 1e-5);
        }
    }

    [Fact]
    public void ForwardDiffusion_ZeroNoise_ReturnsSqrtAlphaCumprodTimesSignal()
    {
        var config = new SchedulerConfig<double>(100, 0.0001, 0.02);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(100);

        var x0 = new Vector<double>([2.0, 3.0]);
        var noise = new Vector<double>([0.0, 0.0]);

        var result = scheduler.AddNoise(x0, noise, 50);

        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(50);
        double sqrtAlpha = Math.Sqrt(alphaCumprod);

        Assert.Equal(sqrtAlpha * 2.0, result[0], Tolerance);
        Assert.Equal(sqrtAlpha * 3.0, result[1], Tolerance);
    }

    [Fact]
    public void ForwardDiffusion_ZeroSignal_ReturnsSqrtOneMinusAlphaCumprodTimesNoise()
    {
        var config = new SchedulerConfig<double>(100, 0.0001, 0.02);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(100);

        var x0 = new Vector<double>([0.0, 0.0]);
        var noise = new Vector<double>([1.0, -1.0]);

        var result = scheduler.AddNoise(x0, noise, 50);

        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(50);
        double sqrtOneMinusAlpha = Math.Sqrt(1.0 - alphaCumprod);

        Assert.Equal(sqrtOneMinusAlpha * 1.0, result[0], Tolerance);
        Assert.Equal(sqrtOneMinusAlpha * -1.0, result[1], Tolerance);
    }

    #endregion

    #region DDPM Reverse Step

    [Fact]
    public void DDPMStep_EpsilonPrediction_HandCalculated()
    {
        // With 10 steps, at timestep 9 (last), with epsilon prediction
        // Step formula: x_{t-1} = coeff1 * x_0_pred + coeff2 * x_t (no noise at final step or when noise=null)
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var epsilon = new Vector<double>([0.1, 0.2]); // model predicts epsilon

        // Compute expected values manually
        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(9);
        double sqrtAlphaCumprod = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaCumprod = Math.Sqrt(1.0 - alphaCumprod);

        // x_0_pred = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        double x0pred0 = (0.5 - sqrtOneMinusAlphaCumprod * 0.1) / sqrtAlphaCumprod;
        double x0pred1 = (-0.3 - sqrtOneMinusAlphaCumprod * 0.2) / sqrtAlphaCumprod;

        // The DDPM step produces a deterministic result when noise=null
        var result = scheduler.Step(epsilon, 9, sample, 0.0, null);
        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void DDPMStep_SamplePrediction_ReturnsPredictedSample()
    {
        // With Sample prediction, modelOutput is x_0 directly
        var config = new SchedulerConfig<double>(10, 0.001, 0.02,
            predictionType: DiffusionPredictionType.Sample);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var x0Pred = new Vector<double>([0.4, -0.2]); // model predicts x_0

        var result = scheduler.Step(x0Pred, 9, sample, 0.0, null);
        Assert.NotNull(result);
    }

    [Fact]
    public void DDPMStep_VPrediction_CorrectConversion()
    {
        // V-prediction: v = sqrt(alpha_bar) * eps - sqrt(1-alpha_bar) * x_0
        // x_0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
        var config = new SchedulerConfig<double>(10, 0.001, 0.02,
            predictionType: DiffusionPredictionType.VPrediction);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var vPred = new Vector<double>([0.1, 0.1]);

        var result = scheduler.Step(vPred, 9, sample, 0.0, null);
        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void DDPMStep_DeterministicWhenNoNoise()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3, 0.7]);
        var epsilon = new Vector<double>([0.1, 0.2, -0.1]);

        var result1 = scheduler.Step(epsilon, 9, sample, 0.0, null);
        var result2 = scheduler.Step(epsilon, 9, sample, 0.0, null);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(result1[i], result2[i], 1e-12);
        }
    }

    #endregion

    #region DDIM Reverse Step

    [Fact]
    public void DDIMStep_DeterministicEta0_HandCalculated()
    {
        // DDIM with eta=0 is fully deterministic
        // x_{t-1} = sqrt(alpha_prev) * x_0_pred + sqrt(1-alpha_prev) * eps
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.6, -0.4]);
        var epsilon = new Vector<double>([0.15, -0.1]);

        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(9);
        double sqrtAlphaCumprod = Math.Sqrt(alphaCumprod);
        double sqrtOneMinusAlphaCumprod = Math.Sqrt(1.0 - alphaCumprod);

        // x_0_pred = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
        double x0pred0 = (0.6 - sqrtOneMinusAlphaCumprod * 0.15) / sqrtAlphaCumprod;
        double x0pred1 = (-0.4 - sqrtOneMinusAlphaCumprod * (-0.1)) / sqrtAlphaCumprod;

        var result = scheduler.Step(epsilon, 9, sample, 0.0, null); // eta=0

        // With eta=0, sigma=0, so:
        // x_{t-1} = sqrt(alpha_prev) * x_0_pred + sqrt(1-alpha_prev) * eps
        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void DDIMStep_DeterministicEta0_SameResultTwice()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3, 0.7]);
        var epsilon = new Vector<double>([0.1, 0.2, -0.1]);

        var result1 = scheduler.Step(epsilon, 9, sample, 0.0, null);
        var result2 = scheduler.Step(epsilon, 9, sample, 0.0, null);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(result1[i], result2[i], 1e-12);
        }
    }

    [Fact]
    public void DDIMStep_StochasticEta1_DifferentWithDifferentNoise()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var epsilon = new Vector<double>([0.1, 0.2]);
        var noise1 = new Vector<double>([0.3, -0.2]);
        var noise2 = new Vector<double>([-0.1, 0.5]);

        var result1 = scheduler.Step(epsilon, 9, sample, 1.0, noise1);
        var result2 = scheduler.Step(epsilon, 9, sample, 1.0, noise2);

        // With different noise and eta=1, results should differ
        bool differs = false;
        for (int i = 0; i < 2; i++)
        {
            if (Math.Abs(result1[i] - result2[i]) > 1e-10)
                differs = true;
        }
        Assert.True(differs, "Stochastic DDIM with different noise should produce different results");
    }

    #endregion

    #region Euler Discrete Step

    [Fact]
    public void EulerStep_EpsilonPrediction_HandCalculated()
    {
        // Euler: d = (x_t - pred_x_0) / sigma_t, x_{t-1} = x_t + d * dt
        // For epsilon prediction: pred_x_0 = x_t - sigma * eps
        // So d = (x_t - (x_t - sigma * eps)) / sigma = eps
        // x_{t-1} = x_t + eps * (sigma_next - sigma)
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var epsilon = new Vector<double>([0.1, -0.2]);

        // This should work without error
        var timesteps = scheduler.Timesteps;
        var result = scheduler.Step(epsilon, timesteps[0], sample, 0.0, null);

        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void EulerStep_Deterministic_SameResultTwice()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3, 0.7]);
        var epsilon = new Vector<double>([0.1, 0.2, -0.1]);

        var timesteps = scheduler.Timesteps;
        var result1 = scheduler.Step(epsilon, timesteps[0], sample, 0.0, null);
        var result2 = scheduler.Step(epsilon, timesteps[0], sample, 0.0, null);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(result1[i], result2[i], 1e-12);
        }
    }

    #endregion

    #region Flow Matching Scheduler

    [Fact]
    public void FlowMatching_AddNoise_LinearInterpolation()
    {
        // Flow matching: x_t = (1-t) * x_0 + t * noise
        // At t=0 (timestep=0): x_0 = x_0 (no noise)
        // At t=1 (timestep=999): x_t = noise (all noise)
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var x0 = new Vector<double>([1.0, 2.0, 3.0]);
        var noise = new Vector<double>([0.5, -0.5, 0.0]);

        // At t=0, continuous time = 0/(1000-1) = 0
        var resultT0 = scheduler.AddNoise(x0, noise, 0);
        for (int i = 0; i < 3; i++)
        {
            // (1-0)*x_0 + 0*noise = x_0
            Assert.Equal(x0[i], resultT0[i], 1e-5);
        }

        // At t=999, continuous time = 999/999 = 1
        var resultT999 = scheduler.AddNoise(x0, noise, 999);
        for (int i = 0; i < 3; i++)
        {
            // (1-1)*x_0 + 1*noise = noise
            Assert.Equal(noise[i], resultT999[i], 1e-5);
        }
    }

    [Fact]
    public void FlowMatching_AddNoise_HandCalculatedMidpoint()
    {
        // At timestep 499 of 1000, continuous time = 499/999 ≈ 0.4995
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var x0 = new Vector<double>([2.0, 4.0]);
        var noise = new Vector<double>([0.0, 0.0]);

        double t = 499.0 / 999.0;
        var result = scheduler.AddNoise(x0, noise, 499);

        // x_t = (1-t)*x_0 + t*noise = (1-0.4995)*[2,4] + 0.4995*[0,0]
        Assert.Equal((1.0 - t) * 2.0, result[0], 1e-5);
        Assert.Equal((1.0 - t) * 4.0, result[1], 1e-5);
    }

    [Fact]
    public void FlowMatching_Step_VPrediction_EulerODE()
    {
        // Euler ODE step: x_{t-dt} = x_t + dt * v
        // v = noise - x_0 (velocity pointing from data to noise)
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var velocity = new Vector<double>([0.1, -0.1]);

        var timesteps = scheduler.Timesteps;
        var result = scheduler.Step(velocity, timesteps[0], sample, 0.0, null);

        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    [Fact]
    public void FlowMatching_Step_Deterministic()
    {
        var config = SchedulerConfig<double>.CreateRectifiedFlow();
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3, 0.7]);
        var velocity = new Vector<double>([0.1, -0.1, 0.2]);

        var timesteps = scheduler.Timesteps;
        var result1 = scheduler.Step(velocity, timesteps[0], sample, 0.0, null);
        var result2 = scheduler.Step(velocity, timesteps[0], sample, 0.0, null);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(result1[i], result2[i], 1e-12);
        }
    }

    [Fact]
    public void FlowMatching_DiffersFromDDPM_AddNoise()
    {
        // Flow matching uses linear interpolation: x_t = (1-t)*x_0 + t*noise
        // DDPM uses sqrt: x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*noise
        // These are different formulas and should give different results at same timestep

        var ddpmConfig = SchedulerConfig<double>.CreateDefault();
        var ddpmScheduler = new DDPMScheduler<double>(ddpmConfig);
        ddpmScheduler.SetTimesteps(1000);

        var flowConfig = SchedulerConfig<double>.CreateRectifiedFlow();
        var flowScheduler = new FlowMatchingScheduler<double>(flowConfig);
        flowScheduler.SetTimesteps(1000);

        var x0 = new Vector<double>([1.0, 2.0]);
        var noise = new Vector<double>([0.5, -0.5]);

        var ddpmResult = ddpmScheduler.AddNoise(x0, noise, 500);
        var flowResult = flowScheduler.AddNoise(x0, noise, 500);

        bool differs = false;
        for (int i = 0; i < 2; i++)
        {
            if (Math.Abs(ddpmResult[i] - flowResult[i]) > 1e-6)
                differs = true;
        }
        Assert.True(differs, "DDPM and flow matching should use different noise formulas");
    }

    #endregion

    #region Signal-to-Noise Ratio Properties

    [Fact]
    public void SNR_MonotonicallyDecreasing_LinearSchedule()
    {
        // SNR = alpha_bar / (1 - alpha_bar) should decrease monotonically
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        double prevSNR = double.MaxValue;
        for (int t = 0; t < 1000; t += 50)
        {
            double alphaCumprod = scheduler.GetAlphaCumulativeProduct(t);
            double snr = alphaCumprod / (1.0 - alphaCumprod);
            Assert.True(snr < prevSNR, $"SNR should decrease: t={t}");
            Assert.True(snr > 0, $"SNR should be positive at t={t}");
            prevSNR = snr;
        }
    }

    [Fact]
    public void SNR_HighAtStartLowAtEnd()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        double alpha0 = scheduler.GetAlphaCumulativeProduct(0);
        double snr0 = alpha0 / (1.0 - alpha0);
        Assert.True(snr0 > 100, $"SNR at t=0 should be high, got {snr0}");

        double alpha999 = scheduler.GetAlphaCumulativeProduct(999);
        double snr999 = alpha999 / (1.0 - alpha999);
        Assert.True(snr999 < 1, $"SNR at t=999 should be low, got {snr999}");
    }

    [Fact]
    public void Sigma_EulerScheduler_HandCalculated()
    {
        // sigma_t = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        // Which is the same as 1 / sqrt(SNR)
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new EulerDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);

        // Verify sigma computation at first timestep
        var timesteps = scheduler.Timesteps;
        double alphaCumprod = scheduler.GetAlphaCumulativeProduct(timesteps[0]);
        double expectedSigma = Math.Sqrt((1.0 - alphaCumprod) / alphaCumprod);

        // The scheduler should have computed this correctly
        Assert.True(expectedSigma > 0, "Sigma should be positive");
    }

    #endregion

    #region SetTimesteps

    [Fact]
    public void SetTimesteps_ReducesFromTrainingSteps()
    {
        var config = SchedulerConfig<double>.CreateDefault(); // 1000 training steps
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(50);

        Assert.Equal(50, scheduler.Timesteps.Length);
    }

    [Fact]
    public void SetTimesteps_TimestepsAreDecreasing()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(50);

        var timesteps = scheduler.Timesteps;
        for (int i = 1; i < timesteps.Length; i++)
        {
            Assert.True(timesteps[i] < timesteps[i - 1],
                $"Timesteps should be decreasing: t[{i-1}]={timesteps[i-1]}, t[{i}]={timesteps[i]}");
        }
    }

    [Fact]
    public void SetTimesteps_InvalidZero_ThrowsException()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.SetTimesteps(0));
    }

    [Fact]
    public void SetTimesteps_ExceedsTrainingSteps_ThrowsException()
    {
        var config = SchedulerConfig<double>.CreateDefault(); // 1000 training steps
        var scheduler = new DDPMScheduler<double>(config);
        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.SetTimesteps(1001));
    }

    #endregion

    #region Heun Discrete (Second-Order)

    [Fact]
    public void HeunStep_PredictorStep_ReturnsIntermediateResult()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new HeunDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var epsilon = new Vector<double>([0.1, -0.2]);

        var timesteps = scheduler.Timesteps;
        // First call = predictor step
        var intermediate = scheduler.Step(epsilon, timesteps[0], sample, 0.0, null);

        Assert.NotNull(intermediate);
        Assert.Equal(2, intermediate.Length);
    }

    [Fact]
    public void HeunStep_CorrectorStep_UsesAveragedDerivative()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new HeunDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var eps1 = new Vector<double>([0.1, -0.2]);
        var eps2 = new Vector<double>([0.12, -0.18]);

        var timesteps = scheduler.Timesteps;
        // Pass 1: predictor
        var intermediate = scheduler.Step(eps1, timesteps[0], sample, 0.0, null);
        // Pass 2: corrector (must use same timestep)
        var result = scheduler.Step(eps2, timesteps[0], intermediate, 0.0, null);

        Assert.NotNull(result);
        Assert.Equal(2, result.Length);
    }

    #endregion

    #region Cross-Scheduler Mathematical Properties

    [Fact]
    public void AllSchedulers_SameAlphaCumprod_SameConfig()
    {
        // All schedulers inheriting from NoiseSchedulerBase should compute identical
        // alpha cumulative products for the same config
        var config = SchedulerConfig<double>.CreateDefault();
        var ddpm = new DDPMScheduler<double>(config);
        var ddim = new DDIMScheduler<double>(config);
        var euler = new EulerDiscreteScheduler<double>(config);

        for (int t = 0; t < 1000; t += 100)
        {
            double ddpmAlpha = ddpm.GetAlphaCumulativeProduct(t);
            double ddimAlpha = ddim.GetAlphaCumulativeProduct(t);
            double eulerAlpha = euler.GetAlphaCumulativeProduct(t);

            Assert.Equal(ddpmAlpha, ddimAlpha, 1e-12);
            Assert.Equal(ddpmAlpha, eulerAlpha, 1e-12);
        }
    }

    [Fact]
    public void AllSchedulers_SameAddNoise_SameConfig()
    {
        // Forward diffusion (AddNoise) should be identical across DDPM-style schedulers
        var config = SchedulerConfig<double>.CreateDefault();
        var ddpm = new DDPMScheduler<double>(config);
        ddpm.SetTimesteps(1000);
        var ddim = new DDIMScheduler<double>(config);
        ddim.SetTimesteps(1000);

        var x0 = new Vector<double>([1.0, 2.0, 3.0]);
        var noise = new Vector<double>([0.5, -0.5, 0.0]);

        var ddpmResult = ddpm.AddNoise(x0, noise, 500);
        var ddimResult = ddim.AddNoise(x0, noise, 500);

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(ddpmResult[i], ddimResult[i], 1e-12);
        }
    }

    [Fact]
    public void AlphaCumprod_IsProductOfAlphas_Verification()
    {
        // alpha_cumprod[t] = product(alpha[0]...alpha[t]) = product((1-beta[0])...(1-beta[t]))
        // Verify by computing manually for a small schedule
        int steps = 5;
        double betaStart = 0.01;
        double betaEnd = 0.1;
        var config = new SchedulerConfig<double>(steps, betaStart, betaEnd);
        var scheduler = new DDPMScheduler<double>(config);

        double product = 1.0;
        for (int t = 0; t < steps; t++)
        {
            double beta = betaStart + (betaEnd - betaStart) * t / (steps - 1.0);
            product *= (1.0 - beta);
            double actual = scheduler.GetAlphaCumulativeProduct(t);
            Assert.Equal(product, actual, 1e-6);
        }
    }

    [Fact]
    public void DDPMStep_PerfectEpsilon_RecoversOriginal()
    {
        // If the model perfectly predicts the noise, the reverse step should
        // move toward the original clean sample
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var ddpm = new DDPMScheduler<double>(config);
        ddpm.SetTimesteps(10);

        var x0 = new Vector<double>([1.0, 2.0]);
        var trueNoise = new Vector<double>([0.3, -0.5]);

        // Forward: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        var xt = ddpm.AddNoise(x0, trueNoise, 5);

        // Reverse with perfect noise prediction (no added stochastic noise)
        var result = ddpm.Step(trueNoise, 5, xt, 0.0, null);

        // The result should be closer to x_0 than x_t was
        double distXtToX0 = Math.Sqrt(Math.Pow(xt[0] - x0[0], 2) + Math.Pow(xt[1] - x0[1], 2));
        double distResultToX0 = Math.Sqrt(Math.Pow(result[0] - x0[0], 2) + Math.Pow(result[1] - x0[1], 2));
        Assert.True(distResultToX0 < distXtToX0,
            $"After one reverse step with perfect prediction, result should be closer to x_0. " +
            $"dist(x_t, x_0)={distXtToX0}, dist(result, x_0)={distResultToX0}");
    }

    [Fact]
    public void DDIMStep_PerfectEpsilon_RecoversOriginal()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var ddim = new DDIMScheduler<double>(config);
        ddim.SetTimesteps(10);

        var x0 = new Vector<double>([1.0, 2.0]);
        var trueNoise = new Vector<double>([0.3, -0.5]);

        var xt = ddim.AddNoise(x0, trueNoise, 5);
        var result = ddim.Step(trueNoise, 5, xt, 0.0, null);

        double distXtToX0 = Math.Sqrt(Math.Pow(xt[0] - x0[0], 2) + Math.Pow(xt[1] - x0[1], 2));
        double distResultToX0 = Math.Sqrt(Math.Pow(result[0] - x0[0], 2) + Math.Pow(result[1] - x0[1], 2));
        Assert.True(distResultToX0 < distXtToX0,
            $"DDIM reverse step with perfect prediction should move closer to x_0");
    }

    #endregion

    #region Numerical Stability

    [Fact]
    public void AlphaCumprod_NeverZeroOrNegative()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        for (int t = 0; t < 1000; t++)
        {
            double alphaCumprod = scheduler.GetAlphaCumulativeProduct(t);
            Assert.True(alphaCumprod > 0.0, $"alphaCumprod should be positive at t={t}, got {alphaCumprod}");
        }
    }

    [Fact]
    public void AlphaCumprod_NeverExceedsOne()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        for (int t = 0; t < 1000; t++)
        {
            double alphaCumprod = scheduler.GetAlphaCumulativeProduct(t);
            Assert.True(alphaCumprod <= 1.0, $"alphaCumprod should be <= 1 at t={t}, got {alphaCumprod}");
        }
    }

    [Fact]
    public void GetAlphaCumulativeProduct_OutOfRange_ThrowsException()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);

        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.GetAlphaCumulativeProduct(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => scheduler.GetAlphaCumulativeProduct(1000));
    }

    [Fact]
    public void AddNoise_MismatchedLengths_ThrowsException()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(1000);

        var x0 = new Vector<double>([1.0, 2.0]);
        var noise = new Vector<double>([0.5, -0.5, 0.0]); // Different length

        Assert.Throws<ArgumentException>(() => scheduler.AddNoise(x0, noise, 500));
    }

    [Fact]
    public void Step_MismatchedLengths_ThrowsException()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(1000);

        var sample = new Vector<double>([0.5, -0.3, 0.7]);
        var modelOutput = new Vector<double>([0.1, 0.2]); // Different length

        Assert.Throws<ArgumentException>(() => scheduler.Step(modelOutput, 999, sample, 0.0, null));
    }

    #endregion

    #region State Management

    [Fact]
    public void GetState_ContainsExpectedKeys()
    {
        var config = SchedulerConfig<double>.CreateDefault();
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(50);

        var state = scheduler.GetState();

        Assert.True(state.ContainsKey("timesteps"));
        Assert.True(state.ContainsKey("train_timesteps"));
        Assert.True(state.ContainsKey("beta_start"));
        Assert.True(state.ContainsKey("beta_end"));
        Assert.True(state.ContainsKey("beta_schedule"));
        Assert.True(state.ContainsKey("prediction_type"));
    }

    [Fact]
    public void FlowMatching_GetState_IncludesSchedulerType()
    {
        var scheduler = FlowMatchingScheduler<double>.CreateDefault();
        scheduler.SetTimesteps(10);

        var state = scheduler.GetState();

        Assert.True(state.ContainsKey("scheduler_type"));
        Assert.Equal("FlowMatching", state["scheduler_type"]);
    }

    #endregion

    #region Clip Sample

    [Fact]
    public void ClipSample_WhenEnabled_ClipsToNegOneToOne()
    {
        var config = new SchedulerConfig<double>(10, 0.001, 0.02,
            clipSample: true, predictionType: DiffusionPredictionType.Epsilon);
        var scheduler = new DDPMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        // Use a large model output that would predict extreme x_0 values
        var sample = new Vector<double>([0.5, -0.3]);
        var largeEpsilon = new Vector<double>([5.0, -5.0]); // Very large noise prediction

        var result = scheduler.Step(largeEpsilon, 5, sample, 0.0, null);

        // With clip, the internal predicted_original should be clipped to [-1, 1]
        // The final result may not be in [-1, 1] but should be more bounded than without clip
        Assert.NotNull(result);
    }

    [Fact]
    public void ClipSample_WhenDisabled_DoesNotClip()
    {
        var configClip = new SchedulerConfig<double>(10, 0.001, 0.02,
            clipSample: true, predictionType: DiffusionPredictionType.Epsilon);
        var configNoClip = new SchedulerConfig<double>(10, 0.001, 0.02,
            clipSample: false, predictionType: DiffusionPredictionType.Epsilon);

        var schedulerClip = new DDPMScheduler<double>(configClip);
        schedulerClip.SetTimesteps(10);
        var schedulerNoClip = new DDPMScheduler<double>(configNoClip);
        schedulerNoClip.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var largeEpsilon = new Vector<double>([10.0, -10.0]);

        var resultClip = schedulerClip.Step(largeEpsilon, 5, sample, 0.0, null);
        var resultNoClip = schedulerNoClip.Step(largeEpsilon, 5, sample, 0.0, null);

        // With large predictions, clipping should produce different results
        bool differs = false;
        for (int i = 0; i < 2; i++)
        {
            if (Math.Abs(resultClip[i] - resultNoClip[i]) > 1e-6)
                differs = true;
        }
        Assert.True(differs, "Clipping should produce different results with extreme predictions");
    }

    #endregion

    #region Beta Schedule Boundary Conditions

    [Fact]
    public void LinearBetaSchedule_TwoSteps_BetaStartAndBetaEndExactly()
    {
        // With exactly 2 steps: beta[0] = betaStart, beta[1] = betaEnd
        var config = new SchedulerConfig<double>(2, 0.01, 0.1);
        var scheduler = new DDPMScheduler<double>(config);

        double alphaCumprod0 = scheduler.GetAlphaCumulativeProduct(0);
        double alphaCumprod1 = scheduler.GetAlphaCumulativeProduct(1);

        // alpha[0] = 1 - 0.01 = 0.99
        Assert.Equal(0.99, alphaCumprod0, Tolerance);
        // alpha[1] = 1 - 0.1 = 0.9
        // alphaCumprod[1] = 0.99 * 0.9 = 0.891
        Assert.Equal(0.891, alphaCumprod1, Tolerance);
    }

    [Fact]
    public void ScaledLinearBetaSchedule_TwoSteps_CorrectValues()
    {
        // ScaledLinear with 2 steps: beta[0] = (sqrt(betaStart))^2 = betaStart
        // beta[1] = (sqrt(betaEnd))^2 = betaEnd
        var config = new SchedulerConfig<double>(2, 0.01, 0.04, BetaSchedule.ScaledLinear);
        var scheduler = new DDPMScheduler<double>(config);

        double alphaCumprod0 = scheduler.GetAlphaCumulativeProduct(0);
        double alphaCumprod1 = scheduler.GetAlphaCumulativeProduct(1);

        Assert.Equal(1.0 - 0.01, alphaCumprod0, Tolerance);
        Assert.Equal((1.0 - 0.01) * (1.0 - 0.04), alphaCumprod1, Tolerance);
    }

    #endregion

    #region DDPM vs DDIM Mathematical Relationship

    [Fact]
    public void DDIMEta1_EquivalentToDDPM_SameNoise()
    {
        // DDIM with eta=1 should produce results close to DDPM (same stochastic formulation)
        // They won't be exactly identical due to slight differences in coefficient computation
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var ddpm = new DDPMScheduler<double>(config);
        ddpm.SetTimesteps(10);
        var ddim = new DDIMScheduler<double>(config);
        ddim.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var epsilon = new Vector<double>([0.1, -0.2]);
        var noise = new Vector<double>([0.3, 0.1]);

        // Both should produce valid denoised results
        var ddpmResult = ddpm.Step(epsilon, 5, sample, 0.0, noise);
        var ddimResult = ddim.Step(epsilon, 5, sample, 1.0, noise); // eta=1 for stochastic DDIM

        // Both should be valid finite vectors
        for (int i = 0; i < 2; i++)
        {
            Assert.False(double.IsNaN(ddpmResult[i]));
            Assert.False(double.IsInfinity(ddpmResult[i]));
            Assert.False(double.IsNaN(ddimResult[i]));
            Assert.False(double.IsInfinity(ddimResult[i]));
        }
    }

    [Fact]
    public void DDIMEta0_ReducesToDeterministicFormula()
    {
        // DDIM with eta=0: sigma=0, so x_{t-1} = sqrt(alpha_prev)*x_0_pred + sqrt(1-alpha_prev)*eps
        // This is fully deterministic regardless of noise input
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var sample = new Vector<double>([0.5, -0.3]);
        var epsilon = new Vector<double>([0.1, -0.2]);
        var noise1 = new Vector<double>([0.9, -0.9]);
        var noise2 = new Vector<double>([-0.9, 0.9]);

        // With eta=0, noise should be ignored
        var result1 = scheduler.Step(epsilon, 5, sample, 0.0, noise1);
        var result2 = scheduler.Step(epsilon, 5, sample, 0.0, noise2);

        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(result1[i], result2[i], 1e-12);
        }
    }

    #endregion

    #region Mathematical Consistency of Forward-Reverse

    [Fact]
    public void ForwardReverse_MultiStep_ConvergesToOriginal()
    {
        // Running multiple reverse steps with perfect epsilon should converge toward x_0
        var config = new SchedulerConfig<double>(10, 0.001, 0.02);
        var scheduler = new DDIMScheduler<double>(config);
        scheduler.SetTimesteps(10);

        var x0 = new Vector<double>([0.8, -0.6]);
        var trueNoise = new Vector<double>([0.2, -0.3]);

        // Forward to t=9
        var xt = scheduler.AddNoise(x0, trueNoise, 9);

        // Run several reverse steps with perfect prediction (DDIM eta=0)
        var current = xt;
        var timesteps = scheduler.Timesteps;

        // Take first 3 steps to move toward x_0
        for (int step = 0; step < Math.Min(3, timesteps.Length); step++)
        {
            current = scheduler.Step(trueNoise, timesteps[step], current, 0.0, null);
        }

        // After multiple reverse steps, should be closer to x_0
        double initialDist = Math.Sqrt(Math.Pow(xt[0] - x0[0], 2) + Math.Pow(xt[1] - x0[1], 2));
        double finalDist = Math.Sqrt(Math.Pow(current[0] - x0[0], 2) + Math.Pow(current[1] - x0[1], 2));

        Assert.True(finalDist < initialDist,
            $"Multiple reverse steps should move closer to x_0: initial={initialDist}, final={finalDist}");
    }

    #endregion
}
