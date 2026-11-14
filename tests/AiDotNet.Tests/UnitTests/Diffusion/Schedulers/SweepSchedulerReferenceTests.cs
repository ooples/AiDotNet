using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.Schedulers.NoiseSchedules;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.Video;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// Reference tests for the scheduler, noise-schedule and video helpers rewritten in the diffusion element-wise
/// sweep. Expected values come from closed forms or from independent double-precision re-implementations of the
/// published algorithms (k-diffusion / diffusers LMS and Heun, DPM-Solver++, UniPC bh2, SA-Solver, Lin et al.
/// zero terminal SNR, Nichol-Dhariwal cosine with the simple-diffusion shift).
/// </summary>
public class SweepSchedulerReferenceTests
{
    private const double Tolerance = 1e-9;

    private static Vector<double> Vec(params double[] values) => new Vector<double>(values);

    private static void AssertClose(double[] expected, Vector<double> actual, double tolerance = Tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"Element {i}: expected {expected[i]}, got {actual[i]}.");
    }

    private static double[] Combine(double[] a, double wa, double[] b, double wb)
    {
        var result = new double[a.Length];
        for (int i = 0; i < a.Length; i++)
            result[i] = (wa * a[i]) + (wb * b[i]);
        return result;
    }

    #region SchedulerTensorOps

    [Fact]
    public void LagrangeBasis_MatchesClosedForm()
    {
        // Nodes 0, 1, 2 at x = 0.5: L0 = (x-1)(x-2)/2, L1 = -x(x-2), L2 = x(x-1)/2.
        var basis = SchedulerTensorOps.LagrangeBasis(Vec(0, 1, 2), 0.5);
        AssertClose(new[] { 0.375, 0.75, -0.125 }, basis);
    }

    [Fact]
    public void IntegrateLagrangeBasis_MatchesExactIntegrals()
    {
        // Integrals over [0, 1] of the quadratic Lagrange basis of nodes 0, 1, 2.
        var integrals = SchedulerTensorOps.IntegrateLagrangeBasis(Vec(0, 1, 2), 0.0, 1.0);
        AssertClose(new[] { 5.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0 }, integrals);
    }

    [Fact]
    public void IntegrateLagrangeBasis_CubicBasisMatchesSimpson()
    {
        var nodes = new[] { 0.9, 1.7, 2.6, 4.0 };
        var integrals = SchedulerTensorOps.IntegrateLagrangeBasis(Vec(nodes), 0.5, 0.2);
        var expected = new double[4];
        for (int j = 0; j < 4; j++)
            expected[j] = Simpson(u => Lagrange(nodes, j, u), 0.5, 0.2);
        AssertClose(expected, integrals, 1e-10);
    }

    [Fact]
    public void ExponentialAdamsCoefficients_MatchQuadrature()
    {
        // b_i = c * integral over u in [-h, 0] of e^{c u} l_i(u) du.
        var offsets = new[] { 0.0, -0.4, -0.9 };
        double h = 0.6;
        double c = 1.3;
        var weights = SchedulerTensorOps.ExponentialAdamsCoefficients(Vec(offsets), h, c);
        var expected = new double[3];
        for (int i = 0; i < 3; i++)
            expected[i] = c * Simpson(u => Math.Exp(c * u) * Lagrange(offsets, i, u), -h, 0.0);
        AssertClose(expected, weights, 1e-9);
    }

    [Fact]
    public void ExponentialAdamsCoefficients_SingleNodeIsDpmSolverFirstOrder()
    {
        var weights = SchedulerTensorOps.ExponentialAdamsCoefficients(Vec(0.0), 0.7, 1.0);
        AssertClose(new[] { 1.0 - Math.Exp(-0.7) }, weights);
    }

    [Fact]
    public void SolveTransposedVandermonde_SatisfiesMomentEquations()
    {
        var nodes = new[] { 0.5, -1.0, 2.0 };
        var rhs = new[] { 1.0, 0.25, -3.0 };
        var w = SchedulerTensorOps.SolveTransposedVandermonde(Vec(nodes), Vec(rhs));
        for (int k = 0; k < 3; k++)
        {
            double sum = 0;
            for (int i = 0; i < 3; i++)
                sum += w[i] * Math.Pow(nodes[i], k);
            Assert.Equal(rhs[k], sum, 10);
        }
    }

    [Fact]
    public void KarrasSigmas_MatchDiffusersFormula()
    {
        var alphas = Vec(0.9, 0.5, 0.1);
        var sigmas = SchedulerTensorOps.KarrasSigmas(alphas, new[] { 2, 0 });
        AssertClose(new[] { Math.Sqrt(0.9 / 0.1), Math.Sqrt(0.1 / 0.9), 0.0 }, sigmas);
    }

    #endregion

    #region Flow schedulers

    public enum FlowSolver
    {
        RectifiedFlow,
        ExFM,
        PeRFlow,
        FlowDpmSolver2,
        FlowDpmSolver3,
        HybridOdeSde,
        StochasticInterpolant,
        VariationalRF
    }

    public static IEnumerable<object[]> FlowSchedulers()
        => Enum.GetValues(typeof(FlowSolver)).Cast<FlowSolver>().Select(solver => new object[] { solver });

    private static NoiseSchedulerBase<double> CreateFlowScheduler(FlowSolver solver)
    {
        var config = SchedulerConfig<double>.CreateDefault();
        return solver switch
        {
            FlowSolver.RectifiedFlow => new RectifiedFlowScheduler<double>(config),
            FlowSolver.ExFM => new ExFMScheduler<double>(config),
            FlowSolver.PeRFlow => new PeRFlowScheduler<double>(config),
            FlowSolver.FlowDpmSolver2 => new FlowDPMSolverScheduler<double>(config, 2),
            FlowSolver.FlowDpmSolver3 => new FlowDPMSolverScheduler<double>(config, 3),
            FlowSolver.HybridOdeSde => new HybridODESDEScheduler<double>(config),
            FlowSolver.StochasticInterpolant => new StochasticInterpolantScheduler<double>(config),
            _ => new VariationalRFScheduler<double>(config)
        };
    }

    [Theory]
    [MemberData(nameof(FlowSchedulers))]
    public void FlowScheduler_ConstantVelocity_IntegratesExactlyToTimeZero(FlowSolver solver)
    {
        // On the straight path x_t = x_0 + t * v the exact velocity is constant, so every consistent solver must
        // land on x_0 = x_T - t_0 * v whatever its step sizes are (an over- or under-shooting step size would not).
        var scheduler = CreateFlowScheduler(solver);
        scheduler.SetTimesteps(12);
        var velocity = Vec(0.4, -1.2, 2.0);
        var start = Vec(1.5, 0.3, -0.7);
        double t0 = SchedulerTensorOps.FlowTime(scheduler.Timesteps[0], scheduler.TrainTimesteps);

        var sample = start;
        foreach (int timestep in scheduler.Timesteps)
            sample = scheduler.Step(velocity, timestep, sample, 0.0);

        AssertClose(Combine(start.ToArray(), 1.0, velocity.ToArray(), -t0), sample, 1e-9);
    }

    [Fact]
    public void ExFM_TimestepsDecreaseWithGapsShrinkingTowardTimeZero()
    {
        var scheduler = new ExFMScheduler<double>(SchedulerConfig<double>.CreateDefault());
        scheduler.SetTimesteps(10);
        var timesteps = scheduler.Timesteps;
        for (int i = 1; i < timesteps.Length; i++)
            Assert.True(timesteps[i] < timesteps[i - 1], "Timesteps must be strictly decreasing.");
        int firstGap = timesteps[0] - timesteps[1];
        int lastGap = timesteps[timesteps.Length - 2] - timesteps[timesteps.Length - 1];
        Assert.True(lastGap < firstGap, "The quadratic schedule must concentrate steps near t = 0.");
    }

    [Fact]
    public void FlowDPMSolver_SecondOrder_MatchesReference()
    {
        var scheduler = new FlowDPMSolverScheduler<double>(SchedulerConfig<double>.CreateDefault(), 2);
        scheduler.SetTimesteps(6);
        var timesteps = scheduler.Timesteps;
        int train = scheduler.TrainTimesteps;

        var x = new[] { 0.8, -0.5 };
        var reference = (double[])x.Clone();
        var sample = Vec(x);
        var history = new List<(double Lambda, double[] Data)>();
        for (int i = 0; i < timesteps.Length; i++)
        {
            // A state-dependent velocity so the multistep corrections matter.
            var v = new[] { (0.3 * reference[0]) + 0.1, (-0.2 * reference[1]) + 0.05 };
            sample = scheduler.Step(Vec((0.3 * sample[0]) + 0.1, (-0.2 * sample[1]) + 0.05), timesteps[i], sample, 0.0);

            double s = SchedulerTensorOps.FlowTime(timesteps[i], train);
            double next = i + 1 < timesteps.Length ? SchedulerTensorOps.FlowTime(timesteps[i + 1], train) : 0.0;
            double lambdaS = FlowLambda(s);
            var data = new[] { reference[0] - (s * v[0]), reference[1] - (s * v[1]) };
            history.Add((lambdaS, data));

            // diffusers: lower_order_final on the last step, lower_order_second before it for short schedules,
            // first order for the warm-up step.
            bool firstOrder = i == 0 || i == timesteps.Length - 1;
            double h = FlowLambda(next) - lambdaS;
            double phi = Math.Exp(-h) - 1.0;
            double alphaT = 1.0 - next;
            var updated = new double[2];
            for (int k = 0; k < 2; k++)
            {
                double value = ((s > 0 ? next / s : 0.0) * reference[k]) - (alphaT * phi * data[k]);
                if (!firstOrder)
                {
                    var previous = history[history.Count - 2];
                    double r0 = (lambdaS - previous.Lambda) / h;
                    value -= 0.5 * alphaT * phi * (data[k] - previous.Data[k]) / r0;
                }

                updated[k] = value;
            }

            reference = updated;
            AssertClose(reference, sample, 1e-9);
        }
    }

    [Fact]
    public void HybridODESDE_SdeStep_AddsScoreCorrection()
    {
        var scheduler = new HybridODESDEScheduler<double>(SchedulerConfig<double>.CreateDefault(), 0.5);
        scheduler.SetTimesteps(10);
        int index = scheduler.Timesteps.Length - 3;
        int timestep = scheduler.Timesteps[index];
        double t = SchedulerTensorOps.FlowTime(timestep, scheduler.TrainTimesteps);
        double dt = t - SchedulerTensorOps.FlowTime(scheduler.Timesteps[index + 1], scheduler.TrainTimesteps);
        Assert.True(t < 0.5);

        double eta = 0.7;
        var x = new[] { 0.3, -1.1 };
        var v = new[] { 1.4, 0.2 };
        var z = new[] { -0.6, 0.9 };
        var result = scheduler.Step(Vec(v), timestep, Vec(x), eta, Vec(z));

        // x - v dt - (eta^2 dt / 2t) * eps_hat + eta sqrt(dt) z, eps_hat = x + (1 - t) v.
        var expected = new double[2];
        for (int k = 0; k < 2; k++)
        {
            double epsHat = x[k] + ((1.0 - t) * v[k]);
            expected[k] = x[k] - (v[k] * dt) - (eta * eta * dt / (2.0 * t) * epsHat) + (eta * Math.Sqrt(dt) * z[k]);
        }

        AssertClose(expected, result);
    }

    [Fact]
    public void StochasticInterpolant_Step_MatchesMarginalPreservingSde()
    {
        double lambda = 0.5;
        var scheduler = new StochasticInterpolantScheduler<double>(SchedulerConfig<double>.CreateDefault(), lambda);
        scheduler.SetTimesteps(8);
        int timestep = scheduler.Timesteps[2];
        double t = SchedulerTensorOps.FlowTime(timestep, scheduler.TrainTimesteps);
        double dt = t - SchedulerTensorOps.FlowTime(scheduler.Timesteps[3], scheduler.TrainTimesteps);

        var x = new[] { 0.9 };
        var v = new[] { -0.4 };
        var z = new[] { 1.3 };
        var result = scheduler.Step(Vec(v), timestep, Vec(x), 0.0, Vec(z));

        // g^2 = lambda^2 t: drift correction -(g^2 dt / 2t) eps_hat = -(lambda^2 dt / 2) eps_hat, noise lambda sqrt(dt t).
        double epsHat = x[0] + ((1.0 - t) * v[0]);
        double expected = x[0] - (v[0] * dt) - (lambda * lambda * dt / 2.0 * epsHat) + (lambda * Math.Sqrt(dt * t) * z[0]);
        AssertClose(new[] { expected }, result);
    }

    [Fact]
    public void VariationalRF_Step_MatchesMarginalPreservingSde()
    {
        double noiseScale = 0.2;
        var scheduler = new VariationalRFScheduler<double>(SchedulerConfig<double>.CreateDefault(), noiseScale);
        scheduler.SetTimesteps(8);
        int timestep = scheduler.Timesteps[3];
        double t = SchedulerTensorOps.FlowTime(timestep, scheduler.TrainTimesteps);
        double dt = t - SchedulerTensorOps.FlowTime(scheduler.Timesteps[4], scheduler.TrainTimesteps);

        double eta = 0.8;
        var x = new[] { -0.2 };
        var v = new[] { 0.6 };
        var z = new[] { 0.4 };
        var result = scheduler.Step(Vec(v), timestep, Vec(x), eta, Vec(z));

        double g = eta * noiseScale * t;
        double epsHat = x[0] + ((1.0 - t) * v[0]);
        double expected = x[0] - (v[0] * dt) - (g * g * dt / (2.0 * t) * epsHat) + (g * Math.Sqrt(dt) * z[0]);
        AssertClose(new[] { expected }, result);
    }

    [Fact]
    public void FlowMatching_EpsilonPrediction_RecoversCleanSample()
    {
        // With the exact noise as model output the step must land on the forward process at the next time,
        // which needs x_0 = (x_t - t * eps) / (1 - t).
        var config = new SchedulerConfig<double>(1000, 0.0001, 0.02, predictionType: DiffusionPredictionType.Epsilon);
        var scheduler = new FlowMatchingScheduler<double>(config);
        scheduler.SetTimesteps(10);
        var clean = Vec(0.7, -0.3, 1.1);
        var noise = Vec(-0.5, 0.8, 0.2);
        int timestep = scheduler.Timesteps[3];
        var noisy = scheduler.AddNoise(clean, noise, timestep);

        var result = scheduler.Step(noise, timestep, noisy, 0.0);

        var expected = scheduler.AddNoise(clean, noise, scheduler.Timesteps[4]);
        AssertClose(expected.ToArray(), result, 1e-9);
    }

    private static double FlowLambda(double t) => Math.Log(1.0 - t) - Math.Log(t);

    #endregion

    #region Sigma-space and VP solvers

    private static double[] KSigmas(NoiseSchedulerBase<double> scheduler)
    {
        var timesteps = scheduler.Timesteps;
        var sigmas = new double[timesteps.Length + 1];
        for (int i = 0; i < timesteps.Length; i++)
        {
            double a = scheduler.GetAlphaCumulativeProduct(timesteps[i]);
            sigmas[i] = Math.Sqrt((1.0 - a) / a);
        }

        return sigmas;
    }

    [Fact]
    public void LMS_SecondStep_UsesIntegratedLagrangeCoefficients()
    {
        var scheduler = new LMSDiscreteScheduler<double>(SchedulerConfig<double>.CreateStableDiffusion(), 4);
        scheduler.SetTimesteps(10);
        var sigmas = KSigmas(scheduler);
        var timesteps = scheduler.Timesteps;

        var x = new[] { 3.0, -2.0 };
        var eps0 = new[] { 0.5, -0.25 };
        var eps1 = new[] { 0.1, 0.7 };

        var x1 = scheduler.Step(Vec(eps0), timesteps[0], Vec(x), 0.0);
        // Epsilon prediction: d = eps. First step: x + (sigma_1 - sigma_0) d_0.
        AssertClose(Combine(x, 1.0, eps0, sigmas[1] - sigmas[0]), x1);

        var x2 = scheduler.Step(Vec(eps1), timesteps[1], x1, 0.0);
        // k-diffusion get_lms_coefficient over [sigma_1, sigma_2] with nodes sigma_1 (newest) and sigma_0.
        double s0 = sigmas[0], s1 = sigmas[1], s2 = sigmas[2];
        double newest = (((s2 - s0) * (s2 - s0)) - ((s1 - s0) * (s1 - s0))) / (2.0 * (s1 - s0));
        double older = ((s2 - s1) * (s2 - s1)) / (2.0 * (s0 - s1));
        var expected = new double[2];
        for (int k = 0; k < 2; k++)
            expected[k] = x1[k] + (newest * eps1[k]) + (older * eps0[k]);
        AssertClose(expected, x2, 1e-9);
    }

    [Fact]
    public void LMS_VPrediction_UsesSigmaSpaceConversion()
    {
        var config = new SchedulerConfig<double>(1000, 0.00085, 0.012, BetaSchedule.ScaledLinear,
            predictionType: DiffusionPredictionType.VPrediction);
        var scheduler = new LMSDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);
        var sigmas = KSigmas(scheduler);

        var x = new[] { 2.0, -1.0 };
        var v = new[] { 0.3, 0.6 };
        var result = scheduler.Step(Vec(v), scheduler.Timesteps[0], Vec(x), 0.0);
        AssertClose(SigmaSpaceVEuler(x, v, sigmas[0], sigmas[1]), result, 1e-9);
    }

    [Fact]
    public void Heun_VPrediction_PredictorUsesSigmaSpaceConversion()
    {
        var config = new SchedulerConfig<double>(1000, 0.00085, 0.012, BetaSchedule.ScaledLinear,
            predictionType: DiffusionPredictionType.VPrediction);
        var scheduler = new HeunDiscreteScheduler<double>(config);
        scheduler.SetTimesteps(10);
        var sigmas = KSigmas(scheduler);

        var x = new[] { 2.0, -1.0 };
        var v = new[] { 0.3, 0.6 };
        var result = scheduler.Step(Vec(v), scheduler.Timesteps[0], Vec(x), 0.0);
        AssertClose(SigmaSpaceVEuler(x, v, sigmas[0], sigmas[1]), result, 1e-9);
    }

    private static double[] SigmaSpaceVEuler(double[] x, double[] v, double sigma, double sigmaNext)
    {
        // diffusers: x_0 = v * (-sigma / sqrt(sigma^2 + 1)) + x / (sigma^2 + 1); d = (x - x_0) / sigma.
        var result = new double[x.Length];
        for (int k = 0; k < x.Length; k++)
        {
            double x0 = (x[k] / ((sigma * sigma) + 1.0)) - (v[k] * sigma / Math.Sqrt((sigma * sigma) + 1.0));
            result[k] = x[k] + ((x[k] - x0) / sigma * (sigmaNext - sigma));
        }

        return result;
    }

    private static (double Alpha, double Sigma, double Lambda) Vp(NoiseSchedulerBase<double> scheduler, int timestep)
    {
        double a = scheduler.GetAlphaCumulativeProduct(timestep);
        double alpha = Math.Sqrt(a);
        double sigma = Math.Sqrt(1.0 - a);
        return (alpha, sigma, Math.Log(alpha) - Math.Log(sigma));
    }

    private static double[] DataPrediction(double[] x, double[] eps, double alpha, double sigma)
    {
        var result = new double[x.Length];
        for (int k = 0; k < x.Length; k++)
            result[k] = (x[k] - (sigma * eps[k])) / alpha;
        return result;
    }

    [Fact]
    public void UniPC_FirstTwoSteps_MatchDiffusersBh2()
    {
        var scheduler = new UniPCScheduler<double>(SchedulerConfig<double>.CreateDefault(), 2, true);
        scheduler.SetTimesteps(6);
        var ts = scheduler.Timesteps;
        var p0 = Vp(scheduler, ts[0]);
        var p1 = Vp(scheduler, ts[1]);
        var p2 = Vp(scheduler, ts[2]);

        var x0 = new[] { 1.2, -0.4 };
        var eps0 = new[] { 0.3, -0.9 };
        var eps1 = new[] { -0.2, 0.5 };

        // Step 0: UniP order 1 = DPM-Solver++ first order.
        var m0 = DataPrediction(x0, eps0, p0.Alpha, p0.Sigma);
        double phi0 = Math.Exp(-(p1.Lambda - p0.Lambda)) - 1.0;
        var x1 = Combine(x0, p1.Sigma / p0.Sigma, m0, -p1.Alpha * phi0);
        var actual1 = scheduler.Step(Vec(eps0), ts[0], Vec(x0), 0.0);
        AssertClose(x1, actual1, 1e-8);

        // Step 1: UniC order 1 corrects step 0 with m1 (rho = 0.5), then UniP order 2 (rho = 0.5).
        var m1 = DataPrediction(x1, eps1, p1.Alpha, p1.Sigma);
        var corrected = new double[2];
        for (int k = 0; k < 2; k++)
            corrected[k] = x1[k] - (p1.Alpha * phi0 * 0.5 * (m1[k] - m0[k]));

        double h = p2.Lambda - p1.Lambda;
        double phi1 = Math.Exp(-h) - 1.0;
        double r = (p0.Lambda - p1.Lambda) / h;
        var x2 = new double[2];
        for (int k = 0; k < 2; k++)
        {
            double d1 = (m0[k] - m1[k]) / r;
            x2[k] = (p2.Sigma / p1.Sigma * corrected[k]) - (p2.Alpha * phi1 * m1[k]) - (p2.Alpha * phi1 * 0.5 * d1);
        }

        var actual2 = scheduler.Step(Vec(eps1), ts[1], actual1, 0.0);
        AssertClose(x2, actual2, 1e-8);
    }

    [Fact]
    public void SASolver_DeterministicFirstStep_IsDpmSolverFirstOrder()
    {
        var scheduler = new SASolverScheduler<double>(SchedulerConfig<double>.CreateDefault(), 3);
        scheduler.SetTimesteps(8);
        var ts = scheduler.Timesteps;
        var p0 = Vp(scheduler, ts[0]);
        var p1 = Vp(scheduler, ts[1]);

        var x = new[] { -0.6, 0.9, 0.1 };
        var eps = new[] { 0.4, 0.2, -1.0 };
        var m0 = DataPrediction(x, eps, p0.Alpha, p0.Sigma);
        double phi = Math.Exp(-(p1.Lambda - p0.Lambda)) - 1.0;

        var result = scheduler.Step(Vec(eps), ts[0], Vec(x), 0.0);
        AssertClose(Combine(x, p1.Sigma / p0.Sigma, m0, -p1.Alpha * phi), result, 1e-9);
    }

    [Fact]
    public void SASolver_FinalStep_ReturnsDataPrediction()
    {
        var scheduler = new SASolverScheduler<double>(SchedulerConfig<double>.CreateDefault(), 2);
        scheduler.SetTimesteps(3);
        var ts = scheduler.Timesteps;
        var sample = Vec(0.5, -0.5);
        var eps = Vec(0.1, 0.3);
        for (int i = 0; i < ts.Length - 1; i++)
            sample = scheduler.Step(eps, ts[i], sample, 0.0);

        var last = Vp(scheduler, ts[ts.Length - 1]);
        var expected = DataPrediction(sample.ToArray(), eps.ToArray(), last.Alpha, last.Sigma);
        AssertClose(expected, scheduler.Step(eps, ts[ts.Length - 1], sample, 0.0), 1e-9);
    }

    #endregion

    #region Noise schedules

    [Fact]
    public void ZeroTerminalSnr_RescalesSquareRootAlphas()
    {
        var alphas = new[] { 0.99, 0.8, 0.5, 0.2, 0.05 };
        var result = new ZeroTerminalSNRSchedule<double>().Apply(Vec(alphas));

        // Lin et al. (2024) Algorithm 1 on sqrt(alpha_cumprod).
        double first = Math.Sqrt(alphas[0]);
        double last = Math.Sqrt(alphas[alphas.Length - 1]);
        var expected = new double[alphas.Length];
        for (int i = 0; i < alphas.Length; i++)
        {
            double rescaled = (Math.Sqrt(alphas[i]) - last) * first / (first - last);
            expected[i] = rescaled * rescaled;
        }

        AssertClose(expected, result, 1e-12);
        Assert.Equal(0.0, result[alphas.Length - 1], 12);
        Assert.Equal(alphas[0], result[0], 12);
    }

    [Fact]
    public void CosineShifted_MatchesNormalizedCosineWithSnrShift()
    {
        double shift = 2.0;
        int steps = 20;
        var result = new CosineShiftedSchedule<double>(shift).ComputeAlphasCumprod(steps);

        double s = 0.008;
        double F(double t) => Math.Pow(Math.Cos((t + s) / (1.0 + s) * Math.PI / 2.0), 2);
        var expected = new double[steps];
        for (int i = 0; i < steps; i++)
        {
            double a = F((double)i / steps) / F(0.0);
            expected[i] = a / (a + (shift * shift * (1.0 - a)));
        }

        AssertClose(expected, result, 1e-12);
        Assert.Equal(1.0, result[0], 12);
    }

    [Fact]
    public void OffsetNoise_AddsSeededGaussianOffsetPerChannel()
    {
        double strength = 0.3;
        var noise = Vec(1, 2, 3, 4, 5, 6, 7);
        var result = new OffsetNoiseSchedule<double>(strength).ApplyOffset(noise, 3, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11));

        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var offsets = new double[3];
        for (int c = 0; c < 3; c++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            offsets[c] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * strength;
        }

        // Two pixels per channel; the trailing element keeps its noise.
        var expected = new[] { 1 + offsets[0], 2 + offsets[0], 3 + offsets[1], 4 + offsets[1], 5 + offsets[2], 6 + offsets[2], 7.0 };
        AssertClose(expected, result, 1e-12);
    }

    [Fact]
    public void LogSnrImportanceSampling_PicksNearestLogSnrTimestep()
    {
        var alphas = new[] { 0.95, 0.7, 0.4, 0.1, 0.01 };
        var picked = new LogSNRImportanceSampling<double>().SampleTimesteps(6, Vec(alphas), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5));

        var logSnr = new double[alphas.Length];
        for (int i = 0; i < alphas.Length; i++)
            logSnr[i] = Math.Log(alphas[i] / (1.0 - alphas[i]));
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);
        for (int b = 0; b < 6; b++)
        {
            double target = logSnr[alphas.Length - 1] + (random.NextDouble() * (logSnr[0] - logSnr[alphas.Length - 1]));
            int best = 0;
            for (int i = 1; i < alphas.Length; i++)
                if (Math.Abs(logSnr[i] - target) < Math.Abs(logSnr[best] - target))
                    best = i;
            Assert.Equal(best, picked[b]);
        }
    }

    #endregion

    #region Image and video helpers

    [Fact]
    public void DallE3_BilinearUpscale_MatchesHalfPixelBilinearForEveryImage()
    {
        // Two 1x2x2 images; torch interpolate(mode="bilinear", align_corners=False) to 4x4.
        var data = new double[] { 1, 2, 3, 4, 10, 20, 30, 40 };
        var image = new Tensor<double>(new[] { 2, 1, 2, 2 }, Vec(data));
        var result = DallE3Model<double>.BilinearUpscale(image, 4, 4);

        Assert.Equal(new[] { 2, 1, 4, 4 }, result.Shape.ToArray());
        for (int n = 0; n < 2; n++)
        {
            for (int y = 0; y < 4; y++)
            {
                for (int x = 0; x < 4; x++)
                {
                    var (y0, y1, wy) = Source(y, 2, 4);
                    var (x0, x1, wx) = Source(x, 2, 4);
                    double At(int r, int c) => data[(n * 4) + (r * 2) + c];
                    double top = At(y0, x0) + (wx * (At(y0, x1) - At(y0, x0)));
                    double bottom = At(y1, x0) + (wx * (At(y1, x1) - At(y1, x0)));
                    double expected = top + (wy * (bottom - top));
                    Assert.Equal(expected, result[n, 0, y, x], 10);
                }
            }
        }
    }

    private static (int Low, int High, double Weight) Source(int dst, int inSize, int outSize)
    {
        double src = Math.Max(0.0, ((dst + 0.5) * inSize / outSize) - 0.5);
        int low = (int)Math.Floor(src);
        int high = Math.Min(low + 1, inSize - 1);
        return (low, high, src - low);
    }

    [Fact]
    public void AnimateDiff_BlendWindowNoise_WeightsWindowAndHandlesSingleFrame()
    {
        int frames = 4;
        var shape = new[] { 1, frames, 1, 1, 1 };
        var accumulated = new Tensor<double>(shape);
        var weights = new Tensor<double>(new[] { 1, frames, 1, 1, 1 });

        // A three-frame window at frame 0 (weights 0.75, 1, 0.75) and a one-frame window at frame 3 (weight 1).
        var window = new Tensor<double>(new[] { 1, 3, 1, 1, 1 }, Vec(2, 4, 6));
        (accumulated, weights) = AnimateDiffModel<double>.BlendWindowNoise(accumulated, weights, window, 0, 3);
        var single = new Tensor<double>(new[] { 1, 1, 1, 1, 1 }, Vec(5));
        (accumulated, weights) = AnimateDiffModel<double>.BlendWindowNoise(accumulated, weights, single, 3, 1);

        AssertClose(new[] { 1.5, 4.0, 4.5, 5.0 }, accumulated.ToVector());
        AssertClose(new[] { 0.75, 1.0, 0.75, 1.0 }, weights.ToVector());
        AssertClose(new[] { 2.0, 4.0, 6.0, 5.0 }, AnimateDiffModel<double>.NormalizeAccumulatedNoise(accumulated, weights).ToVector());
    }

    [Fact]
    public void StableVideoDiffusion_EndImageGuidance_GuidesEveryBatchItem()
    {
        // Batch 2, 3 frames, one latent value per frame.
        var latents = new Tensor<double>(new[] { 2, 3, 1, 1, 1 }, Vec(1, 2, 3, 4, 5, 6));
        var start = new Tensor<double>(new[] { 2, 1, 1, 1 }, Vec(10, 20));
        var end = new Tensor<double>(new[] { 2, 1, 1, 1 }, Vec(30, 60));
        var result = StableVideoDiffusion<double>.ApplyEndImageGuidance(latents, start, end, 500);

        double strength = 0.1 * 0.5;
        var expected = new double[6];
        for (int b = 0; b < 2; b++)
        {
            for (int f = 0; f < 3; f++)
            {
                double p = f / 2.0;
                double target = ((1 - p) * start[b, 0, 0, 0]) + (p * end[b, 0, 0, 0]);
                double w = strength * p;
                expected[(b * 3) + f] = ((1 - w) * latents[b, f, 0, 0, 0]) + (w * target);
            }
        }

        AssertClose(expected, result.ToVector());
    }

    [Fact]
    public void StableVideoDiffusion_EndImageGuidance_SingleFrameStaysFinite()
    {
        var latents = new Tensor<double>(new[] { 1, 1, 1, 1, 1 }, Vec(2.5));
        var start = new Tensor<double>(new[] { 1, 1, 1, 1 }, Vec(1.0));
        var end = new Tensor<double>(new[] { 1, 1, 1, 1 }, Vec(3.0));
        var result = StableVideoDiffusion<double>.ApplyEndImageGuidance(latents, start, end, 900);

        // Progress is 0 for the only frame, so the guidance weight is 0 and the latent is unchanged.
        AssertClose(new[] { 2.5 }, result.ToVector());
    }

    #endregion

    #region Review fixes

    private static readonly AiDotNet.Tensors.Interfaces.INumericOperations<Half> HalfOps =
        AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<Half>();

    private static Vector<Half> HalfVec(params double[] values)
    {
        var data = new Half[values.Length];
        for (int i = 0; i < values.Length; i++)
            data[i] = HalfOps.FromDouble(values[i]);
        return new Vector<Half>(data);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SigmaSpaceVPrediction_Half_SigmaAbove255_KeepsBothTerms(bool heun)
    {
        // sigma^2 + 1 = 90,001 overflows Half, which zeroed both x / (sigma^2 + 1) and v * sigma / sqrt(sigma^2 + 1).
        var config = new SchedulerConfig<Half>(1000, HalfOps.FromDouble(0.00085), HalfOps.FromDouble(0.012),
            BetaSchedule.ScaledLinear, predictionType: DiffusionPredictionType.VPrediction);
        var x = new[] { 2.0, -1.0 };
        var v = new[] { 0.3, 0.6 };
        Half sigma = HalfOps.FromDouble(300.0);

        var x0 = heun
            ? new HeunDiscreteScheduler<Half>(config).ConvertToPredOriginal(HalfVec(v), HalfVec(x), sigma, 0)
            : new LMSDiscreteScheduler<Half>(config).ConvertToPredOriginal(HalfVec(v), HalfVec(x), sigma, 0);

        for (int k = 0; k < x.Length; k++)
        {
            double expected = (x[k] / 90001.0) - (v[k] * 300.0 / Math.Sqrt(90001.0));
            Assert.Equal(expected, HalfOps.ToDouble(x0[k]), tolerance: 1e-3);
        }
    }

    [Fact]
    public void LogSnrImportanceSampling_ZeroTerminalSnr_SamplesTheSameForEveryElementType()
    {
        // alpha = 0 has log-SNR -inf; the sampled range must end at the same finite bound (-ln(1e10), the value of
        // the original 1e-10 guard) for double and float, not at a bound tied to each type's epsilon.
        var alphas = new[] { 0.99, 0.5, 0.0 };
        const int draws = 2000;
        double LogSnr(double a)
        {
            a = Math.Max(a, 1e-10);
            return Math.Log(a / (1.0 - a + 1e-10));
        }

        var logSnr = alphas.Select(LogSnr).ToArray();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var expected = new int[draws];
        for (int b = 0; b < draws; b++)
        {
            double target = logSnr[2] + (random.NextDouble() * (logSnr[0] - logSnr[2]));
            int best = 0;
            for (int i = 1; i < alphas.Length; i++)
                if (Math.Abs(logSnr[i] - target) < Math.Abs(logSnr[best] - target))
                    best = i;
            expected[b] = best;
        }

        var asDouble = new LogSNRImportanceSampling<double>().SampleTimesteps(
            draws, Vec(alphas), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11));
        var asFloat = new LogSNRImportanceSampling<float>().SampleTimesteps(
            draws, new Vector<float>(alphas.Select(a => (float)a).ToArray()), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11));

        Assert.Equal(expected, asDouble);
        Assert.Equal(expected, asFloat);
    }

    [Fact]
    public void StableVideoDiffusion_AddTimeProjection_MatchesDiffusersTimesteps()
    {
        // diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0): [cos(v f_i), sin(v f_i)],
        // f_i = exp(-ln(10000) i / 128).
        const double value = 127.0;
        var projected = StableVideoDiffusion<double>.ProjectConditioningValue(value, 256).ToArray();
        var projectedHalf = StableVideoDiffusion<Half>.ProjectConditioningValue(value, 256).ToArray();

        Assert.Equal(256, projected.Length);
        for (int i = 0; i < 128; i++)
        {
            double phase = value * Math.Exp(-Math.Log(10000.0) * i / 128.0);
            Assert.Equal(Math.Cos(phase), projected[i], tolerance: 1e-12);
            Assert.Equal(Math.Sin(phase), projected[128 + i], tolerance: 1e-12);
            // Half holds the rounded exact feature: the phase itself is never formed in Half.
            Assert.Equal(Math.Cos(phase), HalfOps.ToDouble(projectedHalf[i]), tolerance: 1e-3);
            Assert.Equal(Math.Sin(phase), HalfOps.ToDouble(projectedHalf[128 + i]), tolerance: 1e-3);
        }
    }

    [Fact]
    public void VideoCrafter_MotionModulation_Half_LongEmbeddingStaysExact()
    {
        // Element i gains 0.1 * sin(0.001 * i * motion). As a Half index, i stops being exact at 2048 and becomes
        // infinite at 65,520, which made the modulation NaN.
        const int size = 70000;
        var image = new Tensor<Half>(new[] { size });
        var motion = new Tensor<Half>(new[] { 1 }, HalfVec(1.0));

        var result = AiDotNet.Diffusion.Video.VideoCrafterModel<Half>.CombineImageAndMotion(image, motion).ToArray();

        foreach (int i in new[] { 0, 1000, 2049, 4097, 40001, 65519, 65520, 69999 })
            Assert.Equal(0.1 * Math.Sin(0.001 * i), HalfOps.ToDouble(result[i]), tolerance: 1e-3);
    }

    #endregion

    private static double Lagrange(double[] nodes, int j, double x)
    {
        double value = 1.0;
        for (int k = 0; k < nodes.Length; k++)
        {
            if (k != j)
                value *= (x - nodes[k]) / (nodes[j] - nodes[k]);
        }

        return value;
    }

    private static double Simpson(Func<double, double> f, double from, double to)
    {
        const int intervals = 2000;
        double step = (to - from) / intervals;
        double sum = f(from) + f(to);
        for (int i = 1; i < intervals; i++)
            sum += f(from + (i * step)) * (i % 2 == 1 ? 4 : 2);
        return sum * step / 3.0;
    }
}
