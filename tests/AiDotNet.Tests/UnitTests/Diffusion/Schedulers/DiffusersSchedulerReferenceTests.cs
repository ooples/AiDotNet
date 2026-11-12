using System;
using System.Threading.Tasks;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// The Euler, Euler ancestral and DPM-Solver++ multistep schedulers reproduce diffusers 0.35
/// (<c>tools/reference-data/diffusers_diffusion_reference.py</c>): ten trailing-spaced steps over Stable Diffusion's
/// scaled-linear betas with the toy model output m(x, t) = 0.5 x + c (1 + t / 1000) end at the reference samples.
/// </summary>
/// <remarks>
/// Before this, DPM-Solver++ built its log-SNR from the Karras sigma instead of the VP sigma and added
/// alpha_t (exp(-h) - 1) D0 where the solver subtracts it, and the Euler schedulers converted a v-prediction with
/// the VP coefficients although their sample lives in sigma space. The consistency model re-noised a multistep
/// sample with sigma_next instead of Algorithm 1's sqrt(sigma_next^2 - sigma_min^2).
/// </remarks>
public class DiffusersSchedulerReferenceTests
{
    private const double Tolerance = 1e-10;
    private static readonly double[] InitialSample = { 1.0, -0.5, 0.25, 2.0 };
    private static readonly double[] Offset = { 0.1, -0.2, 0.3, -0.4 };

    private static SchedulerConfig<double> StableDiffusionConfig(DiffusionPredictionType predictionType) =>
        new(1000, 0.00085, 0.012, BetaSchedule.ScaledLinear, clipSample: false, predictionType: predictionType);

    private static double[] Run(INoiseScheduler<double> scheduler, bool ancestralNoise)
    {
        scheduler.SetTimesteps(10);
        var sample = new Vector<double>(InitialSample);
        int step = 0;
        foreach (int t in scheduler.Timesteps)
        {
            var output = new Vector<double>(sample.Length);
            var noise = new Vector<double>(sample.Length);
            for (int i = 0; i < sample.Length; i++)
            {
                output[i] = 0.5 * sample[i] + Offset[i] * (1.0 + t / 1000.0);
                noise[i] = Math.Sin(1.3 * (step + 1) * (i + 1));
            }

            sample = scheduler.Step(output, t, sample, 1.0, ancestralNoise ? noise : null);
            step++;
        }

        return sample.ToArray();
    }

    private static void AssertClose(double[] expected, double[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= Tolerance, $"[{i}] expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Fact(Timeout = 120000)]
    public async Task Timesteps_AreDiffusersTrailingSpacing()
    {
        await Task.Yield();
        var scheduler = new EulerDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon));
        scheduler.SetTimesteps(10);
        Assert.Equal(new[] { 999, 899, 799, 699, 599, 499, 399, 299, 199, 99 }, scheduler.Timesteps);
    }

    [Fact(Timeout = 120000)]
    public async Task Euler_Epsilon_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon));
        AssertClose(new[] { -0.262875569327016, 0.555948232611201, -0.843988046902525, 1.17229065313674 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task Euler_VPrediction_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.VPrediction));
        AssertClose(new[] { -0.113811287473735, 0.236431801580701, -0.357584111248795, 0.490482056427865 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task EulerAncestral_Epsilon_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon));
        AssertClose(new[] { -0.390360182153464, 0.533637051239677, -1.00917052576202, 0.927967158970461 },
            Run(scheduler, ancestralNoise: true));
    }

    [Fact(Timeout = 120000)]
    public async Task EulerAncestral_VPrediction_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new EulerAncestralDiscreteScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.VPrediction));
        AssertClose(new[] { -0.284217462288841, 0.154337565838926, -0.444837736354689, 0.301483692977084 },
            Run(scheduler, ancestralNoise: true));
    }

    [Fact(Timeout = 120000)]
    public async Task DPMSolverMultistep_SecondOrder_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new DPMSolverMultistepScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon), solverOrder: 2);
        AssertClose(new[] { 1.91301665868667, 0.264969760654408, -1.76112233365753, 8.71194567736432 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task DPMSolverMultistep_ThirdOrder_MatchesDiffusers()
    {
        await Task.Yield();
        var scheduler = new DPMSolverMultistepScheduler<double>(StableDiffusionConfig(DiffusionPredictionType.Epsilon), solverOrder: 3);
        AssertClose(new[] { 1.93788526846478, 0.280834257993723, -1.80678631863168, 8.87487810583401 },
            Run(scheduler, ancestralNoise: false));
    }

    [Fact(Timeout = 120000)]
    public async Task ConsistencyRenoise_MatchesDiffusers()
    {
        await Task.Yield();
        // diffusers CMStochasticIterativeScheduler.step from the first to the second of two Karras sigmas
        const double sigmaNext = 2.2404396533966064;
        var denoised = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { 20.0522322038777, -10.1011127621746, 5.15055192806836, 39.8044777668122 }));
        var noise = new Tensor<double>(new[] { 4 }, new Vector<double>(new[] { Math.Sin(1.3), Math.Sin(2.6), Math.Sin(3.9), Math.Sin(5.2) }));

        var renoised = ConsistencyModel<double>.RenoiseToSigma(denoised, noise, sigmaNext, 0.002);

        AssertClose(new[] { 22.2110253106901, -8.94616350754432, 3.60965396672485, 37.8251517128031 }, renoised.ToArray());
    }
}
