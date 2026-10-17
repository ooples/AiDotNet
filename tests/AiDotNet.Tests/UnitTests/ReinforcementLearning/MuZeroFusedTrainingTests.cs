using System;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.MuZero;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// MuZero's joint K-step unroll trains its three networks on the shared fused step (#1804): windows cut at an episode
/// end are padded and masked, so one compiled plan serves every batch of a size. With compilation on, training must
/// leave the networks where the eager steps leave them.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class MuZeroFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step on another batch's data moves weights by far more.
    private const double RelativeTolerance = 1e-5;

    private static Vector<double> Train(bool compile)
    {
        const int obs = 4, actionDim = 2, batch = 4;
        var agent = new MuZeroAgent<double>(new MuZeroOptions<double>
        {
            ObservationSize = obs,
            ActionSize = actionDim,
            LatentStateSize = 6,
            BatchSize = batch,
            UnrollSteps = 3,
            TDSteps = 2,
            Seed = 11,
        });
        var rng = RandomHelper.CreateSeededRandom(5);
        for (int t = 0; t < 24; t++)
        {
            var state = new Vector<double>(obs);
            var next = new Vector<double>(obs);
            for (int i = 0; i < obs; i++)
            {
                state[i] = rng.NextDouble() * 2.0 - 1.0;
                next[i] = rng.NextDouble() * 2.0 - 1.0;
            }
            var action = new Vector<double>(actionDim);
            action[rng.Next(actionDim)] = 1.0;
            // Episode ends every seventh transition, so sampled windows come in different lengths.
            agent.StoreExperience(state, action, rng.NextDouble(), next, done: t % 7 == 6);
        }

        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            for (int step = 0; step < 5; step++) agent.Train();
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
        return agent.GetParameters();
    }

    private static double Worst(Vector<double> a, Vector<double> b, out double scale)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        scale = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(a[i]));
            worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        }
        return worst;
    }

    [Fact(Timeout = 120000)]
    public async Task TwoSeededRuns_Agree()
    {
        await Task.Yield();
        double worst = Worst(Train(compile: false), Train(compile: false), out double scale);
        Assert.True(worst == 0.0, $"two seeded eager runs differ by {worst:G6} (parameter scale {scale:G6}).");
    }

    [Fact(Timeout = 120000)]
    public async Task FusedTraining_LandsWhereTheEagerStepsDo()
    {
        await Task.Yield();
        double worst = Worst(Train(compile: false), Train(compile: true), out double scale);
        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"fused MuZero training diverged from eager by {worst:G6} (parameter scale {scale:G6}).");
    }
}
