using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The portfolio policy agents' episode updates run on the shared fused session (#1804). With compilation on every
/// update must run fused and leave the policy where the eager updates leave it.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public sealed class PolicyAgentFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 per step; an update on another episode's data moves the policy by far more.
    private const double Tolerance = 1e-6;

    public static TheoryData<string> Agents => new() { "Feedforward", "Recurrent" };

    private sealed record RunResult(double[] GreedyActions, int FusedSteps, int EagerSteps);

    private static IPortfolioAgent<double> Build(string kind) => kind switch
    {
        "Feedforward" => new FeedforwardPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, learningRate: 1e-2, seed: 4),
        "Recurrent" => new RecurrentPolicyAgent<double>(stateDim: 3, actionDim: 2, hidden: 8, learningRate: 1e-2, seed: 4),
        _ => throw new ArgumentOutOfRangeException(nameof(kind)),
    };

    // Episodes of changing content and length: a plan that baked one episode's states, actions or weights, or lost
    // the optimizer moments when the length changed, lands somewhere other than the eager updates.
    private static void TrainEpisodes(IPortfolioAgent<double> agent)
    {
        int[] lengths = { 5, 5, 7, 5 };
        for (int episode = 0; episode < lengths.Length; episode++)
        {
            agent.ResetEpisode();
            for (int t = 0; t < lengths[episode]; t++)
            {
                var state = new Vector<double>(new[] { Math.Sin(t + episode), Math.Cos(0.7 * t - episode), 0.1 * t });
                var action = new Vector<double>(new[] { Math.Tanh(0.3 * t - episode), 0.8 * Math.Sin(0.5 * t + episode) });
                agent.StoreExperience(state, action, Math.Cos(0.9 * t + episode), state, done: t == lengths[episode] - 1);
            }
            agent.Train();
        }
    }

    private static double[] GreedyActions(IPortfolioAgent<double> agent)
    {
        agent.ResetEpisode();
        var outputs = new List<double>();
        for (int t = 0; t < 4; t++)
        {
            var action = agent.SelectAction(new Vector<double>(new[] { 0.2 * t, -0.3, 0.5 - 0.1 * t }), explore: false);
            for (int i = 0; i < action.Length; i++) outputs.Add(action[i]);
        }
        return outputs.ToArray();
    }

    private static RunResult Run(string kind, bool compile)
    {
        var original = TensorCodecOptions.Current;
        var agent = Build(kind);
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            TrainEpisodes(agent);
            var (fused, eager) = agent switch
            {
                FeedforwardPolicyAgent<double> f => (f.FusedTrainingStepCount, f.EagerTrainingStepCount),
                RecurrentPolicyAgent<double> r => (r.FusedTrainingStepCount, r.EagerTrainingStepCount),
                _ => (0, 0),
            };
            return new RunResult(GreedyActions(agent), fused, eager);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
            (agent as IDisposable)?.Dispose();
        }
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Agents))]
    public async Task EpisodeUpdates_RunFused_AndTakeTheEagerStep(string kind)
    {
        await Task.Yield();
        var untrained = Build(kind);
        var before = GreedyActions(untrained);
        (untrained as IDisposable)?.Dispose();

        var eager = Run(kind, compile: false);
        var fused = Run(kind, compile: true);

        Assert.Equal(0, eager.FusedSteps);
        Assert.Equal(4, eager.EagerSteps);
        Assert.Equal(0, fused.EagerSteps);
        Assert.Equal(4, fused.FusedSteps);

        double moved = 0.0, worst = 0.0;
        for (int i = 0; i < eager.GreedyActions.Length; i++)
        {
            moved = Math.Max(moved, Math.Abs(eager.GreedyActions[i] - before[i]));
            worst = Math.Max(worst, Math.Abs(eager.GreedyActions[i] - fused.GreedyActions[i]));
        }

        Assert.True(moved > 100 * Tolerance, $"{kind}: training barely moved the policy ({moved:G6}), so the comparison proves nothing.");
        Assert.True(worst <= Tolerance, $"{kind}: fused training diverged from eager by {worst:G6}.");
    }
}
