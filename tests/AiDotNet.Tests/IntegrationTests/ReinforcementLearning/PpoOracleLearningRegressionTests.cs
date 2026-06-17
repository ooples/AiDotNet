using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// Regression repro: PPO's per-episode training reward must RISE on a trivially learnable oracle
/// environment (a feature column equal to the next-bar return). Downstream evidence: this rises reliably
/// on the 0.208.117 package (ec214ec) and DECLINES on master-built 0.208.119 — suspected regression in the
/// 117→master window (prime suspect: the NeuralNetworkBase lazy-init/Layers-ordering changes in #1555,
/// which the PPO actor/critic tape training flows through). This test is the bisection instrument.
/// </summary>
public class PpoOracleLearningRegressionTests
{
    [Fact]
    public void Ppo_training_reward_rises_on_a_perfect_oracle()
    {
        const int n = 600;
        const int windowSize = 4;
        // Single-asset env (shape [time, 1]); the deterministic 8-bar zig-zag is fully learnable from the
        // 4-bar price window alone (the window identifies the cycle phase exactly).
        var data = new Tensor<double>(new[] { n, 1 });
        double p = 100.0;
        for (var i = 0; i < n; i++)
        {
            double nextRet = ((i / 4) % 2 == 0) ? 0.01 : -0.01; // 8-bar zig-zag cycle
            data[i, 0] = p;
            p *= 1.0 + nextRet;
        }

        var env = new StockTradingEnvironment<double>(
            data, windowSize, initialCapital: 100_000.0, tradeSize: 100.0,
            transactionCost: 0.0, allowShortSelling: true, randomStart: false);

        var options = new FinancialPPOAgentOptions<double> { ContinuousActions = false };
        var agent = new FinancialPPOAgent<double>(
            Arch(options.StateSize, options.ActionSize),
            Arch(options.StateSize, 1),
            options);

        const int episodes = 60;
        const int stepsPerEpisode = 200;
        var episodeRewards = new double[episodes];
        for (var e = 0; e < episodes; e++)
        {
            var state = env.Reset();
            double total = 0;
            for (var s = 0; s < stepsPerEpisode; s++)
            {
                var action = agent.SelectAction(state, training: true);
                var (next, reward, done, _) = env.Step(action);
                total += reward;
                agent.StoreExperience(state, action, reward, next, done);
                if (s % 4 == 3 || done)
                {
                    agent.Train();
                }

                state = next;
                if (done)
                {
                    break;
                }
            }

            episodeRewards[e] = total;
        }

        var early = episodeRewards.Take(20).Average();
        var late = episodeRewards.Skip(40).Average();
        Assert.True(late > early,
            $"PPO is not learning a PERFECT oracle: early-episodes avg {early:F3} vs late {late:F3} " +
            $"(full curve: {string.Join(",", episodeRewards.Select(r => r.ToString("F2")))})");
    }

    [Fact]
    public void Ppo_training_reward_rises_on_a_perfect_oracle_float()
    {
        // FLOAT variant — the downstream platform trains agents in float; the float kernel path (incl.
        // native oneDNN/OpenBLAS adoption) differs from double and is where the observed decline lives.
        const int n = 600;
        const int windowSize = 4;
        var data = new Tensor<float>(new[] { n, 1 });
        double p = 100.0;
        for (var i = 0; i < n; i++)
        {
            double nextRet = ((i / 4) % 2 == 0) ? 0.01 : -0.01;
            data[i, 0] = (float)p;
            p *= 1.0 + nextRet;
        }

        var env = new StockTradingEnvironment<float>(
            data, windowSize, initialCapital: 100_000f, tradeSize: 100f,
            transactionCost: 0.0, allowShortSelling: true, randomStart: false);

        var options = new FinancialPPOAgentOptions<float> { ContinuousActions = false };
        var agent = new FinancialPPOAgent<float>(
            new NeuralNetworkArchitecture<float>(inputFeatures: options.StateSize, outputSize: options.ActionSize),
            new NeuralNetworkArchitecture<float>(inputFeatures: options.StateSize, outputSize: 1),
            options);

        const int episodes = 60;
        const int stepsPerEpisode = 200;
        var episodeRewards = new double[episodes];
        for (var e = 0; e < episodes; e++)
        {
            var state = env.Reset();
            double total = 0;
            for (var s = 0; s < stepsPerEpisode; s++)
            {
                var action = agent.SelectAction(state, training: true);
                var (next, reward, done, _) = env.Step(action);
                total += reward;
                agent.StoreExperience(state, action, reward, next, done);
                if (s % 4 == 3 || done)
                {
                    agent.Train();
                }

                state = next;
                if (done)
                {
                    break;
                }
            }

            episodeRewards[e] = total;
        }

        var early = episodeRewards.Take(20).Average();
        var late = episodeRewards.Skip(40).Average();
        Assert.True(late > early,
            $"PPO<float> is not learning a PERFECT oracle: early {early:F3} vs late {late:F3} " +
            $"(curve: {string.Join(",", episodeRewards.Select(r => r.ToString("F2")))})");
    }

    private static NeuralNetworkArchitecture<double> Arch(int inputs, int outputs)
        => new(inputFeatures: inputs, outputSize: outputs);
}
