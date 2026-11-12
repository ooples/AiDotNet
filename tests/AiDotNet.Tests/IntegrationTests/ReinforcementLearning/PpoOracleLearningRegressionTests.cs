using System.Threading.Tasks;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// PPO's per-episode training reward must RISE on a trivially learnable oracle environment: a deterministic 8-bar
/// zig-zag whose phase the 4-bar price window identifies exactly. Downstream evidence first: this rose on the 0.208.117
/// package and declined on 0.208.119.
/// </summary>
/// <remarks>
/// <para>
/// Judged per seed, each one fixed so a run is a repeatable trajectory rather than one draw of exploration and minibatch
/// order: every seed must rise AND end above 0.05, since random drift alone can rise. Seeds 0, 1 and 42 are master's;
/// 2 to 5 widen the sample.
/// </para>
/// <para>
/// The rollout is one 200-step episode (RolloutLength = 200), a setting chosen for this environment: the paper's 2048
/// steps would give 60 episodes only five updates. Measured on seeds 1-8: every seed rises, the late average 0.085 to
/// 0.82 against an early 0.017 to 0.073 (at 2048 steps every seed also rises, but by as little as 0.004 to 0.005).
/// </para>
/// </remarks>
public class PpoOracleLearningRegressionTests
{
    private const int Episodes = 60;
    private const int StepsPerEpisode = 200;

    [Theory(Timeout = 600000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(42)]
    public async Task Ppo_training_reward_rises_on_a_perfect_oracle(int seed)
    {
        await Task.Yield();
        AssertRises("PPO<double>", seed, Curve(seed, (data, options) => new FinancialPPOAgent<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: options.StateSize, outputSize: options.ActionSize),
            new NeuralNetworkArchitecture<double>(inputFeatures: options.StateSize, outputSize: 1), options), Prices<double>(),
            v => v));
    }

    [Theory(Timeout = 600000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(42)]
    public async Task Ppo_training_reward_rises_on_a_perfect_oracle_float(int seed)
    {
        // The downstream platform trains agents in float, whose kernels differ from double's.
        await Task.Yield();
        AssertRises("PPO<float>", seed, Curve(seed, (data, options) => new FinancialPPOAgent<float>(
            new NeuralNetworkArchitecture<float>(inputFeatures: options.StateSize, outputSize: options.ActionSize),
            new NeuralNetworkArchitecture<float>(inputFeatures: options.StateSize, outputSize: 1), options), Prices<float>(),
            v => (double)v));
    }

    private static void AssertRises(string name, int seed, double[] curve)
    {
        double early = curve.Take(20).Average();
        double late = curve.Skip(40).Average();
        // Require a meaningful positive reward as well as improvement; random drift alone can rise.
        Assert.True(late > early && late > 0.05,
            $"{name} (seed {seed}) is not learning a PERFECT oracle: early-episodes avg {early:F3} vs late {late:F3} " +
            $"(full curve: {string.Join(",", curve.Select(r => r.ToString("F2")))})");
    }

    // An 8-bar zig-zag of +-1% returns, one asset.
    private static Tensor<T> Prices<T>()
    {
        const int n = 600;
        var data = new Tensor<T>(new[] { n, 1 });
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        double p = 100.0;
        for (var i = 0; i < n; i++)
        {
            double nextRet = ((i / 4) % 2 == 0) ? 0.01 : -0.01;
            data[i, 0] = ops.FromDouble(p);
            p *= 1.0 + nextRet;
        }
        return data;
    }

    private static double[] Curve<T>(int seed, Func<Tensor<T>, FinancialPPOAgentOptions<T>, FinancialPPOAgent<T>> create,
        Tensor<T> data, Func<T, double> toDouble)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var env = new StockTradingEnvironment<T>(
            data, windowSize: 4, initialCapital: ops.FromDouble(100_000.0), tradeSize: ops.FromDouble(100.0),
            transactionCost: 0.0, allowShortSelling: true, randomStart: false);
        var options = new FinancialPPOAgentOptions<T> { ContinuousActions = false, Seed = seed, RolloutLength = StepsPerEpisode };
        using var agent = create(data, options);

        var rewards = new double[Episodes];
        for (var e = 0; e < Episodes; e++)
        {
            var state = env.Reset();
            double total = 0;
            for (var s = 0; s < StepsPerEpisode; s++)
            {
                var action = agent.SelectAction(state, training: true);
                var (next, reward, done, _) = env.Step(action);
                total += toDouble(reward);
                agent.StoreExperience(state, action, reward, next, done);
                agent.Train();
                state = next;
                if (done) break;
            }
            rewards[e] = total;
        }
        return rewards;
    }
}
