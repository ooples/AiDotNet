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
/// Judged over five seeds, not one. A single run is one draw of exploration and minibatch order, and a rounding change
/// can move it onto another trajectory (an unseeded CI run once rose to 0.31 and then collapsed to "hold"; Intel and AMD
/// exp kernels once sent SegNeXt's Adam down different paths). Four of five seeds must rise and so must their average,
/// so one unlucky draw does not fail the test while a regression in the update still does.
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
    private static readonly int[] Seeds = { 1, 2, 3, 4, 5 };

    [Fact(Timeout = 600000)]
    public async Task Ppo_training_reward_rises_on_a_perfect_oracle()
    {
        await Task.Yield();
        AssertRises("PPO<double>", Seeds.Select(seed => Curve(seed, (data, options) => new FinancialPPOAgent<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: options.StateSize, outputSize: options.ActionSize),
            new NeuralNetworkArchitecture<double>(inputFeatures: options.StateSize, outputSize: 1), options), Prices<double>(),
            v => v)).ToArray());
    }

    [Fact(Timeout = 600000)]
    public async Task Ppo_training_reward_rises_on_a_perfect_oracle_float()
    {
        // The downstream platform trains agents in float, whose kernels differ from double's.
        await Task.Yield();
        AssertRises("PPO<float>", Seeds.Select(seed => Curve(seed, (data, options) => new FinancialPPOAgent<float>(
            new NeuralNetworkArchitecture<float>(inputFeatures: options.StateSize, outputSize: options.ActionSize),
            new NeuralNetworkArchitecture<float>(inputFeatures: options.StateSize, outputSize: 1), options), Prices<float>(),
            v => (double)v)).ToArray());
    }

    private static void AssertRises(string name, double[][] curves)
    {
        var early = curves.Select(c => c.Take(20).Average()).ToArray();
        var late = curves.Select(c => c.Skip(40).Average()).ToArray();
        int rising = early.Zip(late, (e, l) => l > e).Count(r => r);
        string report = string.Join("; ", Seeds.Select((seed, i) => $"seed {seed}: {early[i]:F3} -> {late[i]:F3}"));
        Assert.True(rising >= 4 && late.Average() > early.Average(),
            $"{name} is not learning a PERFECT oracle: {rising} of {Seeds.Length} seeds rose ({report}).");
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
