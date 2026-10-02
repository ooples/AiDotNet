using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.CQL;
using AiDotNet.ReinforcementLearning.Agents.IQL;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// Regression for #2216: CQL and IQL squash their policy mean through MathHelper.Tanh, which returned
/// NaN for any input above ~44 in float. On unnormalised portfolio states (capital ~1e5) the untrained
/// mean is in the thousands, so both agents emitted NaN actions from the very first step, and the NaN
/// actions then poisoned every replayed batch.
/// </summary>
public class OfflineAgentContinuousActionTests
{
    private static PortfolioManagerEnvironment<float> Env()
    {
        var prices = new List<double[]>();
        for (int a = 0; a < 3; a++)
        {
            var series = new double[120];
            series[0] = 100 + 10 * a;
            for (int t = 1; t < series.Length; t++)
                series[t] = series[t - 1] * (1.0 + 0.001 * (a + 1) + 0.01 * Math.Sin(0.3 * t + a));
            prices.Add(series);
        }

        return new PortfolioManagerEnvironment<float>(prices, null, 4, 100_000, new TotalReturnReward(),
            maxLeverage: 2.0, transactionCost: 0.001, slippageCoefficient: 0.0005, annualBorrowCost: 0.03,
            annualHoldingCost: 0.0, allowShortSelling: true, seed: 17);
    }

    /// <summary>The issue's loop: act, step, store, train every 4 steps, 6 episodes x 60 steps.</summary>
    private static void AssertFiniteThroughTraining(DeepReinforcementLearningAgentBase<float> agent, PortfolioManagerEnvironment<float> env)
    {
        int step = 0;
        for (int ep = 0; ep < 6; ep++)
        {
            var state = env.Reset();
            for (int t = 0; t < 60; t++)
            {
                var action = agent.SelectAction(state, training: true);
                AssertFinite(action, $"training action at step {step}");
                var (next, reward, done, _) = env.Step(action);
                agent.StoreExperience(state, action, reward, next, done);
                if (++step % 4 == 0)
                {
                    float loss = agent.Train();
                    Assert.True(float.IsFinite(loss), $"training loss at step {step} was {loss}");
                }

                state = next;
                if (done) break;
            }
        }

        AssertFinite(agent.SelectAction(env.Reset(), training: false), "evaluation action after training");
    }

    private static void AssertFinite(Vector<float> action, string what)
    {
        Assert.Equal(3, action.Length);
        foreach (float v in action.ToArray())
            Assert.True(float.IsFinite(v), $"{what} contained {v}: [{string.Join(", ", action.ToArray())}]");
    }

    [Fact(Timeout = 300000)]
    public async Task CQL_ActionsStayFinite_OnMultiDimensionalContinuousActions()
    {
        await Task.Yield();
        var env = Env();
        AssertFiniteThroughTraining(new CQLAgent<float>(new CQLOptions<float>
        {
            StateSize = env.ObservationSpaceDimension, ActionSize = env.ActionSpaceSize,
            BatchSize = 16, BufferSize = 5_000, PolicyHiddenLayers = [64, 64], QHiddenLayers = [64, 64],
        }), env);
    }

    [Fact(Timeout = 300000)]
    public async Task IQL_ActionsStayFinite_OnMultiDimensionalContinuousActions()
    {
        await Task.Yield();
        var env = Env();
        AssertFiniteThroughTraining(new IQLAgent<float>(new IQLOptions<float>
        {
            StateSize = env.ObservationSpaceDimension, ActionSize = env.ActionSpaceSize,
            BatchSize = 16, BufferSize = 5_000, PolicyHiddenLayers = [64, 64], QHiddenLayers = [64, 64],
            ValueHiddenLayers = [64, 64],
        }), env);
    }
}
