using System.Collections.Generic;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Rewards;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <see cref="TradingEnvironment{T}.Reset"/> must reset per-episode state held by derived environments, not
/// only the base portfolio bookkeeping. <see cref="PortfolioManagerEnvironment{T}"/> keeps its own turnover,
/// exposure, drawdown peak and reward statistics; before the reset hook existed a caller that forgot the
/// separate <c>ResetEpisodeState()</c> call started every new episode with the previous episode's state.
/// </summary>
public sealed class TradingEnvironmentResetTests
{
    private static double[] Ramp(double start, double step, int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = start + (i * step);
        }

        return a;
    }

    [Fact]
    [Trait("category", "unit")]
    public void PortfolioManager_Reset_clears_the_previous_episodes_per_episode_state()
    {
        var prices = new List<double[]> { Ramp(100, 1, 40), Ramp(50, -0.5, 40) };
        var env = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());

        env.Reset();
        env.Step(new Vector<double>(new[] { 0.6, -0.4 }));
        env.Step(new Vector<double>(new[] { -0.5, 0.5 }));
        Assert.True(env.LastTurnover > 0, "precondition: the episode must have traded");
        Assert.True(env.GrossExposure > 0, "precondition: the episode must hold exposure");

        env.Reset();

        Assert.Equal(0.0, env.LastTurnover);
        Assert.Equal(0.0, env.GrossExposure);
        Assert.Equal(100_000.0, env.CurrentValue, 6);
    }

    [Fact]
    [Trait("category", "unit")]
    public void PortfolioManager_episodes_after_Reset_score_identically_to_a_fresh_environment()
    {
        var prices = new List<double[]> { Ramp(100, 1, 40), Ramp(50, -0.5, 40) };
        var actions = new[]
        {
            new Vector<double>(new[] { 0.6, -0.4 }),
            new Vector<double>(new[] { -0.5, 0.5 }),
            new Vector<double>(new[] { 0.3, 0.3 }),
        };

        double RunEpisode(PortfolioManagerEnvironment<double> env)
        {
            env.Reset();
            double total = 0;
            foreach (var action in actions)
            {
                var (_, reward, _, _) = env.Step(action);
                total += reward;
            }

            return total;
        }

        var reused = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());
        RunEpisode(reused); // leaves reward statistics / drawdown peak behind
        double second = RunEpisode(reused);

        var fresh = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());
        double expected = RunEpisode(fresh);

        Assert.Equal(expected, second, 12);
    }
}
