using System.Collections.Generic;
using AiDotNet.Finance.Trading.Environments;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// A subclass of <see cref="TradingEnvironment{T}"/> that keeps its own per-episode state must be able to reset
/// it from <see cref="TradingEnvironment{T}.Reset"/>: the hook runs on every reset, after the base bookkeeping
/// and before the first observation of the new episode is built.
/// </summary>
public sealed class TradingEnvironmentOnResetHookTests
{
    private sealed class EpisodeCountingEnvironment : TradingEnvironment<double>
    {
        public EpisodeCountingEnvironment(Tensor<double> data)
            : base(data, windowSize: 3, initialCapital: 1_000.0)
        {
        }

        public int TradesThisEpisode { get; private set; }

        public List<string> Events { get; } = new();

        public override int ActionSpaceSize => 3;

        public override bool IsContinuousActionSpace => false;

        protected override void ApplyAction(Vector<double> action, Vector<double> prices) => TradesThisEpisode++;

        protected override void OnReset()
        {
            Events.Add("OnReset");
            TradesThisEpisode = 0;
        }

        protected override Vector<double> BuildObservation(int step)
        {
            Events.Add("Observe");
            return base.BuildObservation(step);
        }
    }

    private static Tensor<double> Prices(int bars)
    {
        var data = new Tensor<double>(new[] { bars, 1 });
        for (int i = 0; i < bars; i++)
        {
            data[i, 0] = 100.0 + i;
        }

        return data;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Reset_invokes_the_subclass_hook_so_its_episode_state_starts_fresh()
    {
        var env = new EpisodeCountingEnvironment(Prices(20));
        var hold = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

        env.Reset();
        env.Step(hold);
        env.Step(hold);
        env.Step(hold);
        Assert.Equal(3, env.TradesThisEpisode);

        env.Reset();

        Assert.Equal(0, env.TradesThisEpisode);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Hook_runs_once_per_reset_before_the_first_observation_is_built()
    {
        var env = new EpisodeCountingEnvironment(Prices(20));

        env.Reset();
        Assert.Equal(new[] { "OnReset", "Observe" }, env.Events);

        env.Events.Clear();
        env.Step(new Vector<double>(new[] { 1.0, 0.0, 0.0 }));
        Assert.DoesNotContain("OnReset", env.Events);

        env.Events.Clear();
        env.Reset();
        Assert.Equal(new[] { "OnReset", "Observe" }, env.Events);
    }
}
