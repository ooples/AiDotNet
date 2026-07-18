using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.SAC;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the portfolio agent trainer + experiment runner (the greenfield trading-agent harness). Uses a
/// fixed-policy test double for the agent, so the loop mechanics and holdout-vs-baseline ranking are
/// deterministic — RL convergence itself is a research result, not a unit test.
/// </summary>
public sealed class PortfolioAgentTrainingTests
{
    /// <summary>An <see cref="IRLAgent{T}"/> that always emits the same target weights and counts lifecycle
    /// calls — lets a test drive the harness without RL stochasticity.</summary>
    private sealed class FixedWeightAgent : IPortfolioAgent<double>
    {
        private readonly double[] _weights;
        public int SelectCalls, StoreCalls, TrainCalls, ResetCalls;

        public FixedWeightAgent(double[] weights) => _weights = weights;

        public Vector<double> SelectAction(Vector<double> state, bool explore = true)
        {
            SelectCalls++;
            return new Vector<double>((double[])_weights.Clone());
        }

        public void StoreExperience(Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)
            => StoreCalls++;

        public double Train() { TrainCalls++; return 0.0; }

        public void ResetEpisode() => ResetCalls++;
    }

    private static double[] Ramp(double start, double step, int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = start + i * step;
        return a;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Trainer_drives_the_full_experience_loop_each_episode()
    {
        var env = new PortfolioManagerEnvironment<double>(
            new List<double[]> { Ramp(100, 1, 30) }, null, windowSize: 5, initialCapital: 100_000,
            reward: new TotalReturnReward());
        var agent = new FixedWeightAgent(new[] { 1.0 });

        double meanReturn = PortfolioAgentTrainer.Train(agent, env, episodes: 3);

        Assert.Equal(3, agent.ResetCalls);                     // one reset per episode
        Assert.True(agent.SelectCalls > 0);                    // stepped
        Assert.Equal(agent.SelectCalls, agent.StoreCalls);     // one transition stored per action
        Assert.Equal(agent.SelectCalls, agent.TrainCalls);     // learned each step
        Assert.True(double.IsFinite(meanReturn));
    }

    [Fact]
    [Trait("category", "unit")]
    public void Experiment_runner_flags_an_agent_that_beats_the_baseline_on_holdout()
    {
        // Holdout: asset 0 rises steadily, asset 1 DECLINES. An agent concentrated in the riser must beat the
        // equal-weight baseline on Sharpe — the baseline holds half its book in the decliner, which both lowers
        // return and adds variance, dragging its risk-adjusted score below the pure-riser agent's.
        var holdoutPrices = new List<double[]> { Ramp(100, 3, 50), Ramp(100, -1, 50) };
        var trainPrices = new List<double[]> { Ramp(100, 2, 50), Ramp(100, -0.8, 50) };
        var experiments = new List<PortfolioExperiment>
        {
            new("total-return", new TotalReturnReward()),
            new("diff-sharpe", new DifferentialSharpeReward()),
        };

        var outcomes = PortfolioExperimentRunner.Run<double>(
            trainPrices, null, holdoutPrices, null, windowSize: 5, initialCapital: 100_000,
            experiments, agentFactory: (_, _) => new FixedWeightAgent(new[] { 1.0, 0.0 }), trainEpisodes: 1);

        Assert.Equal(2, outcomes.Count);
        Assert.All(outcomes, o => Assert.True(o.BeatBaseline, $"{o.Name}: agent Sharpe {o.Agent.AnnualizedSharpe} vs base {o.BaselineEqualWeight.AnnualizedSharpe}"));
        // Ranked by holdout Sharpe (descending).
        Assert.True(outcomes[0].Agent.AnnualizedSharpe >= outcomes[1].Agent.AnnualizedSharpe);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Experiment_runner_does_not_flag_a_flat_agent_as_beating_the_baseline()
    {
        // A do-nothing (all-cash) agent earns zero on a rising market; equal-weight earns a positive Sharpe, so
        // the flat agent must NOT be flagged as beating it — the honest-eval guard against false positives.
        var prices = new List<double[]> { Ramp(100, 2, 50), Ramp(100, 2, 50) };
        var outcomes = PortfolioExperimentRunner.Run<double>(
            prices, null, prices, null, windowSize: 5, initialCapital: 100_000,
            new List<PortfolioExperiment> { new("flat", new TotalReturnReward()) },
            agentFactory: (_, _) => new FixedWeightAgent(new[] { 0.0, 0.0 }), trainEpisodes: 1);

        Assert.Single(outcomes);
        Assert.False(outcomes[0].BeatBaseline);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Frictions_are_an_experiment_axis_carry_cost_drags_the_holdout()
    {
        // Same objective + policy; two experiments differ ONLY in holding/carry cost. The held long book pays
        // carry every step, so the heavy-friction experiment must end below the frictionless one — proving
        // frictions are a real experiment axis, not a fixed assumption.
        var prices = new List<double[]> { Ramp(100, 1, 50) };
        var experiments = new List<PortfolioExperiment>
        {
            new("frictionless", new TotalReturnReward(), MaxLeverage: 1.0, Frictions: new PortfolioFrictions(AnnualHoldingCost: 0.0)),
            new("heavy-carry", new TotalReturnReward(), MaxLeverage: 1.0, Frictions: new PortfolioFrictions(AnnualHoldingCost: 1.0)),
        };

        var outcomes = PortfolioExperimentRunner.Run<double>(
            prices, null, prices, null, windowSize: 5, initialCapital: 100_000,
            experiments, agentFactory: (_, _) => new FixedWeightAgent(new[] { 1.0 }), trainEpisodes: 1);

        double frictionless = System.Linq.Enumerable.First(outcomes, o => o.Name == "frictionless").Agent.FinalValue;
        double heavyCarry = System.Linq.Enumerable.First(outcomes, o => o.Name == "heavy-carry").Agent.FinalValue;
        Assert.True(frictionless > heavyCarry, $"carry should drag the book: frictionless {frictionless} !> heavy {heavyCarry}");
    }

    [Fact(Timeout = 120000)]
    [Trait("category", "unit")]
    public async Task Harness_trains_a_real_SAC_agent_end_to_end_and_produces_finite_holdout_metrics()
    {
        await Task.Yield();
        // Integration smoke: drive a REAL AiDotNet SAC agent (via the adapter) through the trainer + holdout
        // evaluator. Asserts the wiring runs end-to-end and yields finite, sane metrics — NOT that it converges
        // (RL convergence is a research result, not a unit test).
        var prices = new List<double[]> { Ramp(100, 1, 40), Ramp(100, 0.5, 40) };
        var trainEnv = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());

        var options = new SACOptions<double>
        {
            StateSize = trainEnv.ObservationSpaceDimension,
            ActionSize = trainEnv.ActionSpaceSize,
            PolicyLearningRate = 3e-4,
            QLearningRate = 3e-4,
            AlphaLearningRate = 3e-4,
            DiscountFactor = 0.99,
            InitialTemperature = 0.2,
            TargetUpdateTau = 0.005,
            BatchSize = 8,
            ReplayBufferSize = 2000,
            WarmupSteps = 4,
            PolicyHiddenLayers = new List<int> { 32, 32 },
            QHiddenLayers = new List<int> { 32, 32 },
            Seed = 1,
        };
        var agent = PortfolioAgent.From(new SACAgent<double>(options));

        double meanReturn = PortfolioAgentTrainer.Train(agent, trainEnv, episodes: 2);
        Assert.True(double.IsFinite(meanReturn), $"training return {meanReturn} was not finite");

        var evalEnv = new PortfolioManagerEnvironment<double>(
            prices, null, windowSize: 5, initialCapital: 100_000, reward: new DifferentialSharpeReward());
        var result = PortfolioBacktest.Run(evalEnv, s => agent.SelectAction(s, explore: false));

        Assert.True(double.IsFinite(result.FinalValue) && result.FinalValue > 0, $"final value {result.FinalValue}");
        Assert.True(result.Steps > 0);
    }
}
