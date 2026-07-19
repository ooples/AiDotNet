using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Finance.Trading.Evaluation;

/// <summary>
/// The MINIMAL agent surface the portfolio trainer needs: choose an action, remember a transition, learn, and
/// reset per episode. Kept narrow (rather than the full <see cref="IRLAgent{T}"/>, which pulls in the entire
/// IFullModel surface) so the trainer depends on nothing more than the RL loop, and a test double is trivial.
/// Wrap any real AiDotNet agent with <see cref="PortfolioAgent.From{T}"/>.
/// </summary>
public interface IPortfolioAgent<T>
{
    Vector<T> SelectAction(Vector<T> state, bool explore);
    void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done);
    T Train();
    void ResetEpisode();
}

/// <summary>Adapts a full <see cref="IRLAgent{T}"/> (PPO/SAC/TD3/…) to the narrow <see cref="IPortfolioAgent{T}"/>.</summary>
public static class PortfolioAgent
{
    public static IPortfolioAgent<T> From<T>(IRLAgent<T> agent) => new RlAgentAdapter<T>(agent);

    private sealed class RlAgentAdapter<T> : IPortfolioAgent<T>
    {
        private readonly IRLAgent<T> _agent;
        public RlAgentAdapter(IRLAgent<T> agent) => _agent = agent ?? throw new ArgumentNullException(nameof(agent));
        public Vector<T> SelectAction(Vector<T> state, bool explore) => _agent.SelectAction(state, explore);
        public void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState, bool done)
            => _agent.StoreExperience(state, action, reward, nextState, done);
        public T Train() => _agent.Train();
        public void ResetEpisode() => _agent.ResetEpisode();
    }
}

/// <summary>
/// Drives one agent through a <see cref="PortfolioManagerEnvironment{T}"/> for a number of episodes — the
/// standard experience-collection loop (select action, step, store transition, learn).
/// </summary>
public static class PortfolioAgentTrainer
{
    /// <summary>
    /// Trains <paramref name="agent"/> on <paramref name="environment"/> for <paramref name="episodes"/> passes.
    /// Each episode resets the environment (and its per-episode reward/peak state), then loops
    /// select → step → store → learn until the episode ends. Returns the mean episode return (sum of rewards),
    /// a rough progress signal — NOT a performance claim; rank agents on a held-out <see cref="PortfolioBacktest"/>.
    /// </summary>
    public static double Train<T>(IPortfolioAgent<T> agent, PortfolioManagerEnvironment<T> environment, int episodes)
    {
        ArgumentNullException.ThrowIfNull(agent);
        ArgumentNullException.ThrowIfNull(environment);
        if (episodes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(episodes));
        }

        double totalEpisodeReturn = 0;
        for (int ep = 0; ep < episodes; ep++)
        {
            environment.ResetEpisodeState();
            agent.ResetEpisode();
            var state = environment.Reset();

            bool done = false;
            double episodeReturn = 0;
            while (!done)
            {
                var action = agent.SelectAction(state, explore: true);
                var (nextState, reward, isDone, _) = environment.Step(action);
                agent.StoreExperience(state, action, reward, nextState, isDone);
                agent.Train();

                episodeReturn += Convert.ToDouble(reward);
                state = nextState;
                done = isDone;
            }

            totalEpisodeReturn += episodeReturn;
        }

        return totalEpisodeReturn / episodes;
    }
}

/// <summary>The friction knobs for an experiment's environment — a market-realism axis. Cheaper frictions flatter
/// a churning policy; heavier ones reward patience. Sweeping them is how "is the edge real net of costs?" becomes
/// an experiment rather than an assumption.</summary>
public sealed record PortfolioFrictions(
    double TransactionCost = 0.001,
    double SlippageCoefficient = 0.0005,
    double AnnualBorrowCost = 0.03,
    double AnnualHoldingCost = 0.0);

/// <summary>One experiment in the greenfield trading-agent research: a named environment configuration to train +
/// evaluate. Every SOTA-relevant axis is a knob here — objective (<see cref="Reward"/>), leverage budget
/// (<see cref="MaxLeverage"/>), and market frictions (<see cref="Frictions"/>) — so the options become experiments
/// ranked on the holdout rather than an either/or pick. (Policy/algorithm is the agent factory's axis.)</summary>
public sealed record PortfolioExperiment(
    string Name,
    IPortfolioReward Reward,
    double MaxLeverage = 1.0,
    PortfolioFrictions? Frictions = null)
{
    /// <summary>Builds the environment this experiment specifies over the given data.</summary>
    public PortfolioManagerEnvironment<T> BuildEnvironment<T>(
        IReadOnlyList<double[]> assetPrices, IReadOnlyList<double[]>? featureColumns, int windowSize, double initialCapital)
    {
        var f = Frictions ?? new PortfolioFrictions();
        return new PortfolioManagerEnvironment<T>(
            assetPrices, featureColumns, windowSize, initialCapital, Reward, MaxLeverage,
            f.TransactionCost, f.SlippageCoefficient, f.AnnualBorrowCost, f.AnnualHoldingCost);
    }
}

/// <summary>The holdout result of one experiment: the trained agent's performance vs the no-skill equal-weight
/// baseline on the untouched holdout, and whether it beat the baseline (higher Sharpe).</summary>
public sealed record PortfolioExperimentOutcome(
    string Name,
    PortfolioBacktestResult Agent,
    PortfolioBacktestResult BaselineEqualWeight,
    bool BeatBaseline);

/// <summary>
/// Runs a set of reward experiments: for each, train a FRESH agent on the training split, then evaluate it and
/// the equal-weight baseline on the UNTOUCHED holdout split, and rank by holdout Sharpe. This is the honest-eval
/// harness for the trading-agent research — chronological train/holdout, vs-baseline, ranked on risk-adjusted
/// holdout performance, never training loss.
/// </summary>
public static class PortfolioExperimentRunner
{
    public static IReadOnlyList<PortfolioExperimentOutcome> Run<T>(
        IReadOnlyList<double[]> trainAssetPrices,
        IReadOnlyList<double[]>? trainFeatureColumns,
        IReadOnlyList<double[]> holdoutAssetPrices,
        IReadOnlyList<double[]>? holdoutFeatureColumns,
        int windowSize,
        double initialCapital,
        IReadOnlyList<PortfolioExperiment> experiments,
        Func<int, int, IPortfolioAgent<T>> agentFactory,
        int trainEpisodes)
    {
        ArgumentNullException.ThrowIfNull(trainAssetPrices);
        ArgumentNullException.ThrowIfNull(holdoutAssetPrices);
        ArgumentNullException.ThrowIfNull(experiments);
        ArgumentNullException.ThrowIfNull(agentFactory);

        // Train and holdout must share the same observation schema — same tradable-asset count and the same
        // number of feature columns — or the agent trained on one cannot be evaluated on the other.
        if (trainAssetPrices.Count != holdoutAssetPrices.Count)
        {
            throw new ArgumentException(
                $"Train and holdout must have the same number of tradable assets ({trainAssetPrices.Count} vs {holdoutAssetPrices.Count}).",
                nameof(holdoutAssetPrices));
        }

        int trainFeatures = trainFeatureColumns?.Count ?? 0;
        int holdoutFeatures = holdoutFeatureColumns?.Count ?? 0;
        if (trainFeatures != holdoutFeatures)
        {
            throw new ArgumentException(
                $"Train and holdout must have the same number of feature columns ({trainFeatures} vs {holdoutFeatures}).",
                nameof(holdoutFeatureColumns));
        }

        int tradableCount = holdoutAssetPrices.Count;
        var outcomes = new List<PortfolioExperimentOutcome>(experiments.Count);

        foreach (var experiment in experiments)
        {
            // Each experiment fully specifies its environment (reward + leverage + frictions), so the harness
            // sweeps all those axes, not just the objective.
            var trainEnv = experiment.BuildEnvironment<T>(trainAssetPrices, trainFeatureColumns, windowSize, initialCapital);
            var agent = agentFactory(trainEnv.ObservationSpaceDimension, trainEnv.ActionSpaceSize);
            PortfolioAgentTrainer.Train(agent, trainEnv, trainEpisodes);

            // Evaluate the trained agent on the untouched holdout (greedy — no exploration). Reset the agent's
            // per-episode state first so a recurrent policy starts the holdout with a clean hidden state rather
            // than one carried over from the last training step.
            agent.ResetEpisode();
            var agentEnv = experiment.BuildEnvironment<T>(holdoutAssetPrices, holdoutFeatureColumns, windowSize, initialCapital);
            var agentResult = PortfolioBacktest.Run(agentEnv, s => agent.SelectAction(s, explore: false));

            // The no-skill control on the SAME holdout environment (same frictions/leverage).
            var baseEnv = experiment.BuildEnvironment<T>(holdoutAssetPrices, holdoutFeatureColumns, windowSize, initialCapital);
            var baseResult = PortfolioBacktest.Run(baseEnv, BaselinePolicies.EqualWeight<T>(tradableCount));

            outcomes.Add(new PortfolioExperimentOutcome(
                experiment.Name, agentResult, baseResult, agentResult.AnnualizedSharpe > baseResult.AnnualizedSharpe));
        }

        // Best holdout Sharpe first.
        return outcomes.OrderByDescending(o => o.Agent.AnnualizedSharpe).ToList();
    }
}
