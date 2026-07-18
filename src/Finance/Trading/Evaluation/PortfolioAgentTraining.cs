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

/// <summary>One experiment in the greenfield trading-agent research: a named objective to train + evaluate.
/// Swapping the <see cref="Reward"/> (and, later, frictions / policy / algorithm) is how the options become
/// experiments ranked on the holdout rather than an either/or pick.</summary>
public sealed record PortfolioExperiment(string Name, IPortfolioReward Reward);

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
        double maxLeverage,
        IReadOnlyList<PortfolioExperiment> experiments,
        Func<int, int, IPortfolioAgent<T>> agentFactory,
        int trainEpisodes)
    {
        ArgumentNullException.ThrowIfNull(trainAssetPrices);
        ArgumentNullException.ThrowIfNull(holdoutAssetPrices);
        ArgumentNullException.ThrowIfNull(experiments);
        ArgumentNullException.ThrowIfNull(agentFactory);

        int tradableCount = holdoutAssetPrices.Count;
        var outcomes = new List<PortfolioExperimentOutcome>(experiments.Count);

        foreach (var experiment in experiments)
        {
            // Train a fresh agent on the training split under this experiment's objective.
            var trainEnv = new PortfolioManagerEnvironment<T>(
                trainAssetPrices, trainFeatureColumns, windowSize, initialCapital, experiment.Reward, maxLeverage);
            var agent = agentFactory(trainEnv.ObservationSpaceDimension, trainEnv.ActionSpaceSize);
            PortfolioAgentTrainer.Train(agent, trainEnv, trainEpisodes);

            // Evaluate the trained agent on the untouched holdout (greedy — no exploration).
            var agentEnv = new PortfolioManagerEnvironment<T>(
                holdoutAssetPrices, holdoutFeatureColumns, windowSize, initialCapital, experiment.Reward, maxLeverage);
            var agentResult = PortfolioBacktest.Run(agentEnv, s => agent.SelectAction(s, explore: false));

            // The no-skill control on the SAME holdout.
            var baseEnv = new PortfolioManagerEnvironment<T>(
                holdoutAssetPrices, holdoutFeatureColumns, windowSize, initialCapital, experiment.Reward, maxLeverage);
            var baseResult = PortfolioBacktest.Run(baseEnv, BaselinePolicies.EqualWeight<T>(tradableCount));

            outcomes.Add(new PortfolioExperimentOutcome(
                experiment.Name, agentResult, baseResult, agentResult.AnnualizedSharpe > baseResult.AnnualizedSharpe));
        }

        // Best holdout Sharpe first.
        return outcomes.OrderByDescending(o => o.Agent.AnnualizedSharpe).ToList();
    }
}
