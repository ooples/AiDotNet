using System;
using System.Collections.Generic;
using AiDotNet.Finance.Trading.Rewards;

namespace AiDotNet.Finance.Trading.Evaluation;

/// <summary>
/// Generates the cross-product of experiment axes — objective x leverage x frictions — as a flat list of
/// <see cref="PortfolioExperiment"/>. Turns "which reward / how much leverage / what cost regime?" into a swept
/// grid ranked on the holdout, the research framing for the greenfield trading-agent core.
/// </summary>
public static class PortfolioExperimentMatrix
{
    public static List<PortfolioExperiment> Cross(
        IReadOnlyList<(string Name, IPortfolioReward Reward)> rewards,
        IReadOnlyList<double>? leverages = null,
        IReadOnlyList<(string Name, PortfolioFrictions Frictions)>? frictions = null)
    {
        ArgumentNullException.ThrowIfNull(rewards);
        if (rewards.Count == 0)
        {
            throw new ArgumentException("At least one reward is required.", nameof(rewards));
        }

        var levels = leverages is { Count: > 0 } ? leverages : new List<double> { 1.0 };
        var frics = frictions is { Count: > 0 } ? frictions : new List<(string, PortfolioFrictions)> { ("default", new PortfolioFrictions()) };

        var experiments = new List<PortfolioExperiment>(rewards.Count * levels.Count * frics.Count);
        foreach (var (rewardName, reward) in rewards)
        {
            foreach (var lev in levels)
            {
                foreach (var (fricName, fric) in frics)
                {
                    experiments.Add(new PortfolioExperiment(
                        $"{rewardName}|lev{lev:0.##}|{fricName}", reward, lev, fric));
                }
            }
        }

        return experiments;
    }
}

/// <summary>
/// One-call research entry point: given the FULL history, split it walk-forward (train on the past, hold out a
/// strictly-later tail with an embargo gap), run every experiment through
/// <see cref="PortfolioExperimentRunner"/>, and return the outcomes ranked by holdout Sharpe. This is the honest
/// study loop end-to-end — the seam a researcher calls to compare candidate objectives / leverage / frictions /
/// agents on data the agents never trained on.
/// </summary>
public static class PortfolioStudy
{
    public static IReadOnlyList<PortfolioExperimentOutcome> Run<T>(
        IReadOnlyList<double[]> assetPrices,
        IReadOnlyList<double[]>? featureColumns,
        double trainFraction,
        int embargo,
        int windowSize,
        double initialCapital,
        IReadOnlyList<PortfolioExperiment> experiments,
        Func<int, int, IPortfolioAgent<T>> agentFactory,
        int trainEpisodes)
    {
        ArgumentNullException.ThrowIfNull(assetPrices);

        var (trainPrices, holdoutPrices) = WalkForwardSplit.Split(assetPrices, trainFraction, embargo);

        List<double[]>? trainFeatures = null, holdoutFeatures = null;
        if (featureColumns is { Count: > 0 })
        {
            var (tf, hf) = WalkForwardSplit.Split(featureColumns, trainFraction, embargo);
            trainFeatures = tf;
            holdoutFeatures = hf;
        }

        return PortfolioExperimentRunner.Run(
            trainPrices, trainFeatures, holdoutPrices, holdoutFeatures,
            windowSize, initialCapital, experiments, agentFactory, trainEpisodes);
    }
}
