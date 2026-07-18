using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Tests for the study entry point: the experiment-matrix cross-product and the one-call walk-forward study that
/// splits the full history and ranks experiments on the untouched holdout.
/// </summary>
public sealed class PortfolioStudyTests
{
    private sealed class FixedWeightAgent : IPortfolioAgent<double>
    {
        private readonly double[] _w;
        public FixedWeightAgent(double[] w) => _w = w;
        public Vector<double> SelectAction(Vector<double> state, bool explore) => new((double[])_w.Clone());
        public void StoreExperience(Vector<double> s, Vector<double> a, double r, Vector<double> n, bool d) { }
        public double Train() => 0.0;
        public void ResetEpisode() { }
    }

    private static double[] Ramp(double start, double step, int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = start + i * step;
        return a;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Matrix_produces_the_full_cross_product()
    {
        var rewards = new List<(string, IPortfolioReward)>
        {
            ("total", new TotalReturnReward()),
            ("sharpe", new DifferentialSharpeReward()),
        };
        var leverages = new List<double> { 1.0, 2.0 };
        var frictions = new List<(string, PortfolioFrictions)>
        {
            ("cheap", new PortfolioFrictions(TransactionCost: 0.0005)),
            ("dear", new PortfolioFrictions(TransactionCost: 0.005)),
        };

        var grid = PortfolioExperimentMatrix.Cross(rewards, leverages, frictions);

        Assert.Equal(2 * 2 * 2, grid.Count);                 // reward x leverage x frictions
        Assert.Equal(grid.Count, grid.Select(e => e.Name).Distinct().Count()); // unique names
        Assert.Contains(grid, e => e.MaxLeverage == 2.0);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Matrix_defaults_to_single_leverage_and_frictions_when_omitted()
    {
        var grid = PortfolioExperimentMatrix.Cross(
            new List<(string, IPortfolioReward)> { ("r", new TotalReturnReward()) });
        Assert.Single(grid);
        Assert.Equal(1.0, grid[0].MaxLeverage);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Study_splits_walk_forward_and_ranks_experiments_on_the_holdout()
    {
        // Full 100-bar history; the study trains on the first 70% and evaluates on the untouched last 30%.
        var prices = new List<double[]> { Ramp(100, 2, 100), Ramp(100, -1, 100) };
        var experiments = PortfolioExperimentMatrix.Cross(
            new List<(string, IPortfolioReward)> { ("total", new TotalReturnReward()), ("sharpe", new DifferentialSharpeReward()) });

        var outcomes = PortfolioStudy.Run<double>(
            prices, null, trainFraction: 0.7, embargo: 3, windowSize: 5, initialCapital: 100_000,
            experiments, agentFactory: (_, _) => new FixedWeightAgent(new[] { 1.0, 0.0 }), trainEpisodes: 1);

        Assert.Equal(2, outcomes.Count);
        // Ranked by holdout Sharpe, descending.
        Assert.True(outcomes[0].Agent.AnnualizedSharpe >= outcomes[1].Agent.AnnualizedSharpe);
        // The riser-concentrated agent beats equal-weight (which holds the decliner) on the holdout.
        Assert.All(outcomes, o => Assert.True(o.BeatBaseline));
        // Holdout has ~30 bars minus embargo minus the window warm-up — a positive, sane step count.
        Assert.All(outcomes, o => Assert.True(o.Agent.Steps > 0 && o.Agent.Steps < 30));
    }
}
