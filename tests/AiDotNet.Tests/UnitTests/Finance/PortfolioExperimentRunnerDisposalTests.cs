using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Finance.Trading.Evaluation;
using AiDotNet.Finance.Trading.Rewards;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <see cref="PortfolioExperimentRunner"/> creates a fresh agent per experiment through the caller's factory, uses
/// it, and never hands it back, so the runner is the only owner and must release it -- on success and when
/// training or evaluation throws. Before this, every <see cref="RecurrentPolicyAgent{T}"/> it built kept its LSTM
/// cell (pool-rented weights) until the garbage collector ran.
/// </summary>
public sealed class PortfolioExperimentRunnerDisposalTests
{
    private static double[] Ramp(double start, double step, int n)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = start + i * step;
        return a;
    }

    private static readonly List<double[]> Prices = new() { Ramp(100, 2, 40), Ramp(100, 1, 40) };

    private static List<PortfolioExperiment> TwoExperiments() => new()
    {
        new("a", new TotalReturnReward()),
        new("b", new TotalReturnReward()),
    };

    [Fact]
    public void Every_recurrent_agent_the_runner_creates_is_disposed()
    {
        var created = new List<RecurrentPolicyAgent<double>>();

        var outcomes = PortfolioExperimentRunner.Run<double>(
            Prices, null, Prices, null, windowSize: 5, initialCapital: 100_000,
            TwoExperiments(),
            agentFactory: (stateDim, actionDim) =>
            {
                var agent = new RecurrentPolicyAgent<double>(stateDim, actionDim, hidden: 4, seed: created.Count);
                created.Add(agent);
                return agent;
            },
            trainEpisodes: 1);

        Assert.Equal(2, outcomes.Count);
        Assert.Equal(2, created.Count);
        foreach (var agent in created)
        {
            var cell = Assert.IsAssignableFrom<IDisposable>(
                typeof(RecurrentPolicyAgent<double>)
                    .GetField("_cell", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?.GetValue(agent));
            // False means the cell was already released through the once-only guard, i.e. the agent was disposed.
            Assert.False(DisposeOnceGuard.TryDispose(cell), "An agent the runner created was never disposed.");
        }
    }

    [Fact]
    public void An_agent_whose_training_throws_is_still_disposed_and_the_failure_surfaces()
    {
        var created = new List<SpyAgent>();

        var thrown = Assert.Throws<InvalidOperationException>(() => PortfolioExperimentRunner.Run<double>(
            Prices, null, Prices, null, windowSize: 5, initialCapital: 100_000,
            TwoExperiments(),
            agentFactory: (_, actionDim) =>
            {
                // The first experiment's agent is healthy; the second one fails during training.
                var agent = new SpyAgent(actionDim, failTraining: created.Count == 1, failDispose: false);
                created.Add(agent);
                return agent;
            },
            trainEpisodes: 1));

        Assert.Equal(SpyAgent.TrainingFailure, thrown.Message);
        Assert.Equal(2, created.Count);
        Assert.All(created, a => Assert.Equal(1, a.DisposeCalls));
    }

    [Fact]
    public void A_dispose_failure_does_not_mask_the_training_failure()
    {
        SpyAgent? created = null;

        var thrown = Assert.Throws<InvalidOperationException>(() => PortfolioExperimentRunner.Run<double>(
            Prices, null, Prices, null, windowSize: 5, initialCapital: 100_000,
            new List<PortfolioExperiment> { new("only", new TotalReturnReward()) },
            agentFactory: (_, actionDim) => created = new SpyAgent(actionDim, failTraining: true, failDispose: true),
            trainEpisodes: 1));

        // The caller sees why the run failed, not the secondary cleanup failure.
        Assert.Equal(SpyAgent.TrainingFailure, thrown.Message);
        Assert.Equal(1, Assert.IsType<SpyAgent>(created).DisposeCalls);
    }

    private sealed class SpyAgent : IPortfolioAgent<double>, IDisposable
    {
        public const string TrainingFailure = "simulated training failure";
        private readonly int _actionDim;
        private readonly bool _failTraining;
        private readonly bool _failDispose;

        public SpyAgent(int actionDim, bool failTraining, bool failDispose)
        {
            _actionDim = actionDim;
            _failTraining = failTraining;
            _failDispose = failDispose;
        }

        public int DisposeCalls { get; private set; }

        public Vector<double> SelectAction(Vector<double> state, bool explore) => new(new double[_actionDim]);

        public void StoreExperience(Vector<double> s, Vector<double> a, double r, Vector<double> n, bool d)
        {
        }

        public double Train() => _failTraining ? throw new InvalidOperationException(TrainingFailure) : 0.0;

        public void ResetEpisode()
        {
        }

        public void Dispose()
        {
            DisposeCalls++;
            if (_failDispose) throw new NotSupportedException("simulated dispose failure");
        }
    }
}
