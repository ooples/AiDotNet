using System;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class ExplorationStrategiesIntegrationTests
{
    [Fact]
    public void NoExploration_ReturnsPolicyAction()
    {
        var strategy = new NoExploration<double>();
        var policyAction = CreateVector(3, 0.2);

        var action = strategy.GetExplorationAction(new Vector<double>(1), policyAction, 3, new Random(1));

        Assert.Same(policyAction, action);
    }

    [Fact]
    public void EpsilonGreedyExploration_UpdateAndReset_AdjustsEpsilon()
    {
        var strategy = new EpsilonGreedyExploration<double>(epsilonStart: 1.0, epsilonEnd: 0.1, epsilonDecay: 0.5);
        var policyAction = CreateOneHotAction(3, 2);

        var randomAction = strategy.GetExplorationAction(new Vector<double>(1), policyAction, 3, new Random(2));

        AssertOneHot(randomAction, 3);
        Assert.Equal(1.0, strategy.CurrentEpsilon, precision: 12);

        strategy.Update();
        Assert.Equal(0.5, strategy.CurrentEpsilon, precision: 12);

        strategy.Reset();
        Assert.Equal(1.0, strategy.CurrentEpsilon, precision: 12);
    }

    [Fact]
    public void BoltzmannExploration_HandlesDiscreteAndContinuousActions()
    {
        var strategy = new BoltzmannExploration<double>(temperatureStart: 1.0, temperatureEnd: 0.1, temperatureDecay: 1.0);
        var discreteAction = CreateOneHotAction(3, 1);
        var continuousAction = CreateVector(2, 0.25);

        var discreteResult = strategy.GetExplorationAction(new Vector<double>(1), discreteAction, 3, new Random(3));
        AssertOneHot(discreteResult, 3);

        var continuousResult = strategy.GetExplorationAction(new Vector<double>(1), continuousAction, 2, new Random(3));
        AssertVectorInRange(continuousResult, 2, -1.0, 1.0);

        strategy.Update();
        strategy.Reset();
    }

    [Fact]
    public void GaussianNoiseExploration_ZeroNoise_KeepsAction()
    {
        var strategy = new GaussianNoiseExploration<double>(initialStdDev: 0.0, noiseDecay: 1.0, minNoise: 0.0);
        var policyAction = CreateVector(2, 0.3);

        var result = strategy.GetExplorationAction(new Vector<double>(1), policyAction, 2, new Random(4));

        Assert.Equal(policyAction.Length, result.Length);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(policyAction[i], result[i]);
        }

        strategy.Update();
    }

    [Fact]
    public void OrnsteinUhlenbeckNoise_ValidatesActionSizeAndResets()
    {
        var strategy = new OrnsteinUhlenbeckNoise<double>(actionSize: 2);
        var policyAction = CreateVector(2, 0.1);

        var result = strategy.GetExplorationAction(new Vector<double>(1), policyAction, 2, new Random(5));
        AssertVectorInRange(result, 2, -1.0, 1.0);

        Assert.Throws<ArgumentException>(() => strategy.GetExplorationAction(new Vector<double>(1), policyAction, 3, new Random(5)));

        strategy.Reset();
    }

    [Fact]
    public void UpperConfidenceBoundExploration_TracksSteps()
    {
        var strategy = new UpperConfidenceBoundExploration<double>(explorationConstant: 1.0);
        var qValues = CreateVector(3, 0.1);

        var firstAction = strategy.GetExplorationAction(new Vector<double>(1), qValues, 3, new Random(6));
        AssertOneHot(firstAction, 3);
        Assert.Equal(1, strategy.TotalSteps);

        var secondAction = strategy.GetExplorationAction(new Vector<double>(1), qValues, 3, new Random(6));
        AssertOneHot(secondAction, 3);
        Assert.Equal(2, strategy.TotalSteps);

        strategy.Reset();
        Assert.Equal(0, strategy.TotalSteps);
    }

    [Fact]
    public void ThompsonSamplingExploration_UpdateDistributionAndReset()
    {
        var strategy = new ThompsonSamplingExploration<double>(priorAlpha: 1.0, priorBeta: 1.0);
        var policyAction = CreateVector(2, 0.1);

        var action = strategy.GetExplorationAction(new Vector<double>(1), policyAction, 2, new Random(7));
        AssertOneHot(action, 2);

        strategy.UpdateDistribution(actionIndex: 0, reward: 1.0);
        strategy.UpdateDistribution(actionIndex: 1, reward: 0.0);

        strategy.Reset();
    }

    private static Vector<double> CreateVector(int size, double start)
    {
        var vector = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = start + i * 0.05;
        }

        return vector;
    }

    private static Vector<double> CreateOneHotAction(int size, int index)
    {
        var action = new Vector<double>(size);
        action[index] = 1.0;
        return action;
    }

    private static void AssertOneHot(Vector<double> action, int actionSize)
    {
        Assert.Equal(actionSize, action.Length);

        int nonZeroCount = 0;
        double nonZeroValue = 0.0;
        for (int i = 0; i < action.Length; i++)
        {
            if (Math.Abs(action[i]) > 1e-6)
            {
                nonZeroCount++;
                nonZeroValue = action[i];
            }
        }

        Assert.Equal(1, nonZeroCount);
        Assert.Equal(1.0, nonZeroValue, precision: 6);
    }

    private static void AssertVectorInRange(Vector<double> action, int actionSize, double min, double max)
    {
        Assert.Equal(actionSize, action.Length);
        for (int i = 0; i < action.Length; i++)
        {
            Assert.InRange(action[i], min, max);
        }
    }
}
