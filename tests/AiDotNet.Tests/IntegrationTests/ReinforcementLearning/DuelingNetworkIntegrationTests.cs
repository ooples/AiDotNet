using AiDotNet.LossFunctions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.DuelingDQN;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class DuelingNetworkIntegrationTests
{
    [Fact]
    public void Train_WithReplayExperience_UpdatesParameters()
    {
        var agent = new DuelingDQNAgent<double>(new DuelingDQNOptions<double>
        {
            StateSize = 2,
            ActionSize = 3,
            LearningRate = 0.01,
            DiscountFactor = 0.9,
            LossFunction = new MeanSquaredErrorLoss<double>(),
            BatchSize = 1,
            ReplayBufferSize = 4,
            TargetUpdateFrequency = 10,
            WarmupSteps = 0,
            EpsilonStart = 0.0,
            EpsilonEnd = 0.0,
            EpsilonDecay = 1.0,
            SharedLayers = new List<int> { 4 },
            ValueStreamLayers = new List<int> { 4 },
            AdvantageStreamLayers = new List<int> { 4 },
            Seed = 123
        });

        var state = CreateVector(0.1);
        var nextState = CreateVector(0.2);
        var action = agent.SelectAction(state, training: true);
        var before = agent.GetParameters();

        agent.StoreExperience(state, action, 1.0, nextState, done: true);
        var loss = agent.Train();

        Assert.False(double.IsNaN(loss));
        var after = agent.GetParameters();
        Assert.Equal(before.Length, after.Length);
        Assert.True(HasParameterChange(before, after), "DuelingDQNAgent.Train should update network parameters.");
    }

    private static Vector<double> CreateVector(double start)
    {
        var vector = new Vector<double>(2);
        vector[0] = start;
        vector[1] = start + 0.05;
        return vector;
    }

    private static bool HasParameterChange(Vector<double> before, Vector<double> after)
    {
        for (int i = 0; i < before.Length; i++)
        {
            if (Math.Abs(before[i] - after[i]) > 1e-12)
            {
                return true;
            }
        }

        return false;
    }
}
