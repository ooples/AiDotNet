using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents.A3C;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;
using AiDotNet.ReinforcementLearning.Agents.MuZero;
using AiDotNet.ReinforcementLearning.Common;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class TrajectoryAndHelpersIntegrationTests
{
    [Fact]
    public void Trajectory_AddStepAndClear_TracksState()
    {
        var trajectory = new Trajectory<double>();
        var state = CreateVector(2, 0.1);
        var action = CreateVector(1, 1.0);

        trajectory.AddStep(state, action, 1.0, value: 0.5, logProb: 0.1, done: false);

        Assert.Equal(1, trajectory.Length);
        Assert.Single(trajectory.States);
        Assert.Single(trajectory.Actions);

        trajectory.Advantages = new List<double> { 0.1 };
        trajectory.Returns = new List<double> { 0.2 };

        trajectory.Clear();

        Assert.Equal(0, trajectory.Length);
        Assert.Empty(trajectory.States);
        Assert.Null(trajectory.Advantages);
        Assert.Null(trajectory.Returns);
    }

    [Fact]
    public void SequenceContext_LengthReflectsStates()
    {
        var context = new SequenceContext<double>();

        context.States.Add(CreateVector(2, 0.1));
        context.Actions.Add(CreateVector(1, 1.0));
        context.ReturnsToGo.Add(1.5);

        Assert.Equal(1, context.Length);
    }

    [Fact]
    public void MCTSNode_DefaultsAndAssignments_Work()
    {
        var node = new MCTSNode<double>
        {
            HiddenState = CreateVector(2, 0.2),
            Value = 0.5
        };

        node.Children[0] = new MCTSNode<double> { HiddenState = CreateVector(2, 0.3) };
        node.VisitCounts[0] = 1;
        node.QValues[0] = 0.25;
        node.Rewards[0] = 0.1;
        node.TotalVisits = 1;

        Assert.Equal(1, node.Children.Count);
        Assert.Equal(1, node.VisitCounts[0]);
        Assert.Equal(1, node.TotalVisits);
    }

    [Fact]
    public void TransitionData_DefaultsAndOverrides_Work()
    {
        var data = new TransitionData<double>();

        Assert.Equal(0.0, data.Reward);
        Assert.Equal(0.0, data.Probability);
        Assert.Equal(string.Empty, data.NextState);

        data.NextState = "s1";
        data.Reward = 1.0;
        data.Probability = 0.5;

        Assert.Equal("s1", data.NextState);
        Assert.Equal(1.0, data.Reward);
        Assert.Equal(0.5, data.Probability);
    }

    [Fact]
    public void WorkerNetworks_DefaultsAndAssignments_Work()
    {
        var worker = new WorkerNetworks<double>();

        Assert.NotNull(worker.Trajectory);
        Assert.Empty(worker.Trajectory);

        worker.PolicyNetwork = CreateNetwork(2, 1);
        worker.ValueNetwork = CreateNetwork(2, 1);

        Assert.NotNull(worker.PolicyNetwork);
        Assert.NotNull(worker.ValueNetwork);
    }

    private static Vector<double> CreateVector(int size, double start)
    {
        var vector = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = start + i * 0.1;
        }
        return vector;
    }

    private static NeuralNetwork<double> CreateNetwork(int inputSize, int outputSize)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.ReinforcementLearning,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize);

        return new NeuralNetwork<double>(architecture);
    }
}
