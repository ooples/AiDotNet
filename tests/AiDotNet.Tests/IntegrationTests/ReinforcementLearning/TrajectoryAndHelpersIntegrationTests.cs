using System.Collections.Generic;
using AiDotNet.ReinforcementLearning.Agents.A3C;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.ReinforcementLearning.Agents.MuZero;
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

public class TrajectoryAndHelpersIntegrationTests
{
    [Fact]
    public void Trajectory_AddStep_IncrementsLength()
    {
        var trajectory = new Trajectory<double>();
        var state = new Vector<double>(1);
        var action = new Vector<double>(1);

        trajectory.AddStep(state, action, reward: 1.0, value: 0.5, logProb: -0.1, done: false);

        Assert.Equal(1, trajectory.Length);
        Assert.Single(trajectory.States);
        Assert.Single(trajectory.Actions);
        Assert.Single(trajectory.Rewards);
        Assert.Single(trajectory.Values);
        Assert.Single(trajectory.LogProbs);
        Assert.Single(trajectory.Dones);
    }

    [Fact]
    public void Trajectory_Clear_ResetsCollections()
    {
        var trajectory = new Trajectory<double>();
        trajectory.AddStep(new Vector<double>(1), new Vector<double>(1), 1.0, 0.5, -0.1, false);
        trajectory.Advantages = new List<double> { 1.0 };
        trajectory.Returns = new List<double> { 1.0 };

        trajectory.Clear();

        Assert.Equal(0, trajectory.Length);
        Assert.Empty(trajectory.States);
        Assert.Empty(trajectory.Actions);
        Assert.Empty(trajectory.Rewards);
        Assert.Empty(trajectory.Values);
        Assert.Empty(trajectory.LogProbs);
        Assert.Empty(trajectory.Dones);
        Assert.Null(trajectory.Advantages);
        Assert.Null(trajectory.Returns);
    }

    [Fact]
    public void SequenceContext_LengthReflectsStoredStates()
    {
        var context = new SequenceContext<double>();
        context.States.Add(new Vector<double>(1));
        context.Actions.Add(new Vector<double>(1));
        context.ReturnsToGo.Add(1.0);

        Assert.Equal(1, context.Length);
        Assert.Single(context.States);
        Assert.Single(context.Actions);
        Assert.Single(context.ReturnsToGo);
    }

    [Fact]
    public void WorkerNetworks_TrajectoryIsInitialized()
    {
        var worker = new WorkerNetworks<double>();

        Assert.NotNull(worker.Trajectory);
        Assert.Empty(worker.Trajectory);

        worker.Trajectory.Add((new Vector<double>(1), new Vector<double>(1), 1.0, false, 0.0));

        Assert.Single(worker.Trajectory);
    }

    [Fact]
    public void MctsNode_InitializesCollections()
    {
        var node = new MCTSNode<double>
        {
            HiddenState = new Vector<double>(1),
            Value = 0.0,
            TotalVisits = 0
        };

        node.Children[0] = new MCTSNode<double> { HiddenState = new Vector<double>(1), Value = 0.0 };
        node.VisitCounts[0] = 1;
        node.QValues[0] = 0.25;
        node.Rewards[0] = 1.0;

        Assert.Single(node.Children);
        Assert.Single(node.VisitCounts);
        Assert.Single(node.QValues);
        Assert.Single(node.Rewards);
    }
}
