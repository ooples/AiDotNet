using System.Collections.Generic;
using System.Linq;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

public class ReplayBuffersIntegrationTests
{
    [Fact]
    public void Experience_DefaultPriority_IsOneAndMutable()
    {
        var state = new Vector<double>(1);
        var action = new Vector<double>(1);
        var nextState = new Vector<double>(1);
        var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);

        Assert.Equal(1.0, experience.Priority, precision: 10);

        experience.Priority = 0.25;

        Assert.Equal(0.25, experience.Priority, precision: 10);
    }

    [Fact]
    public void UniformReplayBuffer_SampleWithIndices_ReturnsUniqueIndices()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 5, seed: 17);

        for (int i = 0; i < 5; i++)
        {
            buffer.Add(CreateExperience(i));
        }

        var (experiences, indices) = buffer.SampleWithIndices(batchSize: 3);

        Assert.Equal(3, experiences.Count);
        Assert.Equal(3, indices.Count);
        Assert.Equal(3, indices.Distinct().Count());
        Assert.All(indices, index => Assert.InRange(index, 0, 4));
    }

    [Fact]
    public void PrioritizedReplayBuffer_Sample_ReturnsWeightsAndAllowsPriorityUpdates()
    {
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 5);

        for (int i = 0; i < 3; i++)
        {
            buffer.Add(
                new Vector<double>(new double[] { i }),
                new Vector<double>(new double[] { 1.0, 0.0 }),
                reward: 1.0,
                nextState: new Vector<double>(new double[] { i + 1 }),
                done: false);
        }

        var (batch, indices, weights) = buffer.Sample(batchSize: 3, alpha: 0.6, beta: 0.4);

        Assert.Equal(3, batch.Count);
        Assert.Equal(3, indices.Count);
        Assert.Equal(3, weights.Count);
        Assert.All(weights, weight => Assert.InRange(weight, 0.0, 1.0));

        buffer.UpdatePriorities(indices, new List<double> { 2.0, 1.5, 1.0 }, epsilon: 1e-3);

        var (updatedBatch, updatedIndices, updatedWeights) = buffer.Sample(batchSize: 3, alpha: 0.6, beta: 0.4);

        Assert.Equal(3, updatedBatch.Count);
        Assert.Equal(3, updatedIndices.Count);
        Assert.Equal(3, updatedWeights.Count);
    }

    private static Experience<double, Vector<double>, Vector<double>> CreateExperience(int seed)
    {
        var state = new Vector<double>(new double[] { seed });
        var action = new Vector<double>(new double[] { 1.0, 0.0 });
        var nextState = new Vector<double>(new double[] { seed + 1 });
        return new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
    }
}
