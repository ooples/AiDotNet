using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class ReplayBuffersIntegrationTests
{
    [Fact]
    public void Experience_DefaultPriority_IsOne()
    {
        var state = CreateVector(1, 0.1);
        var action = CreateVector(1, 1.0);
        var experience = new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, state, false);

        Assert.Equal(1.0, experience.Priority);
        Assert.Same(state, experience.State);
        Assert.Same(action, experience.Action);
    }

    [Fact]
    public void UniformReplayBuffer_AddSampleAndClear_Works()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 3, seed: 11);

        buffer.Add(CreateExperience(0.0));
        buffer.Add(CreateExperience(1.0));
        buffer.Add(CreateExperience(2.0));

        Assert.Equal(3, buffer.Count);
        Assert.True(buffer.CanSample(2));

        var sample = buffer.Sample(2);
        Assert.Equal(2, sample.Count);

        var (batch, indices) = buffer.SampleWithIndices(3);
        Assert.Equal(3, batch.Count);
        Assert.Equal(3, indices.Count);

        var unique = new HashSet<int>(indices);
        Assert.Equal(3, unique.Count);
        foreach (var index in indices)
        {
            Assert.InRange(index, 0, 2);
        }

        buffer.Clear();
        Assert.Equal(0, buffer.Count);
        Assert.False(buffer.CanSample(1));
    }

    [Fact]
    public void UniformReplayBuffer_OverwritesOldest_WhenCapacityExceeded()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, Vector<double>>(capacity: 2, seed: 7);

        buffer.Add(CreateExperience(0.0));
        buffer.Add(CreateExperience(1.0));
        buffer.Add(CreateExperience(2.0));

        var sample = buffer.Sample(2);
        var values = new HashSet<double>();
        foreach (var experience in sample)
        {
            values.Add(experience.State[0]);
        }

        Assert.DoesNotContain(0.0, values);
        Assert.Contains(1.0, values);
        Assert.Contains(2.0, values);
    }

    [Fact]
    public void PrioritizedReplayBuffer_AddSampleUpdate_Works()
    {
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 4);

        buffer.Add(CreateVector(2, 0.1), CreateVector(1, 1.0), 1.0, CreateVector(2, 0.2), false);
        buffer.Add(CreateVector(2, 0.2), CreateVector(1, 0.0), 0.5, CreateVector(2, 0.3), true);

        Assert.Equal(2, buffer.Count);

        var (batch, indices, weights) = buffer.Sample(batchSize: 2, alpha: 0.6, beta: 0.4);

        Assert.Equal(batch.Count, indices.Count);
        Assert.Equal(batch.Count, weights.Count);

        for (int i = 0; i < indices.Count; i++)
        {
            Assert.InRange(indices[i], 0, buffer.Count - 1);
            Assert.InRange(weights[i], 0.0, 1.000001);
        }

        buffer.UpdatePriorities(indices, new List<double> { 0.5, 1.5 }, epsilon: 1e-6);

        var (batch2, indices2, weights2) = buffer.Sample(batchSize: 2, alpha: 0.6, beta: 0.4);

        Assert.Equal(batch2.Count, indices2.Count);
        Assert.Equal(batch2.Count, weights2.Count);
    }

    private static Experience<double, Vector<double>, Vector<double>> CreateExperience(double stateValue)
    {
        var state = CreateVector(1, stateValue);
        var action = CreateVector(1, 1.0);
        var nextState = CreateVector(1, stateValue + 0.1);
        return new Experience<double, Vector<double>, Vector<double>>(state, action, 1.0, nextState, false);
    }

    private static Vector<double> CreateVector(int size, double start)
    {
        var vector = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = start + i * 0.01;
        }
        return vector;
    }
}
