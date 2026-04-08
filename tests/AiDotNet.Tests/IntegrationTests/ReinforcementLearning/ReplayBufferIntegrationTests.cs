using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.ReinforcementLearning.Policies.Exploration;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// Deep integration tests for RL replay buffers and exploration strategies.
/// Verifies circular buffer semantics, prioritized sampling math, epsilon decay,
/// Boltzmann softmax correctness, and Ornstein-Uhlenbeck process properties.
/// </summary>
public class ReplayBufferIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region UniformReplayBuffer - Circular Buffer Semantics

    [Fact]
    public void UniformBuffer_Add_UnderCapacity_AllStored()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 5, seed: 42);

        for (int i = 0; i < 3; i++)
        {
            buffer.Add(new Experience<double, Vector<double>, int>(
                new Vector<double>(new double[] { i }), i, (double)i, new Vector<double>(new double[] { i + 1 }), false));
        }

        Assert.Equal(3, buffer.Count);
        Assert.Equal(5, buffer.Capacity);
    }

    [Fact]
    public void UniformBuffer_Add_OverCapacity_CircularOverwrite()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 3, seed: 42);

        // Add 5 items to a buffer of capacity 3
        for (int i = 0; i < 5; i++)
        {
            buffer.Add(new Experience<double, Vector<double>, int>(
                new Vector<double>(new double[] { i * 10 }), i, (double)i, new Vector<double>(new double[] { i + 1 }), false));
        }

        // Count should be capped at capacity
        Assert.Equal(3, buffer.Count);
    }

    [Fact]
    public void UniformBuffer_Sample_CorrectBatchSize()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 10, seed: 42);

        for (int i = 0; i < 10; i++)
        {
            buffer.Add(new Experience<double, Vector<double>, int>(
                new Vector<double>(new double[] { i }), i, (double)i, new Vector<double>(new double[] { i + 1 }), false));
        }

        var batch = buffer.Sample(5);
        Assert.Equal(5, batch.Count);
    }

    [Fact]
    public void UniformBuffer_Sample_WithoutReplacement()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 100, seed: 42);

        for (int i = 0; i < 100; i++)
        {
            buffer.Add(new Experience<double, Vector<double>, int>(
                new Vector<double>(new double[] { i }), i, (double)i, new Vector<double>(new double[] { i + 1 }), false));
        }

        var (experiences, indices) = buffer.SampleWithIndices(20);

        // All indices should be unique (sampling without replacement)
        Assert.Equal(indices.Count, indices.Distinct().Count());
    }

    [Fact]
    public void UniformBuffer_CanSample_InsufficientData_ReturnsFalse()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 10, seed: 42);

        buffer.Add(new Experience<double, Vector<double>, int>(
            new Vector<double>(new double[] { 1 }), 0, 1.0, new Vector<double>(new double[] { 2 }), false));

        Assert.False(buffer.CanSample(5));
        Assert.True(buffer.CanSample(1));
    }

    [Fact]
    public void UniformBuffer_Sample_InsufficientData_Throws()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 10, seed: 42);

        Assert.Throws<InvalidOperationException>(() => buffer.Sample(5));
    }

    [Fact]
    public void UniformBuffer_Clear_ResetsBuffer()
    {
        var buffer = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 10, seed: 42);

        for (int i = 0; i < 5; i++)
        {
            buffer.Add(new Experience<double, Vector<double>, int>(
                new Vector<double>(new double[] { i }), i, (double)i, new Vector<double>(new double[] { i + 1 }), false));
        }

        buffer.Clear();
        Assert.Equal(0, buffer.Count);
        Assert.False(buffer.CanSample(1));
    }

    [Fact]
    public void UniformBuffer_InvalidCapacity_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new UniformReplayBuffer<double, Vector<double>, int>(capacity: 0));
        Assert.Throws<ArgumentException>(() =>
            new UniformReplayBuffer<double, Vector<double>, int>(capacity: -1));
    }

    [Fact]
    public void UniformBuffer_Deterministic_SameSeedSameResults()
    {
        var buffer1 = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 100, seed: 42);
        var buffer2 = new UniformReplayBuffer<double, Vector<double>, int>(capacity: 100, seed: 42);

        for (int i = 0; i < 50; i++)
        {
            var exp = new Experience<double, Vector<double>, int>(
                new Vector<double>(new double[] { i }), i, (double)i, new Vector<double>(new double[] { i + 1 }), false);
            buffer1.Add(exp);
            buffer2.Add(exp);
        }

        var batch1 = buffer1.Sample(10);
        var batch2 = buffer2.Sample(10);

        // Same seed should produce same samples
        for (int i = 0; i < batch1.Count; i++)
        {
            Assert.Equal(batch1[i].Reward, batch2[i].Reward);
        }
    }

    #endregion

    #region PrioritizedReplayBuffer - Priority Sampling Math

    [Fact]
    public void PrioritizedBuffer_Add_IncreasesCount()
    {
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 10);

        buffer.Add(
            new Vector<double>(new double[] { 1, 2 }),
            new Vector<double>(new double[] { 0 }),
            1.0,
            new Vector<double>(new double[] { 3, 4 }),
            false);

        Assert.Equal(1, buffer.Count);
    }

    [Fact]
    public void PrioritizedBuffer_CircularOverwrite()
    {
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 3);

        for (int i = 0; i < 5; i++)
        {
            buffer.Add(
                new Vector<double>(new double[] { i }),
                new Vector<double>(new double[] { 0 }),
                (double)i,
                new Vector<double>(new double[] { i + 1 }),
                false);
        }

        Assert.Equal(3, buffer.Count);
    }

    [Fact]
    public void PrioritizedBuffer_Sample_CorrectBatchSize()
    {
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 20);

        for (int i = 0; i < 20; i++)
        {
            buffer.Add(
                new Vector<double>(new double[] { i }),
                new Vector<double>(new double[] { 0 }),
                (double)i,
                new Vector<double>(new double[] { i + 1 }),
                false);
        }

        var (batch, indices, weights) = buffer.Sample(batchSize: 5, alpha: 0.6, beta: 0.4);

        Assert.Equal(5, batch.Count);
        Assert.Equal(5, indices.Count);
        Assert.Equal(5, weights.Count);
    }

    [Fact]
    public void PrioritizedBuffer_Weights_MaxIsOne()
    {
        // Importance sampling weights should be normalized so max weight = 1
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 20);

        for (int i = 0; i < 20; i++)
        {
            buffer.Add(
                new Vector<double>(new double[] { i }),
                new Vector<double>(new double[] { 0 }),
                (double)i,
                new Vector<double>(new double[] { i + 1 }),
                false);
        }

        var (_, _, weights) = buffer.Sample(batchSize: 10, alpha: 0.6, beta: 1.0);

        // All weights should be <= 1.0 (normalized by max weight)
        foreach (var w in weights)
        {
            Assert.True(w <= 1.0 + 1e-10, $"Weight {w} exceeds 1.0");
            Assert.True(w > 0, $"Weight {w} is non-positive");
        }
    }

    [Fact]
    public void PrioritizedBuffer_UpdatePriorities_AffectsSampling()
    {
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 10);

        for (int i = 0; i < 10; i++)
        {
            buffer.Add(
                new Vector<double>(new double[] { i }),
                new Vector<double>(new double[] { 0 }),
                (double)i,
                new Vector<double>(new double[] { i + 1 }),
                false);
        }

        // Give very high priority to index 5
        buffer.UpdatePriorities(
            new List<int> { 5 },
            new List<double> { 1000.0 },
            epsilon: 0.01);

        // Sample many times and check if index 5 is sampled disproportionately
        int sampleCount = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var (_, indices, _) = buffer.Sample(batchSize: 3, alpha: 1.0, beta: 1.0);
            if (indices.Contains(5))
            {
                sampleCount++;
            }
        }

        // With very high priority, index 5 should appear in most samples
        Assert.True(sampleCount > 50,
            $"High-priority index 5 only appeared in {sampleCount}/100 samples");
    }

    [Fact]
    public void PrioritizedBuffer_AlphaZero_UniformSampling()
    {
        // Alpha=0 means all priorities are raised to power 0 = 1.0 → uniform
        var buffer = new PrioritizedReplayBuffer<double>(capacity: 10);

        for (int i = 0; i < 10; i++)
        {
            buffer.Add(
                new Vector<double>(new double[] { i }),
                new Vector<double>(new double[] { 0 }),
                (double)i,
                new Vector<double>(new double[] { i + 1 }),
                false);
        }

        // Set varying priorities
        buffer.UpdatePriorities(
            new List<int> { 0, 1, 2 },
            new List<double> { 100.0, 0.01, 50.0 },
            epsilon: 0.0);

        // With alpha=0, all priorities become 1 → uniform sampling
        // All weights should be equal
        var (_, _, weights) = buffer.Sample(batchSize: 5, alpha: 0.0, beta: 1.0);

        // All weights should be approximately equal
        for (int i = 1; i < weights.Count; i++)
        {
            Assert.Equal(weights[0], weights[i], Tolerance);
        }
    }

    #endregion

    #region EpsilonGreedyExploration - Decay Math

    [Fact]
    public void EpsilonGreedy_InitialEpsilon_IsStart()
    {
        var strategy = new EpsilonGreedyExploration<double>(
            epsilonStart: 1.0, epsilonEnd: 0.01, epsilonDecay: 0.995);

        Assert.Equal(1.0, strategy.CurrentEpsilon, Tolerance);
    }

    [Fact]
    public void EpsilonGreedy_DecayFormula_GoldenReference()
    {
        // After n updates: epsilon = max(epsilonEnd, epsilonStart * decay^n)
        double start = 1.0, end = 0.01, decay = 0.99;
        var strategy = new EpsilonGreedyExploration<double>(start, end, decay);

        // After 10 updates: 1.0 * 0.99^10 = 0.9043820...
        for (int i = 0; i < 10; i++) strategy.Update();
        double expected10 = start * Math.Pow(decay, 10);
        Assert.Equal(expected10, strategy.CurrentEpsilon, Tolerance);

        // After 100 more updates (110 total): 1.0 * 0.99^110 = 0.3310...
        for (int i = 0; i < 100; i++) strategy.Update();
        double expected110 = start * Math.Pow(decay, 110);
        Assert.Equal(expected110, strategy.CurrentEpsilon, Tolerance);
    }

    [Fact]
    public void EpsilonGreedy_DecayNeverBelowEnd()
    {
        var strategy = new EpsilonGreedyExploration<double>(
            epsilonStart: 1.0, epsilonEnd: 0.1, epsilonDecay: 0.5);

        // 1.0 * 0.5 = 0.5
        // 0.5 * 0.5 = 0.25
        // 0.25 * 0.5 = 0.125
        // 0.125 * 0.5 = 0.0625 → clamped to 0.1
        for (int i = 0; i < 100; i++) strategy.Update();

        Assert.Equal(0.1, strategy.CurrentEpsilon, Tolerance);
    }

    [Fact]
    public void EpsilonGreedy_Reset_RestoresStart()
    {
        var strategy = new EpsilonGreedyExploration<double>(
            epsilonStart: 1.0, epsilonEnd: 0.01, epsilonDecay: 0.5);

        for (int i = 0; i < 10; i++) strategy.Update();
        Assert.NotEqual(1.0, strategy.CurrentEpsilon);

        strategy.Reset();
        Assert.Equal(1.0, strategy.CurrentEpsilon, Tolerance);
    }

    [Fact]
    public void EpsilonGreedy_HighEpsilon_AlwaysExplores()
    {
        // With epsilon=1.0, should always take random action
        var strategy = new EpsilonGreedyExploration<double>(
            epsilonStart: 1.0, epsilonEnd: 1.0, epsilonDecay: 1.0);

        var state = new Vector<double>(new double[] { 0.5 });
        var policyAction = new Vector<double>(new double[] { 1, 0, 0 }); // one-hot action 0
        var random = new Random(42);

        int differentFromPolicy = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            var action = strategy.GetExplorationAction(state, policyAction, 3, random);
            // Check if action differs from policy
            bool same = true;
            for (int i = 0; i < 3; i++)
            {
                if (Math.Abs(Convert.ToDouble(action[i]) - Convert.ToDouble(policyAction[i])) > 0.01)
                {
                    same = false;
                    break;
                }
            }
            if (!same) differentFromPolicy++;
        }

        // With epsilon=1.0, most actions should be random (not matching policy)
        // Policy action is [1,0,0], random picks 1 of 3 → 2/3 chance of being different
        Assert.True(differentFromPolicy > 50,
            $"Only {differentFromPolicy}/100 actions differed from policy with epsilon=1.0");
    }

    [Fact]
    public void EpsilonGreedy_ZeroEpsilon_AlwaysGreedy()
    {
        // With epsilon=0.0, should always take greedy (policy) action
        var strategy = new EpsilonGreedyExploration<double>(
            epsilonStart: 0.0, epsilonEnd: 0.0, epsilonDecay: 1.0);

        var state = new Vector<double>(new double[] { 0.5 });
        var policyAction = new Vector<double>(new double[] { 0, 1, 0 }); // one-hot action 1
        var random = new Random(42);

        for (int trial = 0; trial < 50; trial++)
        {
            var action = strategy.GetExplorationAction(state, policyAction, 3, random);
            // Should always match policy
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(Convert.ToDouble(policyAction[i]), Convert.ToDouble(action[i]), Tolerance);
            }
        }
    }

    #endregion

    #region BoltzmannExploration - Temperature and Softmax Math

    [Fact]
    public void Boltzmann_InitialTemperature_IsStart()
    {
        var strategy = new BoltzmannExploration<double>(
            temperatureStart: 2.0, temperatureEnd: 0.1, temperatureDecay: 0.99);

        Assert.Equal(2.0, strategy.CurrentTemperature, Tolerance);
    }

    [Fact]
    public void Boltzmann_TemperatureDecay_GoldenReference()
    {
        double start = 1.0, end = 0.01, decay = 0.9;
        var strategy = new BoltzmannExploration<double>(start, end, decay);

        // After 5 updates: 1.0 * 0.9^5 = 0.59049
        for (int i = 0; i < 5; i++) strategy.Update();
        Assert.Equal(start * Math.Pow(decay, 5), strategy.CurrentTemperature, Tolerance);
    }

    [Fact]
    public void Boltzmann_TemperatureNeverBelowEnd()
    {
        var strategy = new BoltzmannExploration<double>(
            temperatureStart: 1.0, temperatureEnd: 0.5, temperatureDecay: 0.1);

        for (int i = 0; i < 100; i++) strategy.Update();

        Assert.Equal(0.5, strategy.CurrentTemperature, Tolerance);
    }

    [Fact]
    public void Boltzmann_Reset_RestoresStart()
    {
        var strategy = new BoltzmannExploration<double>(
            temperatureStart: 5.0, temperatureEnd: 0.01, temperatureDecay: 0.5);

        for (int i = 0; i < 10; i++) strategy.Update();
        Assert.NotEqual(5.0, strategy.CurrentTemperature);

        strategy.Reset();
        Assert.Equal(5.0, strategy.CurrentTemperature, Tolerance);
    }

    [Fact]
    public void Boltzmann_HighTemperature_MoreUniform()
    {
        // High temperature → softmax approaches uniform distribution
        var highTempStrategy = new BoltzmannExploration<double>(
            temperatureStart: 100.0, temperatureEnd: 100.0, temperatureDecay: 1.0);

        var state = new Vector<double>(new double[] { 0.5 });
        // One-hot action: [1, 0, 0] - but high temp should override this
        var policyAction = new Vector<double>(new double[] { 1, 0, 0 });
        var random = new Random(42);

        // Count how often each action is selected
        int[] actionCounts = new int[3];
        for (int trial = 0; trial < 300; trial++)
        {
            var action = highTempStrategy.GetExplorationAction(state, policyAction, 3, random);
            for (int i = 0; i < 3; i++)
            {
                if (Convert.ToDouble(action[i]) > 0.5)
                {
                    actionCounts[i]++;
                    break;
                }
            }
        }

        // With very high temperature, distribution should be roughly uniform
        // Each action should appear roughly 100 times (100 ± ~30)
        for (int i = 0; i < 3; i++)
        {
            Assert.True(actionCounts[i] > 50,
                $"Action {i} only appeared {actionCounts[i]}/300 times with high temperature");
        }
    }

    #endregion

    #region OrnsteinUhlenbeck Noise - Process Properties

    [Fact]
    public void OU_Reset_ZerosState()
    {
        var ou = new OrnsteinUhlenbeckNoise<double>(actionSize: 3);

        // Generate some noise
        var state = new Vector<double>(new double[] { 0, 0, 0 });
        var action = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var random = new Random(42);

        for (int i = 0; i < 10; i++)
        {
            ou.GetExplorationAction(state, action, 3, random);
        }

        // Reset should zero the noise state
        ou.Reset();

        // After reset, the first noise application should start from zero state
        // This means the noise magnitude should be small (just one step from zero)
        var afterReset = ou.GetExplorationAction(state, action, 3, random);
        for (int i = 0; i < 3; i++)
        {
            double diff = Math.Abs(Convert.ToDouble(afterReset[i]) - Convert.ToDouble(action[i]));
            // First step from zero: noise should be small (σ * √dt ≈ 0.2 * 0.1 = 0.02)
            Assert.True(diff < 0.5,
                $"After reset, noise diff {diff} is too large for dimension {i}");
        }
    }

    [Fact]
    public void OU_MeanReversion_GoldenReference()
    {
        // The OU process should revert toward the mean (mu=0 by default)
        // If starting far from mean, subsequent values should trend toward mean
        var ou = new OrnsteinUhlenbeckNoise<double>(
            actionSize: 1, theta: 0.5, sigma: 0.01, mu: 0.0, dt: 0.1);

        var state = new Vector<double>(new double[] { 0 });
        // Start with a large action far from mean
        var action = new Vector<double>(new double[] { 0.0 });
        var random = new Random(42);

        // Run many steps and collect the noise values
        double sum = 0;
        int steps = 1000;
        for (int i = 0; i < steps; i++)
        {
            var result = ou.GetExplorationAction(state, action, 1, random);
            sum += Convert.ToDouble(result[0]);
        }

        // With mu=0 and high theta, the average should be close to 0
        double avg = sum / steps;
        Assert.True(Math.Abs(avg) < 0.5,
            $"OU average {avg} is too far from mu=0");
    }

    [Fact]
    public void OU_TemporalCorrelation_NotIID()
    {
        // OU noise should be temporally correlated (unlike Gaussian noise)
        // Adjacent samples should be more similar than random
        var ou = new OrnsteinUhlenbeckNoise<double>(
            actionSize: 1, theta: 0.15, sigma: 0.2, mu: 0.0, dt: 0.01);

        var state = new Vector<double>(new double[] { 0 });
        var action = new Vector<double>(new double[] { 0.0 });
        var random = new Random(42);

        var values = new List<double>();
        for (int i = 0; i < 100; i++)
        {
            var result = ou.GetExplorationAction(state, action, 1, random);
            values.Add(Convert.ToDouble(result[0]));
        }

        // Compute autocorrelation at lag 1
        double mean = values.Average();
        double variance = values.Select(v => (v - mean) * (v - mean)).Average();
        double autocovariance = 0;
        for (int i = 0; i < values.Count - 1; i++)
        {
            autocovariance += (values[i] - mean) * (values[i + 1] - mean);
        }
        autocovariance /= (values.Count - 1);

        double autocorrelation = variance > 0 ? autocovariance / variance : 0;

        // OU process should have positive autocorrelation (values are similar to neighbors)
        Assert.True(autocorrelation > 0.1,
            $"OU autocorrelation {autocorrelation} is too low - noise should be temporally correlated");
    }

    #endregion

    #region Experience Record Tests

    [Fact]
    public void Experience_RecordCreation_PropertiesCorrect()
    {
        var state = new Vector<double>(new double[] { 1, 2, 3 });
        var action = new Vector<double>(new double[] { 0.5 });
        var nextState = new Vector<double>(new double[] { 4, 5, 6 });

        var exp = new Experience<double, Vector<double>, Vector<double>>(
            state, action, 1.5, nextState, false);

        Assert.Equal(1.5, exp.Reward);
        Assert.False(exp.Done);
        Assert.Equal(1.0, exp.Priority); // Default priority
    }

    [Fact]
    public void Experience_Priority_Settable()
    {
        var exp = new Experience<double, Vector<double>, int>(
            new Vector<double>(new double[] { 1 }), 0, 1.0, new Vector<double>(new double[] { 2 }), false);

        exp.Priority = 5.0;
        Assert.Equal(5.0, exp.Priority);
    }

    [Fact]
    public void Experience_SimplifiedRecord_WorksCorrectly()
    {
        var exp = new Experience<double>(
            new Vector<double>(new double[] { 1, 2 }),
            new Vector<double>(new double[] { 0.5 }),
            1.0,
            new Vector<double>(new double[] { 3, 4 }),
            true);

        Assert.True(exp.Done);
        Assert.Equal(1.0, exp.Reward);
    }

    #endregion
}
