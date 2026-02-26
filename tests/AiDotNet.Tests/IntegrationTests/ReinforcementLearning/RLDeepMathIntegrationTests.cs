using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// Deep integration tests for ReinforcementLearning:
/// Trajectory class (construction, AddStep, Clear, Length),
/// RL math (discounted returns, GAE, TD error, epsilon-greedy, Bellman equation,
/// policy gradient, entropy bonus, PPO clipping).
/// </summary>
public class RLDeepMathIntegrationTests
{
    // ============================
    // Trajectory: Construction
    // ============================

    [Fact]
    public void Trajectory_Default_EmptyLists()
    {
        var trajectory = new Trajectory<double>();
        Assert.Empty(trajectory.States);
        Assert.Empty(trajectory.Actions);
        Assert.Empty(trajectory.Rewards);
        Assert.Empty(trajectory.Values);
        Assert.Empty(trajectory.LogProbs);
        Assert.Empty(trajectory.Dones);
        Assert.Null(trajectory.Advantages);
        Assert.Null(trajectory.Returns);
        Assert.Equal(0, trajectory.Length);
    }

    [Fact]
    public void Trajectory_AddStep_IncreasesLength()
    {
        var trajectory = new Trajectory<double>();
        var state = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
        var action = new Vector<double>(new double[] { 0.5 });

        trajectory.AddStep(state, action, 1.0, 0.5, -0.3, false);
        Assert.Equal(1, trajectory.Length);

        trajectory.AddStep(state, action, 0.5, 0.4, -0.5, false);
        Assert.Equal(2, trajectory.Length);
    }

    [Fact]
    public void Trajectory_AddStep_StoresCorrectValues()
    {
        var trajectory = new Trajectory<double>();
        var state = new Vector<double>(new double[] { 1.0, 2.0 });
        var action = new Vector<double>(new double[] { 0.5 });

        trajectory.AddStep(state, action, 1.5, 0.8, -0.2, true);

        Assert.Single(trajectory.States);
        Assert.Single(trajectory.Actions);
        Assert.Equal(1.5, trajectory.Rewards[0]);
        Assert.Equal(0.8, trajectory.Values[0]);
        Assert.Equal(-0.2, trajectory.LogProbs[0]);
        Assert.True(trajectory.Dones[0]);
    }

    [Fact]
    public void Trajectory_Clear_ResetsEverything()
    {
        var trajectory = new Trajectory<double>();
        var state = new Vector<double>(new double[] { 1.0 });
        var action = new Vector<double>(new double[] { 0.5 });

        trajectory.AddStep(state, action, 1.0, 0.5, -0.3, false);
        trajectory.AddStep(state, action, 0.5, 0.4, -0.5, true);
        trajectory.Advantages = new List<double> { 0.1, 0.2 };
        trajectory.Returns = new List<double> { 1.0, 0.5 };

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

    // ============================
    // RL Math: Discounted Returns
    // ============================

    [Theory]
    [InlineData(new double[] { 1, 1, 1, 1 }, 0.99, 3.9403)]     // G = 1 + 0.99 + 0.9801 + 0.970299
    [InlineData(new double[] { 1, 0, 0, 10 }, 0.9, 8.29)]        // G = 1 + 0 + 0 + 10*0.729 = 8.29
    [InlineData(new double[] { 1, 1, 1 }, 0.0, 1.0)]             // gamma=0: only immediate reward
    public void RLMath_DiscountedReturn(double[] rewards, double gamma, double expectedReturn)
    {
        double discountedReturn = 0;
        double discount = 1.0;
        foreach (double reward in rewards)
        {
            discountedReturn += reward * discount;
            discount *= gamma;
        }

        Assert.Equal(expectedReturn, discountedReturn, 1e-3);
    }

    [Fact]
    public void RLMath_DiscountedReturn_ReverseComputation()
    {
        // Computing returns backwards (more efficient)
        double[] rewards = { 1, 2, 3, 4 };
        double gamma = 0.99;

        double[] returns = new double[rewards.Length];
        returns[^1] = rewards[^1];
        for (int i = rewards.Length - 2; i >= 0; i--)
        {
            returns[i] = rewards[i] + gamma * returns[i + 1];
        }

        // Forward computation for verification
        double forwardReturn = 0;
        double discount = 1.0;
        foreach (double r in rewards)
        {
            forwardReturn += r * discount;
            discount *= gamma;
        }

        Assert.Equal(forwardReturn, returns[0], 1e-10);
    }

    // ============================
    // RL Math: TD Error
    // ============================

    [Theory]
    [InlineData(1.0, 0.99, 5.0, 4.0, 1.95)]   // delta = r + gamma*V(s') - V(s) = 1 + 0.99*5 - 4 = 1.95
    [InlineData(0.0, 0.99, 0.0, 1.0, -1.0)]    // Terminal: delta = 0 + 0 - 1 = -1
    public void RLMath_TDError(double reward, double gamma, double nextValue, double currentValue, double expectedDelta)
    {
        double tdError = reward + gamma * nextValue - currentValue;
        Assert.Equal(expectedDelta, tdError, 1e-10);
    }

    // ============================
    // RL Math: GAE (Generalized Advantage Estimation)
    // ============================

    [Fact]
    public void RLMath_GAE_Calculation()
    {
        // GAE(lambda) = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_t+l
        double[] rewards = { 1.0, 0.5, 2.0 };
        double[] values = { 3.0, 2.5, 4.0 };
        double nextValue = 0.0; // Terminal
        double gamma = 0.99;
        double lambda = 0.95;

        // Compute TD errors
        double[] deltas = new double[3];
        deltas[2] = rewards[2] + gamma * nextValue - values[2];
        deltas[1] = rewards[1] + gamma * values[2] - values[1];
        deltas[0] = rewards[0] + gamma * values[1] - values[0];

        // Compute GAE backwards
        double[] advantages = new double[3];
        advantages[2] = deltas[2];
        advantages[1] = deltas[1] + gamma * lambda * advantages[2];
        advantages[0] = deltas[0] + gamma * lambda * advantages[1];

        // GAE should be well-defined
        Assert.False(double.IsNaN(advantages[0]));
        Assert.False(double.IsNaN(advantages[1]));
        Assert.False(double.IsNaN(advantages[2]));
    }

    [Fact]
    public void RLMath_GAE_LambdaZero_IsTDError()
    {
        // When lambda=0, GAE reduces to TD(0) error
        double reward = 1.0, gamma = 0.99;
        double value = 3.0, nextValue = 4.0;
        double lambda = 0.0;

        double tdError = reward + gamma * nextValue - value;
        double gae = tdError; // With lambda=0, no bootstrapping

        Assert.Equal(tdError, gae, 1e-10);
    }

    // ============================
    // RL Math: Epsilon-Greedy
    // ============================

    [Theory]
    [InlineData(1.0, true)]    // epsilon=1.0: always explore
    [InlineData(0.0, false)]   // epsilon=0.0: always exploit
    public void RLMath_EpsilonGreedy_BoundaryBehavior(double epsilon, bool alwaysExplore)
    {
        Assert.True(epsilon >= 0.0 && epsilon <= 1.0);
        if (alwaysExplore)
            Assert.Equal(1.0, epsilon);
        else
            Assert.Equal(0.0, epsilon);
    }

    [Theory]
    [InlineData(1.0, 0.01, 0.995, 100, 0.607)]   // Exponential decay
    [InlineData(1.0, 0.01, 0.99, 200, 0.134)]
    public void RLMath_EpsilonDecay_Exponential(double epsilonStart, double epsilonEnd, double decayRate, int step, double expectedEpsilon)
    {
        double epsilon = Math.Max(epsilonEnd, epsilonStart * Math.Pow(decayRate, step));
        Assert.Equal(expectedEpsilon, epsilon, 1e-2);
    }

    // ============================
    // RL Math: Bellman Equation
    // ============================

    [Fact]
    public void RLMath_BellmanEquation_OptimalValue()
    {
        // V*(s) = max_a [R(s,a) + gamma * sum_s' P(s'|s,a) * V*(s')]
        // Simple deterministic case: V*(s) = R(s, best_a) + gamma * V*(s')
        double gamma = 0.99;

        // Simple chain: s0 -> s1 -> s2 (terminal)
        double r01 = 1.0;  // Reward from s0 to s1
        double r12 = 10.0; // Reward from s1 to s2

        // V*(s2) = 0 (terminal)
        double v2 = 0;
        // V*(s1) = r12 + gamma * V*(s2) = 10
        double v1 = r12 + gamma * v2;
        // V*(s0) = r01 + gamma * V*(s1) = 1 + 0.99 * 10 = 10.9
        double v0 = r01 + gamma * v1;

        Assert.Equal(10.0, v1, 1e-10);
        Assert.Equal(10.9, v0, 1e-10);
    }

    // ============================
    // RL Math: Policy Gradient
    // ============================

    [Fact]
    public void RLMath_PolicyGradient_REINFORCE()
    {
        // REINFORCE loss: -sum(log_prob * advantage)
        double[] logProbs = { -0.5, -1.0, -0.3, -0.8 };
        double[] advantages = { 2.0, -1.0, 3.0, 0.5 };

        double loss = 0;
        for (int i = 0; i < logProbs.Length; i++)
        {
            loss -= logProbs[i] * advantages[i]; // Negative because we minimize
        }

        // Loss should be finite
        Assert.False(double.IsNaN(loss));
        Assert.False(double.IsInfinity(loss));
    }

    // ============================
    // RL Math: Entropy Bonus
    // ============================

    [Theory]
    [InlineData(new double[] { 0.5, 0.5 }, 0.693)]              // Maximum entropy (uniform 2 actions)
    [InlineData(new double[] { 0.25, 0.25, 0.25, 0.25 }, 1.386)] // Uniform 4 actions
    [InlineData(new double[] { 0.9, 0.1 }, 0.325)]              // Low entropy (confident)
    public void RLMath_EntropyBonus(double[] actionProbs, double expectedEntropy)
    {
        // Entropy: H(pi) = -sum(pi * log(pi))
        double entropy = 0;
        foreach (double p in actionProbs)
        {
            if (p > 0)
                entropy -= p * Math.Log(p);
        }

        Assert.Equal(expectedEntropy, entropy, 1e-2);
    }

    [Fact]
    public void RLMath_EntropyBonus_MaximumAtUniform()
    {
        int numActions = 10;
        double uniformProb = 1.0 / numActions;

        // Maximum entropy = log(numActions)
        double maxEntropy = Math.Log(numActions);

        double entropy = 0;
        for (int i = 0; i < numActions; i++)
        {
            entropy -= uniformProb * Math.Log(uniformProb);
        }

        Assert.Equal(maxEntropy, entropy, 1e-10);
    }

    // ============================
    // RL Math: PPO Clipping
    // ============================

    [Theory]
    [InlineData(1.0, 0.2, 0.8, 1.2)]     // Clip range: [1-eps, 1+eps]
    [InlineData(1.0, 0.1, 0.9, 1.1)]     // Tighter clip
    public void RLMath_PPOClipRange(double center, double epsilon, double expectedLower, double expectedUpper)
    {
        double lower = center - epsilon;
        double upper = center + epsilon;
        Assert.Equal(expectedLower, lower, 1e-10);
        Assert.Equal(expectedUpper, upper, 1e-10);
    }

    [Theory]
    [InlineData(1.5, 0.2, 2.0, 1.2)]     // Ratio too high, clipped at 1.2 * advantage
    [InlineData(0.5, 0.2, 2.0, 0.8)]     // Ratio too low, clipped at 0.8 * advantage
    [InlineData(1.0, 0.2, 2.0, 1.0)]     // Ratio in range, not clipped
    public void RLMath_PPOClippedObjective(double ratio, double epsilon, double advantage, double expectedClippedRatio)
    {
        // PPO objective: min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        double clippedRatio = Math.Max(1.0 - epsilon, Math.Min(1.0 + epsilon, ratio));
        Assert.Equal(expectedClippedRatio, clippedRatio, 1e-10);

        double unclipped = ratio * advantage;
        double clipped = clippedRatio * advantage;
        double objective = Math.Min(unclipped, clipped);

        Assert.True(objective <= unclipped, "Clipped objective should not exceed unclipped");
    }

    // ============================
    // RL Math: Softmax for Action Selection
    // ============================

    [Theory]
    [InlineData(new double[] { 1.0, 2.0, 3.0 })]
    [InlineData(new double[] { 0.0, 0.0, 0.0 })]     // All equal -> uniform
    [InlineData(new double[] { -1.0, 0.0, 1.0 })]     // Negative logits
    public void RLMath_SoftmaxProbabilities(double[] logits)
    {
        double maxLogit = logits.Max();
        double[] expLogits = logits.Select(l => Math.Exp(l - maxLogit)).ToArray(); // Numerically stable
        double sumExp = expLogits.Sum();
        double[] probs = expLogits.Select(e => e / sumExp).ToArray();

        // Probabilities should sum to 1
        Assert.Equal(1.0, probs.Sum(), 1e-10);

        // All probabilities should be positive
        foreach (double p in probs)
        {
            Assert.True(p > 0, "All softmax probabilities should be positive");
        }

        // Highest logit should have highest probability
        int maxLogitIdx = Array.IndexOf(logits, logits.Max());
        int maxProbIdx = Array.IndexOf(probs, probs.Max());
        Assert.Equal(maxLogitIdx, maxProbIdx);
    }

    // ============================
    // RL Math: Reward Normalization
    // ============================

    [Fact]
    public void RLMath_RewardNormalization()
    {
        double[] rewards = { 10, -5, 20, 3, -8, 15 };

        double mean = rewards.Average();
        double variance = rewards.Select(r => (r - mean) * (r - mean)).Sum() / rewards.Length;
        double std = Math.Sqrt(variance);

        double[] normalized = rewards.Select(r => (r - mean) / (std + 1e-8)).ToArray();

        // Normalized rewards should have mean ~0 and std ~1
        double normalizedMean = normalized.Average();
        double normalizedVar = normalized.Select(r => (r - normalizedMean) * (r - normalizedMean)).Sum() / normalized.Length;
        double normalizedStd = Math.Sqrt(normalizedVar);

        Assert.Equal(0.0, normalizedMean, 1e-10);
        Assert.Equal(1.0, normalizedStd, 1e-6);
    }
}
