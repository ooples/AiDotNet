using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning;
using AiDotNet.ReinforcementLearning.Agents.A2C;
using AiDotNet.ReinforcementLearning.Agents.A3C;
using AiDotNet.ReinforcementLearning.Agents.Bandits;
using AiDotNet.ReinforcementLearning.Agents.NStepQLearning;
using AiDotNet.ReinforcementLearning.Agents.NStepSARSA;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.ReinforcementLearning.Agents.Rainbow;
using AiDotNet.ReinforcementLearning.Common;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.ReinforcementLearning.Policies;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Reference values for the return recursions (GAE, n-step and actor-critic returns at rollout and episode
/// boundaries) and for the Half-precision guards of the RL log-densities and counters.
/// </summary>
public class RLReturnAndHalfReferenceTests
{
    private const double Gamma = 0.9;
    private const double Lambda = 0.8;

    private static Vector<double> Vec(params double[] values) => new Vector<double>(values);

    private static readonly Vector<double> S0 = Vec(0.1, -0.2);
    private static readonly Vector<double> S1 = Vec(0.4, 0.3);
    private static readonly Vector<double> S2 = Vec(-0.5, 0.6);
    private static readonly Vector<double> S3 = Vec(0.2, 0.9);
    private static readonly Vector<double> Action0 = Vec(1.0, 0.0);

    private static TField Field<TField>(object owner, string name) where TField : class
        => owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(owner) as TField
            ?? throw new InvalidOperationException("Field missing: " + name);

    private static void Invoke(object owner, string method)
    {
        var info = owner.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new InvalidOperationException("Method missing: " + method);
        info.Invoke(owner, null);
    }

    private static double Value(INeuralNetwork<double> network, Vector<double> state)
        => network.Predict(Tensor<double>.FromVector(state)).ToVector()[0];

    /// <summary>
    /// The episode s0 -> (terminal) then s1 -> s2 -> s3 cut off mid-episode: GAE (Schulman et al. 2016, Eq. 16;
    /// Stable-Baselines3 compute_returns_and_advantage) masks V(s_{t+1}) at the terminal step, and bootstraps the
    /// last step from V(s3), the value of its NEXT state.
    /// </summary>
    [Fact]
    public void PPO_Gae_MasksTheTerminalStepAndBootstrapsFromTheLastNextState()
    {
        using var agent = new PPOAgent<double>(new PPOOptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            DiscountFactor = Gamma,
            GaeLambda = Lambda,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            Seed = 5,
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, true);
        agent.StoreExperience(S1, Action0, 0.5, S2, false);
        agent.StoreExperience(S2, Action0, -1.0, S3, false);

        Invoke(agent, "ComputeAdvantages");
        var trajectory = Field<Trajectory<double>>(agent, "_trajectory");
        var valueNetwork = Field<INeuralNetwork<double>>(agent, "_valueNetwork");
        double v0 = Value(valueNetwork, S0), v1 = Value(valueNetwork, S1);
        double v2 = Value(valueNetwork, S2), v3 = Value(valueNetwork, S3);

        double a2 = -1.0 + Gamma * v3 - v2;
        double a1 = 0.5 + Gamma * v2 - v1 + Gamma * Lambda * a2;
        double a0 = 1.0 - v0;
        var raw = new[] { a0, a1, a2 };
        var values = new[] { v0, v1, v2 };
        double mean = raw.Average();
        double std = Math.Sqrt(raw.Sum(a => (a - mean) * (a - mean)) / (raw.Length - 1));

        var returns = trajectory.Returns ?? throw new InvalidOperationException("Returns not computed.");
        var advantages = trajectory.Advantages ?? throw new InvalidOperationException("Advantages not computed.");
        for (int t = 0; t < 3; t++)
        {
            Assert.Equal(raw[t] + values[t], returns[t], 9);
            Assert.Equal((raw[t] - mean) / (std + 1e-8), advantages[t], 9);
        }
    }

    /// <summary>
    /// A2C's n-step returns (Mnih et al. 2016, Algorithm S3): R = V(s_T) for a rollout cut off mid-episode,
    /// R_t = r_t + gamma R_{t+1}, and R_t = r_t at a terminal step.
    /// </summary>
    [Fact]
    public void A2C_Returns_BootstrapFromTheLastNextStateWhenTheRolloutStopsMidEpisode()
    {
        using var agent = new A2CAgent<double>(new A2COptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            DiscountFactor = Gamma,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            Seed = 5,
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, true);
        agent.StoreExperience(S1, Action0, 0.5, S2, false);
        agent.StoreExperience(S2, Action0, -1.0, S3, false);

        Invoke(agent, "ComputeAdvantages");
        var trajectory = Field<Trajectory<double>>(agent, "_trajectory");
        var valueNetwork = Field<INeuralNetwork<double>>(agent, "_valueNetwork");

        double r2 = -1.0 + Gamma * Value(valueNetwork, S3);
        double r1 = 0.5 + Gamma * r2;
        var expected = new[] { 1.0, r1, r2 };
        var returns = trajectory.Returns ?? throw new InvalidOperationException("Returns not computed.");
        for (int t = 0; t < 3; t++) Assert.Equal(expected[t], returns[t], 9);
    }

    /// <summary>
    /// Termination vs truncation (Gymnasium; SB3 and CleanRL bootstrap on truncation only). Episode A is cut off by a
    /// time limit after s0 (next state s1), then episode B runs s2 -> s3 to the end of the rollout. The truncated step
    /// bootstraps V(s1) and stops the advantage chain: A_0 = 1 + gamma V(s1) - V(s0), not 1 - V(s0) as a terminal
    /// step would give, and not a chain into episode B.
    /// </summary>
    [Fact]
    public void PPO_Gae_BootstrapsATruncatedStepAndStopsTheChainThere()
    {
        using var agent = new PPOAgent<double>(new PPOOptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            DiscountFactor = Gamma,
            GaeLambda = Lambda,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            Seed = 5,
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, terminated: false, truncated: true);
        agent.StoreExperience(S2, Action0, 0.5, S3, terminated: false, truncated: false);

        Invoke(agent, "ComputeAdvantages");
        var trajectory = Field<Trajectory<double>>(agent, "_trajectory");
        var valueNetwork = Field<INeuralNetwork<double>>(agent, "_valueNetwork");
        double v0 = Value(valueNetwork, S0), v1 = Value(valueNetwork, S1);
        double v2 = Value(valueNetwork, S2), v3 = Value(valueNetwork, S3);

        double a0 = 1.0 + Gamma * v1 - v0;
        double a1 = 0.5 + Gamma * v3 - v2;
        var returns = trajectory.Returns ?? throw new InvalidOperationException("Returns not computed.");
        Assert.Equal(a0 + v0, returns[0], 9);
        Assert.Equal(a1 + v2, returns[1], 9);
    }

    /// <summary>A2C's n-step return after a truncated step is r + gamma V(s_{t+1}), not r.</summary>
    [Fact]
    public void A2C_Returns_BootstrapATruncatedStep()
    {
        using var agent = new A2CAgent<double>(new A2COptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            DiscountFactor = Gamma,
            PolicyHiddenLayers = new List<int> { 8 },
            ValueHiddenLayers = new List<int> { 8 },
            Seed = 5,
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, terminated: false, truncated: true);
        agent.StoreExperience(S2, Action0, 0.5, S3, terminated: true, truncated: false);

        Invoke(agent, "ComputeAdvantages");
        var trajectory = Field<Trajectory<double>>(agent, "_trajectory");
        var valueNetwork = Field<INeuralNetwork<double>>(agent, "_valueNetwork");

        var returns = trajectory.Returns ?? throw new InvalidOperationException("Returns not computed.");
        Assert.Equal(1.0 + Gamma * Value(valueNetwork, S1), returns[0], 9);
        Assert.Equal(0.5, returns[1], 9);
    }

    /// <summary>CartPole's time limit is a truncation (as in Gymnasium's CartPole-v1), reported in the info.</summary>
    [Fact]
    public void CartPole_TimeLimit_IsReportedAsTruncation()
    {
        var environment = new CartPoleEnvironment<double>(maxSteps: 1, seed: 3);
        environment.Reset();

        var (_, _, done, info) = environment.Step(Vec(1.0));

        Assert.True(done);
        Assert.True(EnvironmentInfoKeys.IsTruncated(info));
    }

    /// <summary>A3C bootstraps the worker's rollout from V(s_{t_max}), the last step's NEXT state.</summary>
    [Fact]
    public void A3C_Returns_BootstrapFromTheNextStateNotTheLastState()
    {
        using var agent = new A3CAgent<double>(new A3COptions<double> { StateSize = 2, ActionSize = 2, DiscountFactor = Gamma, Seed = 3 });
        var valueNetwork = Field<INeuralNetwork<double>>(agent, "_globalValueNetwork");
        var trajectory = new List<(Vector<double> state, Vector<double> action, double reward, bool done, double value)>
        {
            (S0, Action0, 1.0, false, Value(valueNetwork, S0)),
            (S1, Action0, 0.5, false, Value(valueNetwork, S1)),
        };

        var returns = agent.ComputeReturns(trajectory, valueNetwork, S2);

        double r1 = 0.5 + Gamma * Value(valueNetwork, S2);
        Assert.Equal(r1, returns[1], 9);
        Assert.Equal(1.0 + Gamma * r1, returns[0], 9);
    }

    /// <summary>
    /// n-step Q-learning at an episode end (Sutton &amp; Barto 2018, Section 7.2): EVERY pending step is updated with
    /// its truncated return, not only the oldest. Episode (s0, a0, r 1) -> (s1, a0, r 2, terminal), n = 3, alpha 0.5:
    /// Q(s0, a0) = 0.5 (1 + 0.9 * 2) = 1.4 and Q(s1, a0) = 0.5 * 2 = 1.
    /// </summary>
    [Fact]
    public void NStepQLearning_EpisodeEnd_UpdatesEveryPendingStep()
    {
        var agent = new NStepQLearningAgent<double>(new NStepQLearningOptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            NSteps = 3,
            LearningRate = 0.5,
            DiscountFactor = Gamma,
            LossFunction = new MeanSquaredErrorLoss<double>(),
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, false);
        agent.StoreExperience(S1, Action0, 2.0, S2, true);

        AssertNonZeroQ(agent.GetParameters(), 1.4, 1.0);
    }

    /// <summary>The n-step SARSA counterpart (Sutton &amp; Barto 2018, Figure 7.2), same episode and values.</summary>
    [Fact]
    public void NStepSarsa_EpisodeEnd_UpdatesEveryPendingStep()
    {
        var agent = new NStepSARSAAgent<double>(new NStepSARSAOptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            NSteps = 3,
            LearningRate = 0.5,
            DiscountFactor = Gamma,
            LossFunction = new MeanSquaredErrorLoss<double>(),
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, false);
        agent.StoreExperience(S1, Action0, 2.0, S2, true);

        AssertNonZeroQ(agent.GetParameters(), 1.4, 1.0);
    }

    private static void AssertNonZeroQ(Vector<double> parameters, params double[] expected)
    {
        var nonZero = parameters.ToArray().Where(q => q != 0.0).OrderByDescending(q => q).ToArray();
        Assert.Equal(expected.Length, nonZero.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], nonZero[i], 12);
    }

    /// <summary>
    /// Rainbow's n-step replay (Hessel et al. 2018; Dopamine and Stable-Baselines3 flush the n-step queue at an
    /// episode end): a two-step episode with n = 3 stores both transitions with their truncated returns.
    /// </summary>
    [Fact]
    public void Rainbow_EpisodeEnd_StoresEveryPendingTransition()
    {
        using var agent = new RainbowDQNAgent<double>(new RainbowDQNOptions<double>
        {
            StateSize = 2,
            ActionSize = 2,
            NSteps = 3,
            SharedLayers = new List<int> { 8 },
            ValueStreamLayers = new List<int> { 8 },
            AdvantageStreamLayers = new List<int> { 8 },
            NumAtoms = 5,
            DiscountFactor = Gamma,
        });
        agent.StoreExperience(S0, Action0, 1.0, S1, false);
        agent.StoreExperience(S1, Action0, 2.0, S2, true);

        Assert.Equal(2.0, agent.GetMetrics()["buffer_size"], 12);
    }

    // ── Half guards ───────────────────────────────────────────────────────────────────────────────────────────

    private static readonly INumericOperations<Half> HalfOps = MathHelper.GetNumericOperations<Half>();

    private static Vector<Half> HalfVec(params double[] values)
        => new Vector<Half>(values.Select(v => HalfOps.FromDouble(v)).ToArray());

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    /// <summary>log(p + 1e-10) of a zero probability: 1e-10 is zero in Half, so the floor must be added in double.</summary>
    [Fact]
    public void LogWithFloor_ZeroHalfProbability_IsTheLogOfTheFloor()
    {
        Assert.Equal(0.0, HalfOps.ToDouble(HalfOps.FromDouble(1e-10)));

        double logProb = HalfOps.ToDouble(RLTensorOps.LogWithFloor(HalfOps.Zero, 1e-10));

        Assert.Equal(HalfOps.ToDouble(HalfOps.FromDouble(Math.Log(1e-10))), logProb);
    }

    /// <summary>
    /// A tanh-squashed Gaussian (SAC Appendix C) at a saturated action a = 1: 1 - 1e-6 rounds to 1 in Half, so a
    /// clip at that bound leaves atanh(1) = infinity; the bound must stay below one.
    /// </summary>
    [Fact]
    public void SquashedGaussianLogProb_SaturatedHalfAction_IsFinite()
    {
        Assert.Equal(1.0, HalfOps.ToDouble(HalfOps.FromDouble(1.0 - 1e-6)));

        Half logProb = RLTensorOps.SquashedGaussianLogProb(HalfVec(0.0), HalfVec(0.0), HalfVec(1.0));

        Assert.True(IsFinite(HalfOps.ToDouble(logProb)), $"log pi(1) = {HalfOps.ToDouble(logProb)} in Half.");
    }

    /// <summary>In double the squashed log-density is log N(atanh(a); mu, sigma) - log(1 - a^2 + 1e-6).</summary>
    [Fact]
    public void SquashedGaussianLogProb_Double_MatchesTheChangeOfVariables()
    {
        const double a = 0.7, mu = 0.2, logStd = -0.3;
        double u = 0.5 * Math.Log((1 + a) / (1 - a));
        double sigma = Math.Exp(logStd);
        double z = (u - mu) / sigma;
        double expected = -0.5 * z * z - logStd - 0.5 * Math.Log(2 * Math.PI) - Math.Log(1 - a * a + 1e-6);

        Assert.Equal(expected, RLTensorOps.SquashedGaussianLogProb(Vec(mu), Vec(logStd), Vec(a)), 10);
    }

    /// <summary>A Beta policy's log-density at the action bound: 1 - 1e-7 is 1 in Half, so log(1 - x) would be -inf.</summary>
    [Fact]
    public void BetaLogDensity_HalfActionAtTheUpperBound_IsFinite()
    {
        Half logProb = BetaPolicy<Half>.BetaLogDensity(HalfVec(0.0, 0.0), HalfVec(1.0), 1, -1.0, 1.0);

        Assert.True(IsFinite(HalfOps.ToDouble(logProb)), $"log p(1) = {HalfOps.ToDouble(logProb)} in Half.");
    }

    /// <summary>
    /// Equal advantages have std 0; (A - mean) / (0 + 1e-8) is 0 in float/double but 0 / 0 = NaN in Half, where 1e-8
    /// rounds to zero. The normalised advantages must be zero there.
    /// </summary>
    [Fact]
    public void NormalizeAdvantages_EqualHalfAdvantages_AreZeroNotNaN()
    {
        var normalized = RLTensorOps.NormalizeAdvantages(new List<Half> { HalfOps.One, HalfOps.One, HalfOps.One });

        Assert.All(normalized, value => Assert.Equal(0.0, HalfOps.ToDouble(value)));
    }

    /// <summary>The unbiased std (torch.std, as Stable-Baselines3 normalises): [1, 2, 3] has mean 2 and std 1.</summary>
    [Fact]
    public void NormalizeAdvantages_Double_UsesTheUnbiasedStd()
    {
        var normalized = RLTensorOps.NormalizeAdvantages(new List<double> { 1.0, 2.0, 3.0 });

        Assert.Equal(-1.0 / (1.0 + 1e-8), normalized[0], 12);
        Assert.Equal(0.0, normalized[1], 12);
        Assert.Equal(1.0 / (1.0 + 1e-8), normalized[2], 12);
    }

    /// <summary>
    /// UCB1 (Auer et al. 2002) past t = 65504 steps in Half: ln(t) of the Half step count is ln(inf) = inf for every
    /// arm, which pins the choice to arm 0. Arm 1 always pays 1 and arm 0 pays 0, so the bonus c sqrt(ln t / N) can
    /// pull arm 0 only about c^2 ln t ~ 45 times; the last 200 of 66000 steps must go to arm 1.
    /// </summary>
    [Fact]
    public void UcbBandit_HalfPastHalfMaxSteps_KeepsExploitingTheBetterArm()
    {
        var agent = new UCBBanditAgent<Half>(new UCBBanditOptions<Half> { NumArms = 2, ExplorationParameter = 2.0 });
        var state = HalfVec(0.0);
        int lateBest = 0;
        for (int step = 0; step < 66000; step++)
        {
            var action = agent.SelectAction(state);
            bool best = HalfOps.ToDouble(action[1]) > 0.5;
            if (step >= 65800 && best) lateBest++;
            agent.StoreExperience(state, action, best ? HalfOps.One : HalfOps.Zero, state, false);
        }

        Assert.True(lateBest >= 195, $"{lateBest} of the last 200 steps chose the better arm.");
    }

    /// <summary>The posterior mean s / (s + f) from counts whose sum (80000) overflows Half: 40000 / 80000 = 0.5.</summary>
    [Fact]
    public void ThompsonSampling_HalfMetricsFromLargeCounts_AreTheExactMean()
    {
        var agent = new ThompsonSamplingAgent<Half>(new ThompsonSamplingOptions<Half> { NumArms = 1 });
        Field<Vector<int>>(agent, "_successCounts")[0] = 40000;
        Field<Vector<int>>(agent, "_failureCounts")[0] = 40000;

        Assert.Equal(0.5, HalfOps.ToDouble(agent.GetMetrics()["mean_arm_0"]));
    }

    /// <summary>
    /// Element-wise equality of long Half vectors: a count of matches in Half saturates at 2048, so 2050 equal pairs
    /// would not reach 2050, and 2049 pairs with one mismatch would round up to 2049 (2048).
    /// </summary>
    [Fact]
    public void SameValues_LongHalfVectors_ComparesEveryElement()
    {
        var ones = Enumerable.Repeat(1.0, 2050).ToArray();
        Assert.True(RLTensorOps.SameValues(HalfVec(ones), HalfVec(ones)));

        var shorter = Enumerable.Repeat(1.0, 2049).ToArray();
        var changed = (double[])shorter.Clone();
        changed[100] = 2.0;
        Assert.False(RLTensorOps.SameValues(HalfVec(shorter), HalfVec(changed)));
    }
}
