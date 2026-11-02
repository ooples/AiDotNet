using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.A2C;
using AiDotNet.ReinforcementLearning.Agents.A3C;
using AiDotNet.ReinforcementLearning.Agents.DynamicProgramming;
using AiDotNet.ReinforcementLearning.Agents.IQL;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.ReinforcementLearning.Agents.TD3;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Training behaviours of the deep and tabular agents that their options promise: the entropy bonus, gradient-norm
/// clipping, the configured critic losses, the evaluation tolerance of modified policy iteration, and the release
/// of A3C's worker networks.
/// </summary>
public class RLAgentTrainingBehaviourTests
{
    private const int StateSize = 4;

    private static List<(Vector<double> State, Vector<double> Action, double Reward, Vector<double> Next, bool Done)>
        Data(int count, int actionSize, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var data = new List<(Vector<double>, Vector<double>, double, Vector<double>, bool)>();
        for (int t = 0; t < count; t++)
        {
            var state = new Vector<double>(StateSize);
            var next = new Vector<double>(StateSize);
            for (int i = 0; i < StateSize; i++)
            {
                state[i] = rng.NextDouble() * 2.0 - 1.0;
                next[i] = rng.NextDouble() * 2.0 - 1.0;
            }
            int chosen = rng.Next(actionSize);
            var action = new Vector<double>(actionSize);
            action[chosen] = 1.0;
            double reward = (chosen == 0 ? 1.0 : -1.0) * state[0];
            data.Add((state, action, reward, next, t % 8 == 7));
        }
        return data;
    }

    private static void Feed(DeepReinforcementLearningAgentBase<double> agent, int count, int actionSize, int seed)
    {
        foreach (var (state, action, reward, next, done) in Data(count, actionSize, seed))
            agent.StoreExperience(state, action, reward, next, done);
    }

    private static double Entropy(Vector<double> logits)
    {
        double max = logits.Max();
        var exp = logits.Select(z => Math.Exp(z - max)).ToArray();
        double sum = exp.Sum();
        return -exp.Sum(e => e / sum * Math.Log(e / sum));
    }

    // The mean entropy of softmax(policy logits) over the training states.
    private static double MeanPolicyEntropy(DeepReinforcementLearningAgentBase<double> agent, int actionSize, int seed)
    {
        var policy = agent.GetNetworks()[0];
        return Data(16, actionSize, seed)
            .Average(d => Entropy(policy.Predict(Tensor<double>.FromVector(d.State)).ToVector()));
    }

    // ── Entropy bonus (Mnih et al. 2016, section 4; Schulman et al. 2017, eq. 9) ──────────────────────────────────

    // loss = -mean(log pi * A) - beta * mean(H(pi)), so with the same data, seed and step, a large beta leaves a more
    // random policy than beta = 0. An agent that drops the bonus trains both identically.
    private static void AssertEntropyBonusRaisesEntropy(string agent, Func<double, double> entropyAfterTraining)
    {
        double without = entropyAfterTraining(0.0);
        double with = entropyAfterTraining(5.0);
        Assert.True(with > without + 1e-6,
            $"{agent}: the entropy bonus did not raise the policy's entropy ({with} with beta 5 vs {without} without).");
    }

    [Fact]
    public void A2C_EntropyBonus_RaisesPolicyEntropy()
    {
        AssertEntropyBonusRaisesEntropy("A2C", beta =>
        {
            var agent = new A2CAgent<double>(new A2COptions<double>
            {
                StateSize = StateSize,
                ActionSize = 3,
                StepsPerUpdate = 16,
                PolicyLearningRate = 0.05,
                ValueLearningRate = 0.01,
                EntropyCoefficient = beta,
                PolicyHiddenLayers = new List<int> { 8 },
                ValueHiddenLayers = new List<int> { 8 },
                Seed = 3,
            });
            Feed(agent, 16, 3, seed: 5);
            agent.Train();
            return MeanPolicyEntropy(agent, 3, seed: 5);
        });
    }

    [Fact]
    public void A3C_EntropyBonus_RaisesPolicyEntropy()
    {
        AssertEntropyBonusRaisesEntropy("A3C", beta =>
        {
            var agent = new A3CAgent<double>(new A3COptions<double>
            {
                StateSize = StateSize,
                ActionSize = 3,
                PolicyLearningRate = 0.05,
                ValueLearningRate = 0.01,
                EntropyCoefficient = beta,
                Seed = 3,
            });
            Feed(agent, 16, 3, seed: 5);
            agent.Train();
            return MeanPolicyEntropy(agent, 3, seed: 5);
        });
    }

    [Fact]
    public void PPO_EntropyBonus_RaisesPolicyEntropy()
    {
        AssertEntropyBonusRaisesEntropy("PPO", beta =>
        {
            var agent = new PPOAgent<double>(new PPOOptions<double>
            {
                StateSize = StateSize,
                ActionSize = 3,
                StepsPerUpdate = 16,
                MiniBatchSize = 16,
                TrainingEpochs = 1,
                PolicyLearningRate = 0.05,
                ValueLearningRate = 0.01,
                EntropyCoefficient = beta,
                Seed = 3,
            });
            Feed(agent, 16, 3, seed: 5);
            agent.Train();
            return MeanPolicyEntropy(agent, 3, seed: 5);
        });
    }

    // A continuous A3C head emits [mean | log-std]. The Gaussian's entropy is sum(log-std) + const, so the bonus
    // must raise the log-std half. The log-std slice once took the full output width, which threw instead.
    [Fact]
    public void A3C_Continuous_EntropyBonusRaisesTheLogStdHalf()
    {
        double MeanLogStd(double beta)
        {
            var agent = new A3CAgent<double>(new A3COptions<double>
            {
                StateSize = StateSize,
                ActionSize = 2,
                IsContinuous = true,
                PolicyLearningRate = 0.05,
                ValueLearningRate = 0.01,
                EntropyCoefficient = beta,
                Seed = 3,
            });
            Feed(agent, 16, 2, seed: 7);
            agent.Train();
            var policy = agent.GetNetworks()[0];
            return Data(16, 2, seed: 7).Average(d =>
            {
                var output = policy.Predict(Tensor<double>.FromVector(d.State)).ToVector();
                return (output[2] + output[3]) / 2.0;
            });
        }

        double without = MeanLogStd(0.0);
        double with = MeanLogStd(5.0);
        Assert.True(with > without + 1e-6, $"log-std {with} with the bonus vs {without} without.");
    }

    // ── PPO gradient-norm clipping (Schulman et al. 2017 reference settings: max_grad_norm 0.5) ──────────────────

    [Fact]
    public void PPO_ClipsBothNetworksGradientsAtMaxGradNorm()
    {
        var agent = new PPOAgent<double>(new PPOOptions<double> { StateSize = StateSize, ActionSize = 2, MaxGradNorm = 0.7 });

        foreach (var network in agent.GetNetworks())
        {
            var optimizer = ((NeuralNetworkBase<double>)network).TrainingOptimizer
                ?? throw new InvalidOperationException("A PPO network has no training optimizer.");
            var options = (GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>)optimizer.GetOptions();
            Assert.True(options.EnableGradientClipping);
            Assert.Equal(GradientClippingMethod.ByNorm, options.GradientClippingMethod);
            Assert.Equal(0.7, options.MaxGradientNorm);
        }
    }

    // ── Configured critic losses ──────────────────────────────────────────────────────────────────────────────

    // A loss whose value never depends on the prediction: its gradient is zero, so a critic trained with it keeps
    // its weights. A critic that ignores the configured loss trains with MSE and moves.
    private sealed class ConstantLoss : LossFunctionBase<double>
    {
        public override double CalculateLoss(Vector<double> predicted, Vector<double> actual) => 0.0;

        public override Tensor<double> ComputeTapeLoss(Tensor<double> predicted, Tensor<double> target)
        {
            var allAxes = Enumerable.Range(0, predicted.Shape.Length).ToArray();
            return Engine.TensorMultiplyScalar(Engine.ReduceMean(predicted, allAxes, keepDims: false), 0.0);
        }
    }

    private static double CriticDisplacement(DeepReinforcementLearningAgentBase<double> agent, int criticIndex)
    {
        var critic = agent.GetNetworks()[criticIndex];
        var before = critic.GetParameters();
        agent.Train();
        var after = critic.GetParameters();
        Assert.Equal(before.Length, after.Length);
        double sum = 0.0;
        for (int i = 0; i < before.Length; i++) sum += (after[i] - before[i]) * (after[i] - before[i]);
        return Math.Sqrt(sum);
    }

    [Fact]
    public void TD3_CriticsTrainWithCriticLossFunction()
    {
        double Displacement(ILossFunction<double> loss)
        {
            var agent = new TD3Agent<double>(new TD3Options<double>
            {
                StateSize = StateSize,
                ActionSize = 2,
                BatchSize = 8,
                WarmupSteps = 8,
                CriticLossFunction = loss,
                Seed = 9,
            });
            Feed(agent, 16, 2, seed: 11);
            agent.Train();
            return CriticDisplacement(agent, criticIndex: 2);
        }

        Assert.True(Displacement(new MeanSquaredErrorLoss<double>()) > 0.0, "MSE did not move the critic.");
        Assert.Equal(0.0, Displacement(new ConstantLoss()));
    }

    [Fact]
    public void IQL_QNetworksTrainWithQLossFunction()
    {
        double Displacement(ILossFunction<double> loss)
        {
            var agent = new IQLAgent<double>(new IQLOptions<double>
            {
                StateSize = StateSize,
                ActionSize = 2,
                BatchSize = 8,
                BufferSize = 64,
                QLossFunction = loss,
                Seed = 9,
            });
            Feed(agent, 16, 2, seed: 11);
            agent.Train();
            return CriticDisplacement(agent, criticIndex: 2);
        }

        Assert.True(Displacement(new MeanSquaredErrorLoss<double>()) > 0.0, "MSE did not move the Q-network.");
        Assert.Equal(0.0, Displacement(new ConstantLoss()));
    }

    // ── Modified policy iteration's evaluation tolerance (Puterman & Shin 1978) ──────────────────────────────

    // One state looping to itself with reward 1 and gamma 0.9: evaluation sweep n raises V by 0.9^(n-1). Theta 0.5
    // stops after the 8th sweep (0.9^7 = 0.478 < 0.5), V = (1 - 0.9^8) / 0.1. Theta 1e-12 runs all 10 sweeps.
    [Theory]
    [InlineData(0.5, 8)]
    [InlineData(1e-12, 10)]
    public void ModifiedPolicyIteration_StopsEvaluationAtTheta(double theta, int sweeps)
    {
        var agent = new ModifiedPolicyIterationAgent<double>(new ModifiedPolicyIterationOptions<double>
        {
            StateSize = 1,
            ActionSize = 1,
            MaxEvaluationSweeps = 10,
            Theta = theta,
            DiscountFactor = 0.9,
        });
        var state = new Vector<double>(new[] { 1.0 });
        var action = agent.SelectAction(state);
        agent.StoreExperience(state, action, 1.0, state, false);

        agent.Train();

        Assert.Equal((1.0 - Math.Pow(0.9, sweeps)) / 0.1, agent.StateValue(state), 9);
    }

    // ── A3C releases its worker networks ─────────────────────────────────────────────────────────────────────

    private static bool IsDisposed(object network)
    {
        for (var type = network.GetType(); type is not null; type = type.BaseType)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>))
            {
                var field = type.GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?? throw new InvalidOperationException("NeuralNetworkBase no longer has a _disposed field.");
                return (bool)(field.GetValue(network) ?? false);
            }
        }
        throw new InvalidOperationException($"{network.GetType()} is not a NeuralNetworkBase.");
    }

    [Fact]
    public void A3C_Dispose_ReleasesTheWorkerNetworks()
    {
        var agent = new A3CAgent<double>(new A3COptions<double> { StateSize = StateSize, ActionSize = 2, Seed = 1 });
        var (policy, value) = agent.WorkerNetworks(0);
        Assert.False(IsDisposed(policy));

        agent.Dispose();

        Assert.True(IsDisposed(policy));
        Assert.True(IsDisposed(value));
    }
}
