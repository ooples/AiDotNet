using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

[Trait("category", "unit")]
public sealed class FinancialA2CRolloutContractTests
{
    public enum PolicyReplacement { Parameters, Deserialization, ExplicitGradient }
    public enum FailurePoint { BeforeCriticUpdate, AfterCriticUpdate }
    private enum ForwardFailure { None, Prediction, Training }

    public FinancialA2CRolloutContractTests() => TestModuleInitializer.EnsureInitialized();

    public static IEnumerable<object[]> InvalidActions()
    {
        yield return new object[] { Array.Empty<double>() };
        yield return new object[] { new[] { 1.0, 0.0 } };
        yield return new object[] { new[] { 1.0, 0.0, 0.0, 0.0 } };
        yield return new object[] { new[] { 0.0, 0.0, 0.0 } };
        yield return new object[] { new[] { 1.0, 1.0, 0.0 } };
        yield return new object[] { new[] { 0.25, 0.75, 0.0 } };
        yield return new object[] { new[] { 1.0, -0.001, 0.0 } };
        yield return new object[] { new[] { 1.0, double.NaN, 0.0 } };
        yield return new object[] { new[] { 1.0, double.PositiveInfinity, 0.0 } };
        yield return new object[] { new[] { 1.0, double.NegativeInfinity, 0.0 } };
        yield return new object[] { new[] { BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(1.0) - 1), 0.0, 0.0 } };
    }

    [Theory]
    [MemberData(nameof(InvalidActions))]
    public void Store_rejects_non_one_hot_actions_without_queuing_them(double[] values)
    {
        using var fixture = new RolloutFixture(batchSize: 1);
        var state = State(4, 1);
        Assert.Equal("action", Assert.Throws<ArgumentException>(() => fixture.Agent.StoreExperience(
            state, new Vector<double>(values), 1.0, state, true)).ParamName);
        var before = fixture.Agent.GetParameters().ToArray();
        Assert.Equal(0.0, fixture.Agent.Train());
        Assert.Equal(before, fixture.Agent.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Store_accepts_every_exact_one_hot_action_including_signed_zero(int selected)
    {
        using var fixture = new RolloutFixture(batchSize: 1);
        var action = new Vector<double>(new[] { -0.0, -0.0, -0.0 });
        action[selected] = 1.0;
        var state = State(4, 1);
        fixture.Agent.StoreExperience(state, action, 1.0, state, true);
        fixture.Agent.Train();
        Assert.Single(fixture.Actor.TrainingInputs);
    }

    [Fact]
    public void One_hot_validation_does_not_round_decimal_values_through_double()
    {
        var options = new FinancialA2CAgentOptions<decimal> { StateSize = 1, ActionSize = 2, HiddenLayers = Array.Empty<int>() };
        using var agent = new FinancialA2CAgent<decimal>(
            new NeuralNetworkArchitecture<decimal>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 2),
            new NeuralNetworkArchitecture<decimal>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        decimal almostOne = 0.9999999999999999999999999999m;
        Assert.Equal(1.0, (double)almostOne); // Establish the conversion trap.
        var state = new Vector<decimal>(new[] { 0m });
        Assert.Throws<ArgumentException>(() => agent.StoreExperience(state, new Vector<decimal>(new[] { almostOne, 0m }), 1m, state, true));
        agent.StoreExperience(state, new Vector<decimal>(new[] { 1m, 0m }), 1m, state, true);
    }

    [Fact]
    public void Train_consumes_each_pending_transition_once_without_sampling_or_leftovers()
    {
        using var fixture = new RolloutFixture(batchSize: 2);
        fixture.Store(5);
        fixture.Agent.Train();
        Assert.Equal(Enumerable.Range(0, 5).SelectMany(i => State(4, i).ToArray()),
            Assert.Single(fixture.Actor.TrainingInputs));
        var updated = fixture.Agent.GetParameters().ToArray();
        Assert.Equal(0.0, fixture.Agent.Train());
        Assert.Equal(updated, fixture.Agent.GetParameters().ToArray());
        Assert.Single(fixture.Actor.TrainingInputs);
    }

    [Fact]
    public void Capacity_bounds_the_current_rollout_without_replaying_old_policy_data()
    {
        using var fixture = new RolloutFixture(batchSize: 2, capacity: 3);
        fixture.Store(5);
        fixture.Agent.Train();
        Assert.Equal(Enumerable.Range(2, 3).SelectMany(i => State(4, i).ToArray()),
            Assert.Single(fixture.Actor.TrainingInputs));
        Assert.Equal(0.0, fixture.Agent.Train());
    }

    [Fact]
    public void Initial_warmup_and_subsequent_batch_readiness_are_distinct()
    {
        using var fixture = new RolloutFixture(batchSize: 2, warmup: 4);
        var initial = fixture.Agent.GetParameters().ToArray();
        fixture.Store(2);
        Assert.Equal(0.0, fixture.Agent.Train());
        Assert.Equal(initial, fixture.Agent.GetParameters().ToArray());
        fixture.Store(2, start: 2);
        fixture.Agent.Train();
        Assert.Single(fixture.Actor.TrainingInputs);
        var firstUpdate = fixture.Agent.GetParameters().ToArray();
        Assert.False(initial.SequenceEqual(firstUpdate));

        fixture.Store(1, start: 4);
        Assert.Equal(0.0, fixture.Agent.Train());
        Assert.Equal(firstUpdate, fixture.Agent.GetParameters().ToArray());
        fixture.Store(1, start: 5);
        fixture.Agent.Train();
        Assert.Equal(2, fixture.Actor.TrainingInputs.Count);
        Assert.Equal(8, fixture.Actor.TrainingInputs[1].Length);
        Assert.False(firstUpdate.SequenceEqual(fixture.Agent.GetParameters().ToArray()));
    }

    [Fact]
    public void Stored_vectors_are_owned_snapshots()
    {
        using var expected = new RolloutFixture(batchSize: 1);
        using var actual = new RolloutFixture(batchSize: 1);
        var state = State(4, 1);
        var next = State(4, 2);
        var action = OneHot(3, 2);
        expected.Agent.StoreExperience(state.Clone(), action.Clone(), 1.0, next.Clone(), false);
        actual.Agent.StoreExperience(state, action, 1.0, next, false);
        state[0] = 500;
        next[0] = -500;
        action[2] = 0;
        action[1] = 1;
        expected.Agent.Train();
        actual.Agent.Train();
        Assert.Equal(expected.Agent.GetParameters().ToArray(), actual.Agent.GetParameters().ToArray());
    }

    [Theory]
    [InlineData(FailurePoint.BeforeCriticUpdate)]
    [InlineData(FailurePoint.AfterCriticUpdate)]
    public void Failed_updates_never_retry_a_partially_consumed_rollout(FailurePoint point)
    {
        using var fixture = new RolloutFixture(batchSize: 2);
        fixture.Store(2);
        var initialActor = fixture.Actor.GetParameters().ToArray();
        var initialCritic = fixture.Critic.GetParameters().ToArray();
        if (point == FailurePoint.BeforeCriticUpdate) fixture.Critic.Failure = ForwardFailure.Prediction;
        else fixture.Actor.Failure = ForwardFailure.Training;
        Assert.Throws<ForwardProbeException>(() => fixture.Agent.Train());
        Assert.Equal(initialActor, fixture.Actor.GetParameters().ToArray());
        Assert.Equal(point == FailurePoint.BeforeCriticUpdate,
            initialCritic.SequenceEqual(fixture.Critic.GetParameters().ToArray()));
        fixture.Actor.Failure = ForwardFailure.None;
        fixture.Critic.Failure = ForwardFailure.None;
        var afterFailure = fixture.Agent.GetParameters().ToArray();
        Assert.Equal(0.0, fixture.Agent.Train());
        Assert.Equal(afterFailure, fixture.Agent.GetParameters().ToArray());
        fixture.Store(2, start: 3);
        fixture.Agent.Train(); // Fresh behavior remains trainable after a failed update.
        Assert.False(afterFailure.SequenceEqual(fixture.Agent.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(PolicyReplacement.Parameters)]
    [InlineData(PolicyReplacement.Deserialization)]
    [InlineData(PolicyReplacement.ExplicitGradient)]
    public void Replacing_or_updating_the_policy_invalidates_pending_behavior(PolicyReplacement replacement)
    {
        using var agent = CreateVanillaAgent();
        var state = State(4, 1);
        agent.StoreExperience(state, OneHot(3, 0), 1.0, state, true);
        var changed = agent.GetParameters().Clone();
        changed[0] += 0.25;
        switch (replacement)
        {
            case PolicyReplacement.Parameters:
                agent.SetParameters(changed);
                break;
            case PolicyReplacement.Deserialization:
                using (var source = CreateVanillaAgent())
                {
                    source.SetParameters(changed);
                    agent.Deserialize(source.Serialize());
                }
                break;
            case PolicyReplacement.ExplicitGradient:
                var actor = Networks(agent, A2C, 4, 3).Single(n => n.Role == FinancialNetworkRole.Policy).Network;
                var gradient = new Vector<double>((int)actor.ParameterCount);
                for (int i = 0; i < gradient.Length; i++) gradient[i] = 0.1;
                agent.ApplyGradients(gradient, 0.01);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(replacement));
        }
        var replaced = agent.GetParameters().ToArray();
        Assert.Equal(0.0, agent.Train());
        Assert.Equal(replaced, agent.GetParameters().ToArray());
    }

    internal static FinancialA2CAgent<double> CreateVanillaAgent()
    {
        var options = Options(A2C, 4, 3, seed: 14);
        options.HiddenLayers = Array.Empty<int>();
        options.BatchSize = 1;
        options.WarmupSteps = 0;
        options.EntropyCoefficient = 0;
        var agent = new FinancialA2CAgent<double>(Arch(4, 3), Arch(4, 1), options);
        agent.SelectAction(State(4, 0), false);
        agent.SetParameters(new Vector<double>((int)agent.ParameterCount));
        return agent;
    }

    [Fact]
    public void Explicit_supervised_update_still_bypasses_batch_and_initial_warmup()
    {
        using var fixture = new RolloutFixture(batchSize: 8, warmup: 20);
        var before = fixture.Agent.GetParameters().ToArray();
        fixture.Agent.Train(State(4, 1), new Vector<double>(new[] { 0.0, 0.0, 1.0 }));
        Assert.Single(fixture.Actor.TrainingInputs);
        Assert.False(before.SequenceEqual(fixture.Agent.GetParameters().ToArray()));
        Assert.Equal(0.0, fixture.Agent.Train());
    }

    private sealed class ForwardProbeException : Exception { }

    private sealed class RecordingDenseLayer : DenseLayer<double>
    {
        public RecordingDenseLayer(int outputs) : base(outputs, (IActivationFunction<double>)new IdentityActivation<double>()) { }
        public List<double[]> TrainingInputs { get; } = new();
        public ForwardFailure Failure { get; set; }
        protected override Tensor<double> ForwardTraced(Tensor<double> input)
        {
            if (Failure == ForwardFailure.Training && IsTrainingMode ||
                Failure == ForwardFailure.Prediction && !IsTrainingMode)
                throw new ForwardProbeException();
            if (IsTrainingMode) TrainingInputs.Add(input.ToArray());
            return base.ForwardTraced(input);
        }
    }

    private sealed class RolloutFixture : IDisposable
    {
        public FinancialA2CAgent<double> Agent { get; }
        public RecordingDenseLayer Actor { get; } = new(3);
        public RecordingDenseLayer Critic { get; } = new(1);
        public RolloutFixture(int batchSize, int warmup = 0, int capacity = 100)
        {
            var options = Options(A2C, 4, 3, seed: 14);
            options.BatchSize = batchSize;
            options.WarmupSteps = warmup;
            options.ReplayBufferSize = capacity;
            options.EntropyCoefficient = 0;
            var actor = Arch(4, 3);
            var critic = Arch(4, 1);
            actor.RandomSeed = 101;
            critic.RandomSeed = 102;
            actor.Layers.Add(Actor);
            critic.Layers.Add(Critic);
            Agent = new FinancialA2CAgent<double>(actor, critic, options);
            Agent.SelectAction(State(4, 0), false);
            Agent.SetParameters(new Vector<double>((int)Agent.ParameterCount));
            Actor.TrainingInputs.Clear();
            Critic.TrainingInputs.Clear();
        }
        public void Store(int count, int start = 0)
        {
            for (int i = start; i < start + count; i++)
                Agent.StoreExperience(State(4, i), OneHot(3, i % 3), 1.0 + i, State(4, i + 1), true);
        }
        public void Dispose() => Agent.Dispose();
    }
}
