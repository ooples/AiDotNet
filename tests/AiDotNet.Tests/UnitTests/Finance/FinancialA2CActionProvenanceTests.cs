using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

[Trait("category", "unit")]
public sealed class FinancialA2CActionProvenanceTests
{
    public enum UntrustedAction { CallerCreated, Cloned, ForeignAgent }
    public enum InvalidTransition { NextStateLength, StateValues, ActionValues }

    public FinancialA2CActionProvenanceTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(UntrustedAction.CallerCreated)]
    [InlineData(UntrustedAction.Cloned)]
    [InlineData(UntrustedAction.ForeignAgent)]
    public void Unstamped_actions_are_rejected_without_changing_a_valid_pending_rollout(UntrustedAction source)
    {
        using var fixture = new ProvenanceFixture();
        using var other = new ProvenanceFixture();
        var state = State(4, 1);
        var sampled = fixture.Agent.SelectAction(state, true);
        Vector<double> untrusted = source switch
        {
            UntrustedAction.CallerCreated => OneHot(3, 0),
            UntrustedAction.Cloned => sampled.Clone(),
            UntrustedAction.ForeignAgent => other.Agent.SelectAction(state, true),
            _ => throw new ArgumentOutOfRangeException(nameof(source)),
        };
        fixture.Agent.StoreExperience(state, sampled, 2.0, state, true);
        Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(state, untrusted, 3.0, state, true));
        fixture.Agent.Train();
        Assert.Equal(state.ToArray(), Assert.Single(fixture.Actor.TrainingInputs));
        Assert.Equal(0.0, fixture.Agent.Train());
    }

    [Fact]
    public void A_sampled_action_is_consumed_exactly_once()
    {
        using var fixture = new ProvenanceFixture();
        var state = State(4, 1);
        var action = fixture.Agent.SelectAction(state, true);
        fixture.Agent.StoreExperience(state, action, 2.0, state, true);
        Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(state, action, 2.0, state, true));
        fixture.Agent.Train();
        Assert.Equal(state.ToArray(), Assert.Single(fixture.Actor.TrainingInputs));
    }

    [Fact]
    public void A_selection_is_bound_to_a_defensive_snapshot_of_its_state()
    {
        using var fixture = new ProvenanceFixture();
        var state = State(4, 1);
        var original = state.Clone();
        var action = fixture.Agent.SelectAction(state, true);
        state[0] += 0.5;
        Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(state, action, 2.0, original, true));
        fixture.Agent.StoreExperience(original, action, 2.0, original, true);
        original[0] += 1.0;
        fixture.Agent.Train();
        Assert.Equal(State(4, 1).ToArray(), Assert.Single(fixture.Actor.TrainingInputs));
    }

    [Theory]
    [InlineData(InvalidTransition.NextStateLength)]
    [InlineData(InvalidTransition.StateValues)]
    [InlineData(InvalidTransition.ActionValues)]
    public void Validation_failure_does_not_consume_the_valid_selection(InvalidTransition invalid)
    {
        using var fixture = new ProvenanceFixture();
        var state = State(4, 1);
        var action = fixture.Agent.SelectAction(state, true);
        switch (invalid)
        {
            case InvalidTransition.NextStateLength:
                Assert.Throws<ArgumentException>(() => fixture.Agent.StoreExperience(state, action, 2.0, State(3, 1), true));
                break;
            case InvalidTransition.StateValues:
                Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(State(4, 2), action, 2.0, state, true));
                break;
            case InvalidTransition.ActionValues:
                int selected = Array.IndexOf(action.ToArray(), 1.0);
                Assert.InRange(selected, 0, 2);
                action[selected] = 0.5;
                Assert.Throws<ArgumentException>(() => fixture.Agent.StoreExperience(state, action, 2.0, state, true));
                action[selected] = 1.0;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(invalid));
        }
        fixture.Agent.StoreExperience(state.Clone(), action, 2.0, state, true);
        fixture.Agent.Train();
        Assert.Equal(state.ToArray(), Assert.Single(fixture.Actor.TrainingInputs));
    }

    [Fact]
    public void A_value_equal_state_copy_is_valid_but_an_action_copy_is_not()
    {
        using var fixture = new ProvenanceFixture();
        var state = State(4, 1);
        var action = fixture.Agent.SelectAction(state, true);
        Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(state.Clone(), action.Clone(), 2.0, state, true));
        fixture.Agent.StoreExperience(state.Clone(), action, 2.0, state, true);
        fixture.Agent.Train();
        Assert.Equal(state.ToArray(), Assert.Single(fixture.Actor.TrainingInputs));
    }

    [Fact]
    public void Supervised_update_isolates_its_label_from_pending_on_policy_transitions()
    {
        using var fixture = new ProvenanceFixture(batchSize: 8, warmup: 20);
        var pendingState = State(4, 1);
        var oldAction = fixture.Agent.SelectAction(pendingState, true);
        fixture.Agent.StoreExperience(pendingState, oldAction, 2.0, pendingState, true);
        var unconsumed = fixture.Agent.SelectAction(pendingState, true);
        var labelledState = State(4, 9);
        var before = fixture.Agent.GetParameters().ToArray();
        fixture.Agent.Train(labelledState, OneHot(3, 2));
        Assert.Equal(labelledState.ToArray(), Assert.Single(fixture.Actor.TrainingInputs));
        Assert.Equal(labelledState.ToArray(), Assert.Single(fixture.Critic.TrainingInputs));
        Assert.False(before.SequenceEqual(fixture.Agent.GetParameters().ToArray()));
        Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(pendingState, unconsumed, 2.0, pendingState, true));
        Assert.Equal(0.0, fixture.Agent.Train());
    }

    [Fact]
    public void Supervised_training_does_not_enable_a_public_collection_bypass()
    {
        using var fixture = new ProvenanceFixture();
        var state = State(4, 1);
        int callbacks = 0;
        fixture.Actor.BeforeTraining = () =>
        {
            callbacks++;
            Assert.Throws<InvalidOperationException>(() => fixture.Agent.StoreExperience(state, OneHot(3, 0), 1.0, state, true));
        };
        fixture.Agent.Train(state, OneHot(3, 2));
        Assert.True(callbacks > 0);
        Assert.Equal(0.0, fixture.Agent.Train());
    }

    [Fact]
    public void Other_agents_keep_the_existing_supervised_store_dispatch()
    {
        using var agent = new StoreObservingDqn();
        var state = State(4, 1);
        var before = agent.GetParameters().ToArray();
        agent.Train(state, OneHot(3, 2));
        Assert.Equal(1, agent.StoreCalls);
        Assert.Equal(OneHot(3, 2).ToArray(), agent.LastAction);
        Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(FinRLAlgorithm.A2C, 1, 0)]
    [InlineData(FinRLAlgorithm.A2C, 8, 20)]
    [InlineData(FinRLAlgorithm.DQN, 8, 20)]
    [InlineData(FinRLAlgorithm.PPO, 8, 20)]
    [InlineData(FinRLAlgorithm.SAC, 8, 20)]
    public void FinRL_supervised_dispatch_preserves_the_inner_agents_contract(FinRLAlgorithm algorithm, int batchSize, int warmup)
    {
        var kind = algorithm switch
        {
            FinRLAlgorithm.A2C => A2C,
            FinRLAlgorithm.DQN => Dqn,
            FinRLAlgorithm.PPO => Ppo,
            FinRLAlgorithm.SAC => Sac,
            _ => throw new ArgumentOutOfRangeException(nameof(algorithm)),
        };
        var options = Options(kind, 4, 3, seed: 14);
        options.BatchSize = batchSize;
        options.WarmupSteps = warmup;
        options.HiddenLayers = Array.Empty<int>();
        options.EntropyCoefficient = 0;
        var secondary = Arch(algorithm == FinRLAlgorithm.SAC ? 7 : 4, 1);
        using var agent = new FinRLAgent<double>(Arch(4, 3), options, algorithm, secondary);
        var state = State(4, 1);
        agent.SelectAction(state, false);
        agent.SetParameters(new Vector<double>((int)agent.ParameterCount));
        var before = agent.GetParameters().ToArray();
        Assert.NotEmpty(before);
        agent.Train(state, OneHot(3, 2));
        Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
    }

    private sealed class StoreObservingDqn : FinancialDQNAgent<double>
    {
        public int StoreCalls { get; private set; }
        public double[] LastAction { get; private set; } = Array.Empty<double>();
        public StoreObservingDqn() : base(Arch(4, 3), CreateOptions())
        {
            SelectAction(State(4, 0), false);
            SetParameters(new Vector<double>((int)ParameterCount));
        }
        private static AiDotNet.Models.Options.TradingAgentOptions<double> CreateOptions()
        {
            var options = FinancialAgentTestKit.Options(Dqn, 4, 3, seed: 14);
            options.HiddenLayers = Array.Empty<int>();
            options.BatchSize = 8;
            options.WarmupSteps = 20;
            return options;
        }
        public override void StoreExperience(Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)
        {
            StoreCalls++;
            LastAction = action.ToArray();
            base.StoreExperience(state, action, reward, nextState, done);
        }
    }

    private sealed class RecordingLayer : DenseLayer<double>
    {
        public RecordingLayer(int outputs) : base(outputs, (IActivationFunction<double>)new IdentityActivation<double>()) { }
        public List<double[]> TrainingInputs { get; } = new();
        public Action? BeforeTraining { get; set; }
        protected override Tensor<double> ForwardTraced(Tensor<double> input)
        {
            if (IsTrainingMode)
            {
                BeforeTraining?.Invoke();
                TrainingInputs.Add(input.ToArray());
            }
            return base.ForwardTraced(input);
        }
    }

    private sealed class ProvenanceFixture : IDisposable
    {
        public FinancialA2CAgent<double> Agent { get; }
        public RecordingLayer Actor { get; } = new(3);
        public RecordingLayer Critic { get; } = new(1);
        public ProvenanceFixture(int batchSize = 1, int warmup = 0)
        {
            var options = Options(A2C, 4, 3, seed: 14);
            options.BatchSize = batchSize;
            options.WarmupSteps = warmup;
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
        public void Dispose() => Agent.Dispose();
    }
}
