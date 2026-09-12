using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

[Trait("category", "unit")]
public sealed class FinancialA2CLiveStorageTests
{
    public FinancialA2CLiveStorageTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void Normal_inference_preserves_running_statistics_and_pending_behavior()
    {
        using var normalization = new BatchNormalizationLayer<double>(4);
        using var agent = CreateWithPrefix(normalization);
        var actor = Networks(agent, A2C, 4, 3).Single(n => n.Role == FinancialNetworkRole.Policy).Network;
        var mean = normalization.GetRunningMean();
        var variance = normalization.GetRunningVariance();
        var meanBefore = mean.ToArray();
        var varianceBefore = variance.ToArray();
        int meanVersion = mean.Version, varianceVersion = variance.Version;
        var state = State(4, 1);
        var action = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, action, 1.0, state, true);
        // A real multi-row actor prediction must establish evaluation mode, even after
        // a caller previously selected training mode on its supplied normalization layer.
        normalization.SetTrainingMode(true);
        actor.Predict(new Tensor<double>(new[] { 2, 4 }, new Vector<double>(state.ToArray().Concat(state.ToArray()).ToArray())));
        Assert.Equal(meanBefore, mean.ToArray());
        Assert.Equal(varianceBefore, variance.ToArray());
        Assert.Equal(meanVersion, mean.Version);
        Assert.Equal(varianceVersion, variance.Version);
        var secondAction = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, secondAction, 2.0, state, true);
        var before = agent.GetParameters().ToArray();
        agent.Train();
        Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
    }

    [Fact]
    public void External_running_statistic_change_is_a_real_policy_change()
    {
        using var normalization = new BatchNormalizationLayer<double>(4);
        using var agent = CreateWithPrefix(normalization);
        var actor = Networks(agent, A2C, 4, 3).Single(n => n.Role == FinancialNetworkRole.Policy).Network;
        var state = State(4, 1);
        var input = Tensor<double>.FromVector(state);
        var outputBefore = actor.Predict(input).ToArray();
        var oldAction = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, oldAction, 1.0, state, true);
        var outstandingAction = agent.SelectAction(state, training: true);
        var mean = normalization.GetRunningMean();
        mean.SetFlat(0, mean.GetFlat(0) + 3);
        Assert.False(outputBefore.SequenceEqual(actor.Predict(input).ToArray()));
        Assert.Throws<InvalidOperationException>(() => agent.StoreExperience(state, outstandingAction, 2.0, state, true));
        var before = agent.GetParameters().ToArray();
        Assert.Equal(0.0, agent.Train());
        Assert.Equal(before, agent.GetParameters().ToArray());
    }

    [Fact]
    public void Forward_only_scratch_churn_does_not_invalidate_actions()
    {
        using var scratch = new ScratchPrefix();
        using var agent = CreateWithPrefix(scratch);
        var state = State(4, 1);
        var first = agent.SelectAction(state, training: true);
        var second = agent.SelectAction(state, training: true);
        Assert.True(scratch.Forwards >= 2);
        agent.StoreExperience(state, first, 1.0, state, true);
        agent.StoreExperience(state, second, 2.0, state, true);
        var before = agent.GetParameters().ToArray();
        agent.Train();
        Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
    }

    [Fact]
    public void Alias_registration_and_declaration_order_do_not_change_policy_ownership()
    {
        using var prefix = new ReorderingPrefix();
        using var agent = CreateWithPrefix(prefix);
        var state = State(4, 1);
        var first = agent.SelectAction(state, training: true);
        var chunks = (IParameterChunkSource<double>)prefix;
        var originalOrder = chunks.GetParameterStateChunks().Select(c => c.SourceTensor).ToArray();
        prefix.ReverseDeclarations();
        var reversedOrder = chunks.GetParameterStateChunks().Select(c => c.SourceTensor).ToArray();
        Assert.Equal(2, reversedOrder.Length);
        Assert.Same(originalOrder[0], reversedOrder[1]);
        Assert.Same(originalOrder[1], reversedOrder[0]);
        var second = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, first, 1.0, state, true);
        agent.StoreExperience(state, second, 2.0, state, true);
        var before = agent.GetParameters().ToArray();
        agent.Train();
        Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Opaque_child_under_LayerBase_does_not_copy_detached_parameters_during_collection(bool replace)
    {
        using var child = new FinancialA2CPolicyOwnershipTests.LegacyLayer();
        using var composite = new OpaqueComposite(child);
        var options = Options(A2C, 4, 3, seed: 51);
        options.BatchSize = 2;
        options.WarmupSteps = 0;
        var architecture = Arch(4, 3);
        architecture.Layers.Add(composite);
        using var agent = new FinancialA2CAgent<double>(architecture, Arch(4, 1), options);
        var state = State(4, 1);
        var firstAction = agent.SelectAction(state, training: true); // Materialize before measuring the collection path.
        int reads = child.ParameterReads;
        agent.StoreExperience(state, firstAction, 1.0, state, true);
        Assert.Equal(reads, child.ParameterReads);
        if (replace) agent.SetParameters(agent.GetParameters());
        reads = child.ParameterReads;
        var secondAction = agent.SelectAction(state, training: true);
        agent.StoreExperience(state, secondAction, 2.0, state, true);
        Assert.Equal(reads, child.ParameterReads);
        var before = agent.GetParameters().ToArray();
        var loss = agent.Train();
        if (replace)
        {
            Assert.Equal(0.0, loss);
            Assert.Equal(before, agent.GetParameters().ToArray());
        }
        else
        {
            Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Fp16_resident_parameter_tracks_real_storage_not_conversion_snapshots(bool mutate)
    {
        using var prefix = new HalfPrefix();
        using var agent = CreateWithPrefix(prefix);
        var state = State(4, 1);
        var action = agent.SelectAction(state, training: true);
        if (mutate)
        {
            prefix.Scale.SetFlat(0, (Half)2f);
            Assert.Throws<InvalidOperationException>(() => agent.StoreExperience(state, action, 1.0, state, true));
        }
        else
        {
            agent.StoreExperience(state, action, 1.0, state, true);
            var second = agent.SelectAction(state, training: true);
            agent.StoreExperience(state, second, 2.0, state, true);
            var before = agent.GetParameters().ToArray();
            agent.Train();
            Assert.False(before.SequenceEqual(agent.GetParameters().ToArray()));
        }
    }

    [Fact]
    public void Storage_iterator_returns_the_original_fp16_tensor_and_current_version()
    {
        using var prefix = new HalfPrefix();
        var first = Assert.Single(prefix.GetParameterStorageVersions());
        Assert.Same(prefix.Scale, first.Storage);
        Assert.IsType<Tensor<Half>>(first.Storage);
        prefix.Scale.SetFlat(0, (Half)2f);
        var second = Assert.Single(prefix.GetParameterStorageVersions());
        Assert.Same(first.Storage, second.Storage);
        Assert.Equal(prefix.Scale.Version, second.Version);
        Assert.True(second.Version > first.Version);
    }

    private static FinancialA2CAgent<double> CreateWithPrefix(ILayer<double> prefix)
    {
        var options = Options(A2C, 4, 3, seed: 51);
        options.BatchSize = 2;
        options.WarmupSteps = 0;
        options.EntropyCoefficient = 0;
        var architecture = Arch(4, 3);
        architecture.Layers.Add(prefix);
        architecture.Layers.Add(new DenseLayer<double>(3, (IActivationFunction<double>)new IdentityActivation<double>()));
        return new FinancialA2CAgent<double>(architecture, Arch(4, 1), options);
    }

    private sealed class ScratchPrefix : LayerBase<double>
    {
        [Scratch]
        private readonly Tensor<double> _forwardScratch = new(new[] { 1 });
        public ScratchPrefix() : base(new[] { 4 }, new[] { 4 }) { }
        public int Forwards { get; private set; }
        public override bool SupportsTraining => false;
        protected override Tensor<double> ForwardTraced(Tensor<double> input)
        {
            _forwardScratch.SetFlat(0, ++Forwards);
            return input;
        }
        public override void ResetState() { }
    }

    private sealed class OpaqueComposite : LayerBase<double>
    {
        private readonly FinancialA2CPolicyOwnershipTests.LegacyLayer _child;
        public OpaqueComposite(FinancialA2CPolicyOwnershipTests.LegacyLayer child)
            : base(new[] { 4 }, new[] { 3 })
        {
            _child = child;
            RegisterSubLayer(child);
        }
        public override bool SupportsTraining => true;
        protected override Tensor<double> ForwardTraced(Tensor<double> input) => _child.Forward(input);
        public override void ResetState() => _child.ResetState();
    }

    private sealed class ReorderingPrefix : LayerBase<double>
    {
        private readonly Tensor<double> _first = new(new[] { 1 });
        private readonly Tensor<double> _second = new(new[] { 1 });
        private bool _reverse;
        public ReorderingPrefix() : base(new[] { 4 }, new[] { 4 }) { }
        public override bool SupportsTraining => false;
        public void ReverseDeclarations()
        {
            _reverse = true;
            // This is the same storage, not a newly added policy parameter. Registration
            // refreshes the declaration cache while its identity dedup removes the alias.
            RegisterBuffer(_first, "same_storage_alias");
        }
        protected override void AppendDeclaredParameterComponents(List<DeclaredParameterComponent> components)
        {
            DeclareParameterBuffer(components, _reverse ? _second : _first, "first", ParameterSlotRole.Buffer);
            DeclareParameterBuffer(components, _reverse ? _first : _second, "second", ParameterSlotRole.Buffer);
        }
        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;
        public override void ResetState() { }
    }

    private sealed class HalfPrefix : LayerBase<double>
    {
        public Tensor<Half> Scale { get; } = new(new[] { 1 });
        public HalfPrefix() : base(new[] { 4 }, new[] { 4 }) => Scale.SetFlat(0, (Half)1f);
        public override bool SupportsTraining => true;
        protected override void AppendDeclaredParameterComponents(List<DeclaredParameterComponent> components) =>
            DeclareTrainableParameter(components, tensor: null, lowPrecisionTensor: Scale);
        protected override Tensor<double> ForwardTraced(Tensor<double> input) =>
            Engine.TensorMultiplyScalar(input, (double)(float)Scale.GetFlat(0));
        public override void ResetState() { }
    }
}
