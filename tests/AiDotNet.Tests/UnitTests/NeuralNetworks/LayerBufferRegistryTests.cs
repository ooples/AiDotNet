using AiDotNet.Attributes;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>Locks the canonical, identity-stable persistent-buffer registry contract.</summary>
public sealed class LayerBufferRegistryTests
{
    [Fact]
    public async Task SameIdentity_IsIdempotent_AndReplacementPreservesOrder()
    {
        await Task.Yield();
        using var layer = new BufferProbeLayer();
        var first = new Tensor<double>([2]);
        var second = new Tensor<double>([3]);
        var replacement = new Tensor<double>([2]);

        layer.Add(first, "first");
        layer.Add(first, "first");
        layer.Add(second, "second");
        layer.Add(replacement, "first");

        var buffers = layer.GetRegisteredBuffers();
        Assert.Equal(2, buffers.Count);
        Assert.Equal("first", buffers[0].Name);
        Assert.Same(replacement, buffers[0].Tensor);
        Assert.Equal("second", buffers[1].Name);
        Assert.Same(second, buffers[1].Tensor);
    }

    [Fact]
    public async Task SameIdentity_WithConflictingRole_IsRejected()
    {
        await Task.Yield();
        using var layer = new BufferProbeLayer();
        layer.Add(new Tensor<double>([1]), "state");

        var error = Assert.Throws<InvalidOperationException>(() =>
            layer.Add(
                new Tensor<double>([1]),
                "state",
                PersistentTensorRole.Constant,
                ParameterSlotRole.LearnedState));

        Assert.Contains("state", error.Message, StringComparison.Ordinal);
        Assert.Contains("roles", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ConcurrentPublication_ProducesOneCanonicalEntry()
    {
        await Task.Yield();
        using var layer = new BufferProbeLayer();
        var candidates = Enumerable.Range(0, 64)
            .Select(_ => new Tensor<double>([4]))
            .ToArray();

        Parallel.For(0, candidates.Length, i => layer.Add(candidates[i], "running_mean"));

        var registered = Assert.Single(layer.GetRegisteredBuffers());
        Assert.Equal("running_mean", registered.Name);
        Assert.Contains(registered.Tensor, candidates);
    }

    [Fact]
    public async Task GeneratedBatchNormRegistration_IsIdempotentUnderConcurrentReads()
    {
        await Task.Yield();
        using var layer = new BatchNormalizationLayer<double>(8);

        Parallel.For(0, 64, iteration =>
        {
            layer.GetRegisteredBuffers();
            layer.GetTrainableParameters();
        });

        var buffers = layer.GetRegisteredBuffers();
        Assert.Equal(2, buffers.Count);
        Assert.Equal(new[] { "running_mean", "running_variance" },
            buffers.Select(buffer => buffer.Name).ToArray());
        Assert.Equal(2, layer.GetTrainableParameters().Count);
    }

    [Fact]
    public async Task Dispose_ClearsRegistry_AndMakesPublicationTerminal()
    {
        await Task.Yield();
        var layer = new BufferProbeLayer();
        layer.Add(new Tensor<double>([1]), "state");

        layer.Dispose();

        Assert.Empty(layer.GetRegisteredBuffers());
        Assert.Throws<ObjectDisposedException>(() =>
            layer.Add(new Tensor<double>([1]), "state"));
    }

    [Fact]
    public void DeclaredCount_IncludesAdditionalRuntimeTrainablesWithoutDoubleCountingDeclarations()
    {
        using var layer = new MixedDeclarationProbeLayer();

        bool known = layer.TryGetOwnDeclaredParameterCount(
            out long declaredCount,
            out bool materialized);

        Assert.True(known);
        Assert.True(materialized);
        Assert.Equal(5, declaredCount);
        Assert.Equal(5, layer.GetParameters().Length);
    }

    [ElementWiseShape]
    private sealed class BufferProbeLayer : LayerBase<double>
    {
        public BufferProbeLayer() : base([1], [1])
        {
        }

        public void Add(
            Tensor<double> tensor,
            string name,
            PersistentTensorRole persistenceRole = PersistentTensorRole.Constant,
            ParameterSlotRole stateRole = ParameterSlotRole.Buffer) =>
            RegisterBuffer(tensor, name, persistenceRole, stateRole);

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;

        public override bool SupportsTraining => false;

        public override void ResetState()
        {
        }
    }

    [ElementWiseShape]
    private sealed class MixedDeclarationProbeLayer : LayerBase<double>
    {
        private readonly Tensor<double> _declared = new([2]);
        private readonly Tensor<double> _runtime = new([3]);

        public MixedDeclarationProbeLayer() : base([1], [1])
        {
            RegisterTrainableParameter(_declared, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_runtime, PersistentTensorRole.Biases);
        }

        protected override IReadOnlyList<(
            Tensor<double>? Tensor,
            TensorShape Expected,
            PersistentTensorRole Role)> DeclaredParameterShapes()
            => [(_declared, ShapeOf(2), PersistentTensorRole.Weights)];

        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;

        public override bool SupportsTraining => false;

        public override void ResetState()
        {
        }
    }
}
