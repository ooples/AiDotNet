using AiDotNet.Interfaces;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

[Collection("NonParallelIntegration")]
public sealed class CompiledBindingTraversalTests
{
    [Fact]
    public void BindingsIncludeNestedAndExtraSourcesOnceAndStayStableUntilChanged()
    {
        using var model = new BindingNetwork();
        var nested = new BindingLayer();
        var extra = new BindingLayer();
        model.Install(new Composite(nested));
        model.Extras.Add(new Composite(extra));
        model.Extras.Add(nested);
        long initial = model.CompiledBindingVersion;
        Assert.Equal(1, nested.Reads);
        Assert.Equal(1, extra.Reads);
        Assert.Equal(initial, model.CompiledBindingVersion);
        nested.Version++;
        long nestedChange = model.CompiledBindingVersion;
        Assert.NotEqual(initial, nestedChange);
        extra.Version++;
        Assert.NotEqual(nestedChange, model.CompiledBindingVersion);
        Assert.Equal(model.CompiledBindingVersion, model.CompiledBindingVersion);
    }

    [Fact]
    public void EqualVersionSourceReplacementChangesGenerationWithoutStructureNotification()
    {
        using var model = new BindingNetwork();
        var original = new BindingLayer();
        model.Install(original);
        long generation = model.CompiledBindingVersion;
        int structure = model.LayerStructureVersion;
        ((IList<ILayer<double>>)model.LayersReadOnly)[0] = new BindingLayer();
        Assert.Equal(structure, model.LayerStructureVersion);
        Assert.NotEqual(generation, model.CompiledBindingVersion);
    }

    [AiDotNet.Attributes.TensorLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Features, Direction = AiDotNet.Attributes.TensorLayoutDirection.Input)]
    [AiDotNet.Attributes.TensorLayout(AiDotNet.Enums.TensorAxis.Batch, AiDotNet.Enums.TensorAxis.Features, Direction = AiDotNet.Attributes.TensorLayoutDirection.Output)]
    private sealed class BindingNetwork : NeuralNetworkBase<double>
    {
        internal List<LayerBase<double>?> Extras { get; } = new();
        internal BindingNetwork() : base(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new MeanSquaredErrorLoss<double>()) { }
        internal void Install(ILayer<double> layer) => AddLayerToCollection(layer);
        protected override void InitializeLayers() { }
        protected override IEnumerable<LayerBase<double>?> GetExtraTrainableLayers() => Extras;
        public override ModelMetadata<double> GetModelMetadata() => new() { Name = nameof(BindingNetwork) };
    }

    [AiDotNet.Attributes.ElementWiseShape]
    private sealed class BindingLayer : LayerBase<double>, ICompiledBindingSource
    {
        internal int Version = 1, Reads;
        int ICompiledBindingSource.CompiledBindingVersion { get { Reads++; return Version; } }
        internal BindingLayer() : base(new[] { 1 }, new[] { 1 }) { }
        public override bool SupportsTraining => true;
        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;
        public override void ResetState() { }
    }

    [AiDotNet.Attributes.ElementWiseShape]
    private sealed class Composite : LayerBase<double>
    {
        internal Composite(LayerBase<double> child) : base(new[] { 1 }, new[] { 1 }) => RegisterSubLayer(child);
        public override bool SupportsTraining => true;
        protected override Tensor<double> ForwardTraced(Tensor<double> input) => input;
        public override void ResetState() { }
    }
}
