using System.IO;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.LazyShape;

/// <summary>
/// Issue #1214: NeuralNetworkBase models should be constructable without a
/// semantic architecture. The layer-only ctor on NeuralNetworkBase passes a
/// stub <see cref="NeuralNetworkArchitecture{T}.CreateLayerOnly"/> so the
/// existing 100+ <c>Architecture.X</c> consumers in derived classes keep
/// working, while methods that need a real input shape consult
/// <see cref="ILayer{T}"/>[0].GetInputShape() instead.
/// </summary>
public class LayerOnlyArchitectureTests
{
    /// <summary>
    /// Concrete subclass exercising the parameterless (layer-only) base ctor.
    /// Implements the minimum abstract surface so the layer-only path is
    /// observable from a test.
    /// </summary>
    private sealed class LayerOnlyNetwork : NeuralNetworkBase<float>
    {
        public LayerOnlyNetwork()
            : base(lossFunction: new MeanSquaredErrorLoss<float>(), maxGradNorm: 1.0)
        {
            Layers.Add(new DenseLayer<float>(outputSize: 16));
            Layers.Add(new DenseLayer<float>(outputSize: 4));
        }

        protected override void InitializeLayers() { /* layers added in ctor */ }

        public override void UpdateParameters(Vector<float> parameters) { /* not exercised by these tests */ }

        public override ModelMetadata<float> GetModelMetadata()
            => new() { Name = "LayerOnlyNetwork" };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(BinaryReader reader) { }

        protected override IFullModel<float, Tensor<float>, Tensor<float>> CreateNewInstance()
            => new LayerOnlyNetwork();
    }

    [Fact]
    public void LayerOnly_Architecture_FlagsTrue()
    {
        var net = new LayerOnlyNetwork();
        Assert.True(net.Architecture.IsLayerOnly);
        Assert.True(net.IsLayerOnlyModel);
    }

    [Fact]
    public void LayerOnly_GetInputShape_DelegatesToFirstLayer()
    {
        var net = new LayerOnlyNetwork();
        var shape = net.GetInputShape();
        Assert.NotNull(shape);

        // GetInputShape must delegate to Layers[0] when layers exist
        // (the layer-derived path) — not to Architecture.InputSize. We
        // verify by reading the same first-layer shape directly and
        // confirming it matches; both layer-only and architecture-bound
        // models share this branch when Layers.Count > 0, but for
        // layer-only the architecture-fallback would give -1 if the
        // delegation regressed.
        var expected = net.Layers[0].GetInputShape();
        Assert.Equal(expected, shape);
    }

    [Fact]
    public void ArchitectureRequired_StillWorks_ForExistingModels()
    {
        // Sanity: passing a real architecture still flags IsLayerOnly = false.
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 4);
        Assert.False(arch.IsLayerOnly);
    }

    [Fact]
    public void CreateLayerOnly_Stub_HasSentinelSpatialDims()
    {
        var stub = NeuralNetworkArchitecture<float>.CreateLayerOnly();
        Assert.True(stub.IsLayerOnly);
        // Stub builds through CreateDynamicSpatial so spatial dims are
        // -1 sentinels — that's how the layer-only contract is conveyed.
        Assert.Equal(-1, stub.InputHeight);
        Assert.Equal(-1, stub.InputWidth);
        Assert.True(stub.HasDynamicSpatialDims);
    }
}
