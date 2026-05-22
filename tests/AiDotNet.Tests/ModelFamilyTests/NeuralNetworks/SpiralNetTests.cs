using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Per Gong et al. 2019 "SpiralNet++: A Fast and Highly Efficient Mesh Convolution
/// Operator" (arXiv 1911.05856), the model processes 3D mesh data as a rank-3
/// tensor of shape [batch, num_vertices, in_features]. The paper's CoMA / face-mesh
/// experiments use 5023 vertices × 3 coordinate features per vertex; the
/// SpiralNetOptions defaults to a 64-vertex fallback mesh for small-input testing
/// (NumVertices = 64, InputFeatures = 3 = xyz coords).
/// </summary>
public class SpiralNetTests : NeuralNetworkModelTestBase
{
    // [batch, num_vertices, in_features] matches SpiralNetOptions defaults
    // (NumVertices = 64, InputFeatures = 3). The base's default [1, 4] rank-2
    // shape feeds a rank-2 tensor into GlobalPoolingLayer (which requires
    // rank-3, rank-4, or rank-5), producing ArgumentException at line 236
    // of GlobalPoolingLayer.OnFirstForward.
    protected override int[] InputShape => [1, 64, 3];

    // Default output: ModelNet40 classification (NumClasses = 40 per Gong et al.).
    protected override int[] OutputShape => [1, 40];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new SpiralNet<double>();

    /// <summary>
    /// SpiralConvLayer is lazy — its weight tensor is constructed at [0, 0] in
    /// the ctor and only resolves to its final [outputChannels, inputChannels ×
    /// spiralLength] shape during the first Forward (OnFirstForward at
    /// src/NeuralNetworks/Layers/SpiralConvLayer.cs:485 reads input.Shape to
    /// determine InputChannels). The base
    /// <c>NeuralNetworkBase.ParameterCount</c> calls
    /// <c>ResolveLazyLayerShapes</c> which propagates the architecture's input
    /// shape through generic Dense/Conv chains, but SpiralConv's
    /// vertex-features input contract <c>[B, V, C]</c> doesn't fit that
    /// propagation (the chain expects flat-feature layers), so the lazy
    /// layers stay at length 0 pre-Forward and <c>ParameterCount</c>
    /// returns 0. Override the test with an explicit warm-up Predict to
    /// materialize the weights before the count is read, matching the
    /// pattern used in <c>Training_ShouldChangeParameters</c>.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task Parameters_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var input = CreateRandomTensor(InputShape, rng);
        // Warm-up Predict materializes lazy SpiralConvLayer weights.
        network.Predict(input);
        Assert.True(network.ParameterCount > 0,
            "Neural network should have learnable parameters after warm-up Predict.");
    }
}
