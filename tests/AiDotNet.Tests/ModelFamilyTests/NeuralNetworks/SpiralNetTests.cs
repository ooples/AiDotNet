using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

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
}
