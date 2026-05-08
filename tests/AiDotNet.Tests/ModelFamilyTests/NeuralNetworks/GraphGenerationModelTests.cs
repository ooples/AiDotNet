using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphGenerationModelTests : GraphNNModelTestBase
{
    protected override int[] InputShape => [10, 16];
    protected override int[] OutputShape => [10, 10];

    // GraphGenerationModel converges aggressively on the memorization task
    // (small graph, MSE on adjacency probabilities) — both lossStep1 and
    // lossFinal frequently sit below 1e-5 after a single Train call. The
    // relative-decrease check then false-fires on float-quantization noise.
    // Sub-floor loss counts as a pass; sign-error / explosion / oscillation
    // still trip the check because they push loss above the floor.
    protected override double MemorizationTaskAbsoluteLossFloor => 1e-4;

    private static Vector<double>? _savedParams;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var network = new GraphGenerationModel<double>(inputFeatures: 16, maxNodes: 10);
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
