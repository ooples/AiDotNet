using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class NeuralNetworkTests : NeuralNetworkModelTestBase
{
    // NeuralNetwork default: inputSize=128, outputSize=1
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    // Default NeuralNetwork is a small MLP that converges to near-zero MSE
    // on the memorization task in a single Train call. The relative-decrease
    // check then false-fires when both lossStep1 and lossFinal sit at the
    // float-quantization floor (~1e-5). NeuralNetworkModelTestBase exposes
    // MemorizationTaskAbsoluteLossFloor exactly for fast-converging models —
    // sub-floor loss counts as a pass, while sign-error / oscillation /
    // explosion still trip the check because they push loss above the floor.
    protected override double MemorizationTaskAbsoluteLossFloor => 1e-4;
}
