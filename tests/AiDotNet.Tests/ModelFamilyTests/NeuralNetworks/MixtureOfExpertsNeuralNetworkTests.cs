using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for MixtureOfExpertsNeuralNetwork per Shazeer et al. (2017)
/// "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer".
///
/// Paper architecture: d_model input/output, 4x FFN expansion per expert, Top-K sparse routing.
/// Test uses scaled-down dims (64) while preserving architectural ratios and constraints:
///   - InputDim == OutputDim (residual compatibility per paper §3.1)
///   - HiddenExpansion = 4 (standard FFN expansion per Transformer convention)
///   - NumExperts = 4, TopK = 2 (paper evaluates K=2 as default sparse routing)
/// </summary>
public class MixtureOfExpertsNeuralNetworkTests : NeuralNetworkModelTestBase
{
    // Paper: d_model = 1024–4096; scaled to 64 for test speed
    // InputDim == OutputDim per §3.1 (MoE replaces FFN in Transformer, residual connection)
    protected override int[] InputShape => [64];
    protected override int[] OutputShape => [64];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var options = new MixtureOfExpertsOptions<double>
        {
            NumExperts = 4,          // Paper §4: 4–4096 experts
            TopK = 2,                // Paper §3.2: noisy top-k gating with K=2
            InputDim = 64,           // Scaled from paper's 1024
            OutputDim = 64,          // Must match InputDim for residual connection
            HiddenExpansion = 4,     // Paper §3.1: 4x expansion (Transformer FFN standard)
            UseLoadBalancing = true,  // Paper §4.1: importance loss for balanced routing
            LoadBalancingWeight = 0.01
        };

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 64,
            outputSize: 64);

        return new MixtureOfExpertsNeuralNetwork<double>(options, architecture);
    }
}
