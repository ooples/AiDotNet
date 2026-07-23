using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class GraphAttentionNetworkTests : GraphNNModelTestBase<float>
{
    protected override int[] InputShape => [10, 128];
    protected override int[] OutputShape => [10, 7];

    // Ensure all network instances start from identical weights.
    // Tensor<T>.CreateRandom uses cryptographic seeds, so each call produces
    // different weights. We save the first network's params and reuse them.
    private static Vector<float>? _savedParams;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // dropoutRate: 0 for the invariant suite. The paper default (0.6, Veličković et al. 2018 §3.3)
        // is a TRAINING-TIME regularizer that zeros 60 % of the attention coefficients each step to
        // curb overfitting on real graphs — it is NOT part of the attention mechanism's math. The
        // memorization/gradient probes here test CAPACITY (can the model fit a fixed pair?), the exact
        // opposite of what dropout aids: at 0.6 the per-step gradient is so noisy the loss cannot
        // descend (observed step-1 0.355 → step-100 0.609). Disabling dropout leaves the paper-faithful
        // multi-head attention forward/backward fully exercised while making the capacity probes valid.
        var network = new GraphAttentionNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.MultiClassClassification,
                inputSize: 128,
                outputSize: 7),
            dropoutRate: 0.0);
        if (_savedParams == null)
            _savedParams = network.GetParameters();
        else
            network.UpdateParameters(_savedParams);
        return network;
    }
}
