using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Guards the fixed-matrix execution path used while an echo-state reservoir settles.
/// </summary>
public sealed class EchoStateNetworkAllocationTests
{
    [Fact]
    public void Predict_WithMaximumSettling_ReusesFixedWeightLayouts()
    {
        const int width = 64;
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: width,
            outputSize: 1);
        using var model = new EchoStateNetwork<float>(
            architecture,
            reservoirSize: width,
            warmupPeriod: 200,
            reservoirInputScalarActivation: null);
        var input = new Tensor<float>([1, width]);
        for (int i = 0; i < input.Length; i++)
            input[i] = (i + 1) / 100f;

        // Warm JIT and engine dispatch. The next prediction still executes all 200 settle steps.
        _ = model.Predict(input);

        long before = GC.GetAllocatedBytesForCurrentThread();
        var output = model.Predict(input);
        long allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - before;
        GC.KeepAlive(output);

        // Re-transposing both 64x64 float matrices for every step creates at least 6,553,600 bytes
        // of matrix payload alone. Leave ample room for the state vectors while keeping that former
        // implementation deterministically outside the contract.
        Assert.True(
            allocatedBytes < 4_000_000,
            $"A maximum-settle prediction allocated {allocatedBytes:N0} bytes; fixed reservoir " +
            "weight layouts must be derived once and reused across settling iterations.");
    }
}
