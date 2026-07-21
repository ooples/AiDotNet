using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

[Collection("FusedOptimizerGlobalState")]
public class VoxelCNNTests : NeuralNetworkModelTestBase<float>
{
    // VoxelCNN default: 32x32x32 voxels, 1 channel
    // Actual output is 128-dim from conv feature extraction
    protected override int[] InputShape => [1, 32, 32, 32];
    protected override int[] OutputShape => [128];

    // 3D convolutions on 32³ voxel grids are inherently expensive on CPU
    // (one Conv3D forward at the default Layers stack takes ≳ 200 ms on
    // consumer hardware). MoreData_ShouldNotDegrade at the default 50/200
    // iter count = 250 × 200 ms ≈ 50 s per network × 2 networks ≈ 100 s,
    // and pairs with the test's setup / arena work to overflow the 120 s
    // xUnit per-test timeout. 1 / 2 still exercises the "long ≥ short
    // shouldn't degrade" invariant — same pattern Forecasting Foundation
    // models and paper-scale CLIP encoders use.
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;
    protected override double MoreDataTolerance => 0.5;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new VoxelCNN<float>();

    [Fact(Timeout = 120000)]
    public async Task FusedTraining_ShouldUpdateParametersWithoutNaN()
    {
        await Task.Yield();
        var originalOptions = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            CompiledTapeTrainingStep<float>.Invalidate();
            CompiledTapeTrainingStep<float>.ResetFusedStepCount();

            using var network = new TestableVoxelCNN();
            var rng = ModelTestHelpers.CreateSeededRandom();
            var input = CreateRandomTensor(InputShape, rng);
            var target = CreateRandomTargetTensor(OutputShape, rng);
            var parametersBefore = network.GetParameters().ToArray();

            network.Train(input, target);

            Assert.True(
                CompiledTapeTrainingStep<float>.GetFusedStepCount() > 0,
                "VoxelCNN training fell back instead of executing the fused compiled path.");
            Assert.False(
                network.FusedTrainingDisabled,
                "VoxelCNN's fused step failed and sticky-disabled compilation before eager fallback.");
            var parametersAfter = network.GetParameters().ToArray();
            Assert.True(
                parametersBefore.Where((value, index) => value != parametersAfter[index]).Any(),
                "VoxelCNN's fused compiled step did not update any live model parameter.");
            foreach (var parameter in network.GetParameterChunks())
            {
                Assert.All(
                    parameter.ToArray(),
                    value => Assert.True(!float.IsNaN(value) && !float.IsInfinity(value)));
            }
        }
        finally
        {
            CompiledTapeTrainingStep<float>.Invalidate();
            TensorCodecOptions.SetCurrent(originalOptions);
        }
    }

    private sealed class TestableVoxelCNN : VoxelCNN<float>
    {
        internal bool FusedTrainingDisabled => _fusedTrainingDisabled;
    }
}
