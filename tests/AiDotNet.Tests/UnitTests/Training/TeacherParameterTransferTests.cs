using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.Diagnostics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

[Collection("NonParallelIntegration")]
public class TeacherParameterTransferTests
{
    [Theory]
    [InlineData(0, 0)] [InlineData(0, .75)]
    [InlineData(1, 0)] [InlineData(1, .75)]
    [InlineData(2, 0)] [InlineData(2, .75)]
    [Trait("Category", "GPU")]
    public void ProjectorCopyAndEma_StayOnDeviceAndPreserveIndependentStorage(int kind, double momentum)
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            IProjectorHead<float> Make(bool source) => kind switch
            {
                0 => new LinearProjector<float>(3, 2),
                1 => new MLPProjector<float>(3, 4, 2),
                _ => new SymmetricProjector<float>(3, 4, 2, source ? 4 : 0)
            };
            var source = Make(true); var target = Make(false);
            var from = source.GetTrainableTensors(); var to = target.GetTrainableTensors();
            for (int i = 0; i < from.Count; i++) for (int j = 0; j < from[i].Length; j++) from[i][j] = .1f * (i + j);
            var expected = to.Select((t, i) => t.ToArray().Select((v, j) => (float)(v * momentum + from[i][j] * (1 - momentum))).ToArray()).ToArray();
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                ProjectorParameterTransfer<float>.Update(source, target, momentum, kind == 2);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            for (int i = 0; i < to.Count; i++)
            {
                Assert.Same(to[i], target.GetTrainableTensors()[i]);
                Assert.NotSame(from[i], to[i]);
                for (int j = 0; j < to[i].Length; j++) Assert.InRange(MathF.Abs(to[i][j] - expected[i][j]), 0, 1e-6f);
            }
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void MomentumEncoder_CopyAndUpdateUseResidentChunks()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            NeuralNetwork<float> Make() => new(new NeuralNetworkArchitecture<float>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 3, outputSize: 2, layers: new List<ILayer<float>> { new DenseLayer<float>(2) }));
            using var source = Make(); using var target = Make();
            var from = source.GetParameterChunks().ToArray(); var to = target.GetParameterChunks().ToArray();
            Assert.NotEmpty(from); Assert.Equal(from.Length, to.Length);
            foreach (var tensor in from) { Assert.True(tensor.Length > 0); for (int j = 0; j < tensor.Length; j++) tensor[j] = 2f; }
            foreach (var tensor in to) { Assert.True(tensor.Length > 0); for (int j = 0; j < tensor.Length; j++) tensor[j] = -3f; }
            Assert.NotEqual(from[0][0], to[0][0]);
            var momentum = new MomentumEncoder<float>(target, .75);
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                momentum.CopyFromMainEncoder(source);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            Assert.All(to, tensor => Assert.All(tensor.ToArray(), v => Assert.Equal(2f, v)));
            foreach (var tensor in from) ((IEngine)gpu).TensorCopy(gpu.TensorAddScalar(tensor, 4f), tensor);
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                momentum.UpdateFromMainEncoder(source);
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            for (int i = 0; i < to.Length; i++)
                for (int j = 0; j < to[i].Length; j++) Assert.InRange(MathF.Abs(to[i][j] - 3f), 0, 1e-6f);
        }
        finally { AiDotNetEngine.Current = prior; }
    }

    [Fact]
    [Trait("Category", "GPU")]
    public void CompositeSslModel_CheckpointUsesLiveEncoderAndProjectorChunks()
    {
        using var backend = new OpenClBackend(); Assert.True(backend.IsAvailable);
        using var direct = new DirectGpuEngine(backend); using var gpu = new DirectGpuTensorEngine(direct);
        var prior = AiDotNetEngine.Current; AiDotNetEngine.Current = gpu;
        try
        {
            using var encoder = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 3, outputSize: 2, layers: new List<ILayer<float>> { new DenseLayer<float>(2) }));
            var projector = new MLPProjector<float>(2, 4, 2);
            var model = new SimCLR<float>(encoder, projector);
            var expected = model.GetParameters().ToArray();
            var checkpoint = new AiDotNet.Training.ModelParameterCheckpoint<float, Tensor<float>, Tensor<float>>(model);
            using (var scope = GpuResidencyScope.Begin(captureOperations: true))
            {
                var chunks = model.GetParameterChunks().ToArray();
                Assert.All(projector.GetTrainableTensors(), tensor => Assert.Contains(tensor, chunks));
                checkpoint.Capture();
                foreach (var tensor in chunks) ((IEngine)gpu).TensorCopy(gpu.TensorAddScalar(tensor, 3f), tensor);
                checkpoint.Restore();
                Assert.True(scope.Downloads == 0, string.Join("\n", scope.Events));
            }
            Assert.Equal(expected, model.GetParameters().ToArray());
        }
        finally { AiDotNetEngine.Current = prior; }
    }
}
