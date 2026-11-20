using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TimeSeries;

[Collection("FusedTrainingSerial")]
public sealed class InformerPositionEmbeddingTests
{
    [Fact]
    public void Cpu_position_tables_preserve_broadcast_values_and_input_gradients() => Verify(new CpuEngine());

    [SkippableFact]
    public void Cuda_position_tables_preserve_broadcast_values_and_input_gradients()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable && gpu.GetBackend() is CudaBackend, "CUDA unavailable.");
        Verify(gpu);
    }

    private static void Verify(IEngine engine)
    {
        var original = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int width = 8;
            var network = new InformerNetwork<float>(engine, new List<LayerBase<float>>(),
                new InformerConfiguration(1, 1, 1, width, 1, 1, 1, 16, 1, 0, false, false,
                    InformerAttention.ProbSparse, false, 0, 1, 1));
            var encoder = network.EncoderEmbedding;
            var decoder = network.DecoderEmbedding;
            var held = encoder.Positions(5, width);
            var snapshot = held.ToArray();
            Assert.Same(held, encoder.Positions(5, width));
            Assert.NotSame(held, decoder.Positions(5, width));

            foreach (var (length, batch) in new[] { (5, 1), (7, 3), (5, 2) })
            {
                var inputValues = Enumerable.Range(0, batch * length * width).Select(i => i * .03125f).ToArray();
                using var tokens = new Tensor<float>(inputValues, new[] { batch, length, width });
                using var tape = new GradientTape<float>();
                var actual = encoder.AddPositions(engine, tokens);
                var loss = engine.ReduceSum(engine.TensorMultiply(actual, actual), null);
                var gradients = tape.ComputeGradients(loss, new[] { tokens })[tokens];
                var expected = new float[inputValues.Length];
                for (int i = 0; i < expected.Length; i++)
                {
                    int position = i / width % length, feature = i % width;
                    double angle = position * Math.Exp((feature & ~1) * -(Math.Log(10000.0) / width));
                    expected[i] = inputValues[i] + (float)(feature % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
                }
                Assert.Equal(expected, actual.ToArray());
                Assert.Equal(expected.Select(value => 2 * value).ToArray(), gradients.ToArray());
            }
            Assert.Equal(snapshot, held.ToArray());
        }
        finally { AiDotNetEngine.Current = original; }
    }
}
