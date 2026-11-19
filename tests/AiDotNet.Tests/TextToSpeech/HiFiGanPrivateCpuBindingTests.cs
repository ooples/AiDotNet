using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class HiFiGanPrivateCpuBindingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReusedBindingReadsChangedInputsAndLiveNormalizedWeights(bool transposed)
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            var cpu = new CpuEngine(); AiDotNetEngine.Current = cpu;
            var layer = new NormedConv1DLayer<float>(2, 3, transposed ? 4 : 3,
                transposed ? 2 : 1, 1, 1, 1, transposed, ConvolutionNormalization.Weight);
            layer.SetTrainingMode(false);
            const int time = 11;
            Assert.True(layer.TryPrivateCpuOutputTime(time, out int outputTime));
            var data = Enumerable.Range(0, 2 * time).Select(i => .13f * (float)Math.Sin(i * .31)).ToArray();
            var result = new float[3 * outputTime];
            using var input3 = Tensor<float>.FromMemory(data, new[] { 1, 2, time });
            using var input4 = Tensor<float>.FromMemory(data, new[] { 1, 2, 1, time });
            using var output4 = Tensor<float>.FromMemory(result, new[] { 1, 3, 1, outputTime });
            using var residual = new Tensor<float>(Enumerable.Range(0, result.Length)
                .Select(i => .017f * (float)Math.Cos(i * .29)).ToArray(), new[] { 1, 3, outputTime });
            using var noGrad = new NoGradScope<float>();
            var binding = layer.BindPrivateCpuInference(input3, input4, output4, residual, true, .1f);
            for (int repeat = 0; repeat < 2; repeat++)
            {
                if (repeat > 0)
                {
                    for (int i = 0; i < data.Length; i++) data[i] += .03f * (i % 5 - 2);
                    var parameters = layer.GetParameters(); parameters[0] += .07f;
                    layer.SetParameters(parameters);
                }
                Array.Fill(result, 777f);
                binding.Execute();
                // Run the binding first so the reference cannot refresh a weight cache for it.
                var actual = output4.ToArray();
                using var expected = layer.Forward(input3);
                cpu.TensorAddInPlace(expected, residual);
                cpu.LeakyReLUInto(expected, expected, .1f);
                Assert.Equal(expected.ToArray(), actual);
                Assert.True(binding.Matches(input3, input4, output4, residual, true, .1f));
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void BindingRejectsNchwViewsOfDifferentBackingStorage()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var layer = new NormedConv1DLayer<float>(2, 3, 3, 1, 1, 1, 1, false, ConvolutionNormalization.Weight);
            layer.SetTrainingMode(false);
            using var input3 = Tensor<float>.FromMemory(new float[14], new[] { 1, 2, 7 });
            using var unrelated4 = Tensor<float>.FromMemory(new float[14], new[] { 1, 2, 1, 7 });
            using var output4 = Tensor<float>.FromMemory(new float[21], new[] { 1, 3, 1, 7 });
            using var noGrad = new NoGradScope<float>();
            Assert.Throws<InvalidOperationException>(() => layer.BindPrivateCpuInference(input3, unrelated4, output4, null, false, 0));
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
