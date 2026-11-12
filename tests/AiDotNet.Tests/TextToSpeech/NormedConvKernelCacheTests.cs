using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class NormedConvKernelCacheTests
{
    [Fact]
    public void RecomputedKernelUsesCurrentWeightsWhenScratchStorageHasAnOldCachedTranspose()
    {
        using var scope = new NoGradScope<float>();
        const int channels = 17, outputs = 131, rowWidth = channels * 3;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var weights = Enumerable.Range(0, outputs * rowWidth).Select(i => .07 * Math.Sin(i * .11)).ToArray();
        NormedConv1DLayer<float> Create() => new(channels, outputs, 3, 1, 1, 1, 1,
            false, ConvolutionNormalization.Weight);
        void Load(NormedConv1DLayer<float> layer, double gain)
            => layer.LoadTorchWeights(weights, Enumerable.Repeat(gain, outputs).ToArray(), new double[outputs]);

        var layer = Create();
        Load(layer, 1);
        var oldKernel = layer.Kernel().ToArray();
        Load(layer, 1.5);
        var reference = Create();
        Load(reference, 1.5);
        var input = new Tensor<float>(new[] { 1, channels, 4 }, new Vector<float>(
            Enumerable.Range(0, channels * 4).Select(i => (float)Math.Sin(i * .13))));
        var expected = reference.Forward(input).ToArray();

        // Use the real backend pool that recycles forward intermediates after a tape.
        // Its internal API is accessed here without adding a production testing hook.
        var cache = typeof(Tensor<float>).Assembly.GetType("AiDotNet.Tensors.Helpers.AutoTensorCache")!;
        var rent = cache.GetMethod("RentOrAllocate", BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(typeof(float));
        var giveBack = cache.GetMethod("Return", BindingFlags.NonPublic | BindingFlags.Static)!
            .MakeGenericMethod(typeof(float));
        var shape = new[] { outputs, rowWidth };
        void DrainShape()
        {
            for (int i = 0; i < 8; i++) rent.Invoke(null, new object[] { shape });
        }

        DrainShape();
        try
        {
            // Each reusable array already keys a transpose of the old normalized weights.
            // The next normalization overwrites that storage with the current weights.
            for (int i = 0; i < 4; i++)
            {
                var recycled = new Tensor<float>(shape, new Vector<float>(oldKernel));
                var kernel = engine.Reshape(recycled, new[] { outputs, channels, 1, 3 });
                engine.Conv2D(engine.Reshape(input, new[] { 1, channels, 1, 4 }), kernel,
                    new[] { 1, 1 }, new[] { 0, 1 }, new[] { 1, 1 });
                giveBack.Invoke(null, new object[] { recycled });
            }

            Assert.Equal(expected, layer.Forward(input).ToArray());
        }
        finally { DrainShape(); }
    }
}
