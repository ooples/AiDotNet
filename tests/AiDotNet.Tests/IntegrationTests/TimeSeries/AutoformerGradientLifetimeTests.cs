using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

[Collection("EngineCurrentGlobalState")]
public sealed class AutoformerGradientLifetimeTests
{
    [Fact]
    public void Accumulated_gradients_survive_scratch_reuse_between_samples()
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var model = new AutoformerModel<double>(new AutoformerOptions<double>
            {
                EmbeddingDim = 8, NumAttentionHeads = 2,
                NumEncoderLayers = 1, NumDecoderLayers = 1,
                LookbackWindow = 4, ForecastHorizon = 1,
            });
            using var arena = TensorArena.Create();
            var input = new Tensor<double>(new[] { 32, 8 },
                new Vector<double>(Enumerable.Repeat(1.0, 256).ToArray()));
            var gradient = AiDotNetEngine.Current.TensorMultiplyScalar(input, 2.0);
            var accumulated = model.AccumulateGradient(null, gradient);
            accumulated = model.AccumulateGradient(accumulated, gradient);

            // The next sample rewinds the arena and rents tensors with the same element
            // count but a different shape, as Autoformer's paired FFN weights do.
            arena.Reset();
            var other = new Tensor<double>(new[] { 8, 32 },
                new Vector<double>(Enumerable.Repeat(7.0, 256).ToArray()));
            for (var i = 0; i < 16; i++)
                _ = AiDotNetEngine.Current.TensorMultiplyScalar(other, 3.0);

            Assert.Equal(new[] { 32, 8 }, accumulated.Shape.ToArray());
            for (var i = 0; i < accumulated.Length; i++) Assert.Equal(4.0, accumulated[i]);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
