// #1624 (component #1 rollout): LayerNormalizationLayer caches _lastInput /
// _lastMean / _lastVariance for its manual backward. They are write-only under
// tape autodiff (Engine.LayerNorm records its own backward state; the layer has
// no Backward method reading them), so Forward skips them when a GradientTape is
// recording — freeing a redundant reference to each activation. This proves the
// skip is correctness-neutral: the gradient that flows back THROUGH the layer is
// bit-identical whether the manual caches are populated or skipped.

using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class LayerNormalizationCacheSkipParityTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = (float)(((i * 7 + seed * 13) % 17) - 8) * 0.1f;
        return new Tensor<float>(data, shape);
    }

    [Fact]
    public void TapeGradients_Identical_WhetherManualBackwardCacheIsKeptOrSkipped()
    {
        var engine = new CpuEngine();
        var layer = new LayerNormalizationLayer<float>();
        layer.SetTrainingMode(true);
        var input = Rand(new[] { 4, 8 }, seed: 1);

        layer.Forward(input);             // init (no tape)
        var snapshot = layer.GetParameters();

        float[] InputGrad(bool keepCache)
        {
            LayerBase<float>.KeepActivationCacheUnderTape = keepCache;
            try
            {
                layer.SetParameters(snapshot);
                using var tape = new GradientTape<float>();
                var output = layer.Forward(input);
                var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null);
                var grads = tape.ComputeGradients(loss, new[] { input });
                return grads[input].AsSpan().ToArray();
            }
            finally
            {
                LayerBase<float>.KeepActivationCacheUnderTape = false;
            }
        }

        var keep = InputGrad(keepCache: true);
        var skip = InputGrad(keepCache: false);

        Assert.Equal(keep.Length, skip.Length);
        Assert.Contains(keep, g => g != 0f);
        for (int i = 0; i < keep.Length; i++)
            Assert.Equal(keep[i], skip[i], 5);
    }
}
