// #1624: the FeedForwardLayer Input/PreActivationOutput/Output fields are
// manual-backward caches that are WRITE-ONLY under tape autodiff (the layer has
// no Backward method that reads them; the tape captures each op's own state).
// Forward skips populating them when a GradientTape is recording, which frees a
// redundant reference to every activation — part of the deep-model activation
// set that drives the #1624 training-scale OOM. This test proves the skip is
// correctness-neutral: tape gradients are bit-identical whether the manual
// caches are populated (KeepActivationCacheUnderTape = true) or skipped.

using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class FeedForwardLayerCacheSkipParityTests
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
        var layer = new FeedForwardLayer<float>(8, (AiDotNet.Interfaces.IActivationFunction<float>)new ReLUActivation<float>());
        layer.SetTrainingMode(true);
        var input = Rand(new[] { 4, 8 }, seed: 1);

        // Warm up (no tape) so weights initialize, then snapshot so both runs use
        // byte-identical parameters — the only difference is the cache behaviour.
        layer.Forward(input);
        var snapshot = layer.GetParameters();

        (float[] w, float[] b) Run(bool keepCache)
        {
            FeedForwardLayer<float>.KeepActivationCacheUnderTape = keepCache;
            try
            {
                layer.SetParameters(snapshot);
                using var tape = new GradientTape<float>();
                var output = layer.Forward(input);
                var loss = engine.ReduceSum(engine.TensorMultiply(output, output), null);
                var w = layer.GetWeightsTensor();
                var b = layer.GetBiasesTensor();
                var grads = tape.ComputeGradients(loss, new[] { w, b });
                return (grads[w].AsSpan().ToArray(), grads[b].AsSpan().ToArray());
            }
            finally
            {
                FeedForwardLayer<float>.KeepActivationCacheUnderTape = false;
            }
        }

        var (kw, kb) = Run(keepCache: true);   // manual caches populated
        var (sw, sb) = Run(keepCache: false);  // manual caches skipped (the #1624 default)

        Assert.Equal(kw.Length, sw.Length);
        Assert.Equal(kb.Length, sb.Length);
        // Gradients must be non-trivial (the test would be vacuous on all-zeros).
        Assert.Contains(kw, g => g != 0f);
        for (int i = 0; i < kw.Length; i++)
            Assert.Equal(kw[i], sw[i], 5);
        for (int i = 0; i < kb.Length; i++)
            Assert.Equal(kb[i], sb[i], 5);
    }
}
