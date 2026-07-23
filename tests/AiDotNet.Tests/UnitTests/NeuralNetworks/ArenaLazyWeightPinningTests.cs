// Regression test for #1643. The DEFAULT training route wraps a step in an
// active TensorArena (the shared ModelFamily test base does exactly
// `using var _arena = TensorArena.Create()`). MobileNetV3 — built from
// ConvolutionalLayer + DenseLayer — materializes its weights LAZILY on the
// first forward, i.e. INSIDE that arena. Before the fix those layers allocated
// their long-lived weights through TensorAllocator.Rent (the RECYCLABLE scratch
// tier), so the next step's Reset() rewound the scratch cursor and the
// following transient allocations reissued the exact buffers the weights lived
// in — silently overwriting the weights. Symptom: eval-mode Forward became
// non-deterministic and GetParameters drifted between calls.
//
// The fix routes lazy weight materialization through TensorAllocator.RentPinned
// (the pinned tier, which survives Reset()). This test pins the contract for
// the two culprit layer types: after materializing inside an arena, a Reset()
// plus a flood of same-shaped scratch tensors must NOT change the weights, and
// a second eval forward on the same input must be bit-identical to the first.

using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class ArenaLazyWeightPinningTests
{
    private static Tensor<float> Filled(int[] shape, int seed)
    {
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = (float)(((i * 7 + seed * 13) % 17) - 8) * 0.1f;
        return new Tensor<float>(data, shape);
    }

    // Rewind the arena and flood the scratch tier with same-shaped transient
    // tensors carrying a poison value. Pre-fix, these reissued the buffers the
    // lazy weights were (wrongly) Rent'd from, so any weight aliased to scratch
    // would be overwritten with the poison.
    private static void ResetAndClobberScratch(TensorArena arena, params int[] elementCounts)
    {
        arena.Reset();
        foreach (var count in elementCounts)
        {
            for (int rep = 0; rep < 4; rep++)
            {
                var t = TensorAllocator.Rent<float>(new[] { count });
                var span = t.Data.Span;
                for (int i = 0; i < span.Length; i++) span[i] = 123456.0f; // poison
            }
        }
    }

    private static void AssertBitIdentical(Vector<float> a, Vector<float> b, string what)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]); // exact: weights must not move at all
    }

    [Fact]
    public void DenseLayer_LazyWeightsMaterializedInsideArena_SurviveResetAndScratchReuse()
    {
        using var arena = TensorArena.Create();
        var layer = new DenseLayer<float>(outputSize: 8);
        var input = Filled(new[] { 4, 16 }, seed: 1); // [batch=4, features=16]

        var out1 = layer.Forward(input).ToVector().Clone(); // materializes weights inside the arena
        var w1 = layer.GetParameters().Clone();

        // Next "training step": weights [16,8]=128 and biases [8]=8 elements.
        ResetAndClobberScratch(arena, 16 * 8, 8);

        var w2 = layer.GetParameters();
        var out2 = layer.Forward(input).ToVector();

        AssertBitIdentical(w1, w2, "DenseLayer weights drifted after arena Reset()");
        AssertBitIdentical(out1, out2, "DenseLayer eval output became non-deterministic under an arena");
    }

    [Fact]
    public void ConvolutionalLayer_LazyKernelsMaterializedInsideArena_SurviveResetAndScratchReuse()
    {
        using var arena = TensorArena.Create();
        var layer = new ConvolutionalLayer<float>(outputDepth: 4, kernelSize: 3, stride: 1, padding: 1);
        var input = Filled(new[] { 1, 3, 8, 8 }, seed: 2); // [N=1, C=3, H=8, W=8]

        var out1 = layer.Forward(input).ToVector().Clone(); // materializes kernels inside the arena
        var w1 = layer.GetParameters().Clone();

        // Kernels [4,3,3,3]=108 and biases [4]=4 elements.
        ResetAndClobberScratch(arena, 4 * 3 * 3 * 3, 4);

        var w2 = layer.GetParameters();
        var out2 = layer.Forward(input).ToVector();

        AssertBitIdentical(w1, w2, "ConvolutionalLayer kernels drifted after arena Reset()");
        AssertBitIdentical(out1, out2, "ConvolutionalLayer eval output became non-deterministic under an arena");
    }
}
