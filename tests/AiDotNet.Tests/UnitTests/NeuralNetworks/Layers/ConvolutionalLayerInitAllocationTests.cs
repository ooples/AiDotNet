using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Materializing a convolutional layer must allocate its kernels ONCE.
/// </summary>
/// <remarks>
/// InitializeWeights filled the kernels through a full-size temporary array, so materializing a
/// layer allocated 2x its own weights. The buffer exists only so the SIMD-batched xoshiro256** fill
/// still applies; the generic branch of that same method was already chunked, and the double/float
/// fast paths now match it.
///
/// Measured on a 512x512 VAE decoder (24 conv layers, 239.9MB of kernels): materialization
/// allocated 479.9MB before, an exact 2.00x, and 240.7MB after. That transient doubling is part of
/// what pushed the 339-layer clone sweep into OutOfMemoryException.
/// </remarks>
public class ConvolutionalLayerInitAllocationTests
{
    private const int OutputDepth = 64;
    private const int KernelSize = 5;
    private const int InputDepth = 64;

    private static ConvolutionalLayer<double> MakeLayer() =>
        new ConvolutionalLayer<double>(outputDepth: OutputDepth, kernelSize: KernelSize, stride: 1, padding: 2);

    [Fact]
    public void SeededInit_IsBitExact_AcrossTheChunkedFill()
    {
        // The chunked fill must draw the SAME random stream as a single full-size draw, or every
        // seeded model in the library silently re-rolls its weights. Two layers with the same seed
        // must agree exactly, and a different seed must actually differ.
        var a = MakeLayer();
        a.RandomSeed = 4242;
        a.Forward(Tensor<double>.CreateRandom(InputDepth, 8, 8));

        var b = MakeLayer();
        b.RandomSeed = 4242;
        b.Forward(Tensor<double>.CreateRandom(InputDepth, 8, 8));

        var wa = a.GetParameters();
        var wb = b.GetParameters();

        Assert.Equal(wa.Length, wb.Length);
        Assert.True(wa.Length > 4096, "the kernel must span multiple fill chunks for this to mean anything");
        for (int i = 0; i < wa.Length; i++)
        {
            if (wa[i] != wb[i])
            {
                Assert.Fail($"seeded init diverged at {i}: {wa[i]:R} vs {wb[i]:R}");
            }
        }

        var c = MakeLayer();
        c.RandomSeed = 9999;
        c.Forward(Tensor<double>.CreateRandom(InputDepth, 8, 8));
        Assert.NotEqual(wa.ToArray(), c.GetParameters().ToArray());
    }

    [Fact]
    public void ChunkedFill_ConsumesTheSameRandomStreamAsOneUnchunkedDraw()
    {
        // THE ACTUAL STREAM GUARD. The test above proves determinism (same seed twice agrees) but
        // would still pass if chunking had shifted the whole stream. This compares the layer's
        // weights against a SINGLE full-length draw from the same generator, which is what the
        // pre-chunking code did, so a chunk boundary that skipped or repeated draws fails here.
        //
        // The Kaiming bound is an unknown positive scale, so compare cross-multiplied ratios
        // instead of absolute values: w[i] = (raw[i]*2-1) * bound  =>  w[i]*r[0] == w[0]*r[i].
        const int seed = 4242;
        var layer = MakeLayer();
        layer.RandomSeed = seed;
        layer.Forward(Tensor<double>.CreateRandom(InputDepth, 8, 8));

        var w = layer.GetParameters();
        int kernelCount = OutputDepth * InputDepth * KernelSize * KernelSize;
        Assert.True(w.Length >= kernelCount, $"expected at least {kernelCount} kernel weights, got {w.Length}");
        Assert.True(kernelCount > 4096 * 3, "kernel must span several fill chunks for this to be meaningful");

        var raw = new double[kernelCount];
        new AiDotNet.Tensors.Helpers.SimdRandom(seed).NextDoubles(raw.AsSpan());

        double r0 = raw[0] * 2.0 - 1.0;
        double w0 = w[0];
        Assert.NotEqual(0.0, r0);

        for (int i = 0; i < kernelCount; i++)
        {
            double ri = raw[i] * 2.0 - 1.0;
            double lhs = w[i] * r0;
            double rhs = w0 * ri;
            double scale = Math.Max(Math.Abs(lhs), Math.Abs(rhs));
            if (scale > 0 && Math.Abs(lhs - rhs) / scale > 1e-12)
            {
                Assert.Fail(
                    $"weight {i} does not match a single unchunked draw of the same stream: "
                        + $"w[{i}]={w[i]:R}, expected proportional to {ri:R}. A chunk boundary has "
                        + "shifted, skipped or repeated random draws.");
            }
        }
    }

#if NET6_0_OR_GREATER
    // Thread-local, not GC.GetTotalAllocatedBytes: the process-wide counter picks up other tests'
    // allocations under xunit's parallel run and makes this flaky.
    [Fact]
    public void Materialization_AllocatesTheKernelsOnce()
    {
        var layer = MakeLayer();

        long before = GC.GetAllocatedBytesForCurrentThread();
        layer.Forward(Tensor<double>.CreateRandom(InputDepth, 8, 8));
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        long weightBytes = layer.ParameterCount * sizeof(double);
        Assert.True(weightBytes > 0, "layer should own parameters after materializing");

        // 1x the kernels plus activations for a small probe. The full-size staging array made the
        // weight portion alone 2x, which clears this bound comfortably.
        Assert.True(
            allocated < weightBytes * 17 / 10,
            $"materializing allocated {allocated:N0} bytes against {weightBytes:N0} bytes of weights "
                + $"({allocated / (double)weightBytes:N2}x). ~2x means InitializeWeights is staging "
                + "the whole kernel through a temporary array again.");
    }
#endif
}
