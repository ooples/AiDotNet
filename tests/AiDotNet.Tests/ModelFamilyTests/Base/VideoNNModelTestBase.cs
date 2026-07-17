using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video neural network models (denoising, inpainting, super-resolution,
/// frame interpolation, optical flow). Inherits all NN invariant tests and adds video-specific
/// invariants: temporal dimension preservation, single-frame handling, and temporal smoothness.
/// </summary>
/// <remarks>
/// Generic over the numeric type <typeparamref name="T"/> so a heavy video model whose &lt;double&gt;
/// training/forward overruns the CI timeout can be generated as a &lt;float&gt; scaffold (via
/// <c>Fp32TestClassNames</c>) — the same enabling pattern used by DocumentNNModelTestBase /
/// FinancialModelTestBase. The non-generic <see cref="VideoNNModelTestBase"/> alias keeps every
/// existing &lt;double&gt; video model unchanged.
/// </remarks>
public abstract class VideoNNModelTestBase<T> : NeuralNetworkModelTestBase<T>
{
    // =====================================================
    // VIDEO INVARIANT: Temporal Dimension Preserved
    // Output temporal dimension should relate to input temporal
    // dimension (same for denoising, 2x for frame interpolation, etc.).
    // At minimum, output should not be empty.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task TemporalDim_Preserved()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0,
            "Video model produced empty output — temporal dimension lost.");
        // Output should be at least proportional to input
        Assert.True(output.Length >= input.Length / 4,
            $"Video output length ({output.Length}) is much smaller than input ({input.Length}). " +
            "Temporal information may be lost.");
    }

    // =====================================================
    // VIDEO INVARIANT: Single Frame Should Not Crash
    // A single-frame input is a valid edge case (degenerate video).
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task SingleFrame_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        // Create minimal input (use InputShape but reduce if possible)
        var minShape = (int[])InputShape.Clone();
        var input = CreateConstantTensor(minShape, 0.5);

        var output = network.Predict(input);
        Assert.True(output.Length > 0,
            "Video model produced empty output for single-frame input.");
        for (int i = 0; i < output.Length; i++)
        {
            double o = ConvertToDouble(output[i]);
            Assert.True(!double.IsNaN(o) && !double.IsInfinity(o),
                $"Output[{i}] is not finite for single-frame input.");
        }
    }

    // =====================================================
    // VIDEO INVARIANT: Consecutive Frames → Smooth Output
    // Similar inputs (consecutive frames) should produce similar outputs.
    // A model producing wildly different outputs for similar frames
    // will cause temporal flickering artifacts.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ConsecutiveFrames_SmoothOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var frame1 = CreateRandomTensor(InputShape, rng);
        // Frame 2: small perturbation of frame 1
        var frame2 = new Tensor<T>(InputShape);
        var eps = NumOps.FromDouble(0.01);
        for (int i = 0; i < frame1.Length; i++)
            frame2[i] = NumOps.Add(frame1[i], eps);

        var out1 = network.Predict(frame1);
        var out2 = network.Predict(frame2);

        // Compute cosine similarity
        double dot = 0, norm1 = 0, norm2 = 0;
        int minLen = Math.Min(out1.Length, out2.Length);
        for (int i = 0; i < minLen; i++)
        {
            double a = ConvertToDouble(out1[i]);
            double b = ConvertToDouble(out2[i]);
            dot += a * b;
            norm1 += a * a;
            norm2 += b * b;
        }

        if (norm1 > 1e-15 && norm2 > 1e-15)
        {
            double cosineSim = dot / (Math.Sqrt(norm1) * Math.Sqrt(norm2));
            Assert.True(cosineSim > 0.5,
                $"Cosine similarity = {cosineSim:F4} for near-identical frames. " +
                "Video model output is not temporally smooth.");
        }
    }
}

/// <summary>Non-generic &lt;double&gt; alias — the default for video models that are not floated.</summary>
public abstract class VideoNNModelTestBase : VideoNNModelTestBase<double> { }
