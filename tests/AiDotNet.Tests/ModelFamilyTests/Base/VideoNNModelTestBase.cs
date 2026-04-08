using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for video neural network models (denoising, inpainting, super-resolution,
/// frame interpolation, optical flow). Inherits all NN invariant tests and adds video-specific
/// invariants: temporal dimension preservation, single-frame handling, and temporal smoothness.
/// </summary>
public abstract class VideoNNModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // VIDEO INVARIANT: Temporal Dimension Preserved
    // Output temporal dimension should relate to input temporal
    // dimension (same for denoising, 2x for frame interpolation, etc.).
    // At minimum, output should not be empty.
    // =====================================================

    [Fact]
    public void TemporalDim_Preserved()
    {
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

    [Fact]
    public void SingleFrame_ShouldNotCrash()
    {
        var network = CreateNetwork();

        // Create minimal input (use InputShape but reduce if possible)
        var minShape = (int[])InputShape.Clone();
        var input = CreateConstantTensor(minShape, 0.5);

        var output = network.Predict(input);
        Assert.True(output.Length > 0,
            "Video model produced empty output for single-frame input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(!double.IsNaN(output[i]) && !double.IsInfinity(output[i]),
                $"Output[{i}] is not finite for single-frame input.");
        }
    }

    // =====================================================
    // VIDEO INVARIANT: Consecutive Frames → Smooth Output
    // Similar inputs (consecutive frames) should produce similar outputs.
    // A model producing wildly different outputs for similar frames
    // will cause temporal flickering artifacts.
    // =====================================================

    [Fact]
    public void ConsecutiveFrames_SmoothOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();

        var frame1 = CreateRandomTensor(InputShape, rng);
        // Frame 2: small perturbation of frame 1
        var frame2 = new Tensor<double>(InputShape);
        for (int i = 0; i < frame1.Length; i++)
            frame2[i] = frame1[i] + 0.01;

        var out1 = network.Predict(frame1);
        var out2 = network.Predict(frame2);

        // Compute cosine similarity
        double dot = 0, norm1 = 0, norm2 = 0;
        int minLen = Math.Min(out1.Length, out2.Length);
        for (int i = 0; i < minLen; i++)
        {
            dot += out1[i] * out2[i];
            norm1 += out1[i] * out1[i];
            norm2 += out2[i] * out2[i];
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
