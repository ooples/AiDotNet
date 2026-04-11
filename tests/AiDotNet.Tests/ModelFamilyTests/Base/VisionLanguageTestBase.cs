using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for vision-language models (CLIP, BLIP, LLaVA, etc.).
/// Inherits all NN invariant tests and adds VL-specific invariants:
/// image-only input, embedding diversity, bounded norms, and zero-image handling.
/// </summary>
public abstract class VisionLanguageTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // VISION-LANGUAGE INVARIANT: Image Input → Finite Output
    // The model should produce finite output from image-only input.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ImageOnly_ShouldProduceOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "VL model produced empty output for image input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(!double.IsNaN(output[i]) && !double.IsInfinity(output[i]),
                $"Output[{i}] is not finite — VL model failed on image input.");
        }
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Different Images → Different Embeddings
    // Distinct visual inputs must produce distinct representations.
    // A model mapping everything to the same embedding is collapsed.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentImages_DifferentEmbeddings()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        var img1 = CreateConstantTensor(InputShape, 0.1);
        var img2 = CreateConstantTensor(InputShape, 0.9);

        var emb1 = network.Predict(img1);
        var emb2 = network.Predict(img2);

        bool anyDifferent = false;
        int minLen = Math.Min(emb1.Length, emb2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(emb1[i] - emb2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "VL model produces identical embeddings for different images — representation has collapsed.");
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Output Norm Should Be Bounded
    // Embedding L2 norm should be reasonable. Extreme norms
    // indicate numerical instability or unbounded growth.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputNorm_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        double normSq = 0;
        for (int i = 0; i < output.Length; i++)
            normSq += output[i] * output[i];
        double norm = Math.Sqrt(normSq);

        Assert.True(norm < 1e4,
            $"Embedding L2 norm = {norm:E4} exceeds bound of 1e4. " +
            "VL model embedding is not well-bounded.");
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Zero Image Should Not Crash
    // All-black image is a valid edge case.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ZeroImage_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var blackImage = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(blackImage);
        Assert.True(output.Length > 0, "VL model produced empty output for black image.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN for all-zero image input.");
        }
    }
}
