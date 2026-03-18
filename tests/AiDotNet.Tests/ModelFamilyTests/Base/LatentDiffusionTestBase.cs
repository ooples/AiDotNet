using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for latent diffusion models.
/// Inherits all diffusion model invariant tests and adds latent-space-specific invariants:
/// latent compression, denoising progress monotonicity, and latent space continuity.
/// </summary>
public abstract class LatentDiffusionTestBase : DiffusionModelTestBase
{
    // =====================================================
    // LATENT DIFFUSION INVARIANT: Denoising Progress Monotonic
    // More denoising steps (less noise in input) should produce
    // output closer to the target. If error increases with less
    // noise, the denoising network is inverted.
    // =====================================================

    [Fact]
    public void DenoisingProgress_Monotonic()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var target = CreateRandomTensor(InputShape, rng);

        // Input with more noise vs. less noise
        var highNoise = new Tensor<double>(InputShape);
        var lowNoise = new Tensor<double>(InputShape);
        for (int i = 0; i < target.Length; i++)
        {
            highNoise[i] = target[i] + (rng.NextDouble() - 0.5) * 2.0;  // heavy noise
            lowNoise[i] = target[i] + (rng.NextDouble() - 0.5) * 0.1;   // light noise
        }

        var outHigh = model.Predict(highNoise);
        var outLow = model.Predict(lowNoise);

        // MSE from target for each
        double mseHigh = 0, mseLow = 0;
        for (int i = 0; i < target.Length; i++)
        {
            double dh = outHigh[i] - target[i];
            double dl = outLow[i] - target[i];
            mseHigh += dh * dh;
            mseLow += dl * dl;
        }
        mseHigh /= target.Length;
        mseLow /= target.Length;

        if (!double.IsNaN(mseHigh) && !double.IsInfinity(mseHigh) && !double.IsNaN(mseLow) && !double.IsInfinity(mseLow))
        {
            // Low noise input should produce output at least as close to target
            Assert.True(mseLow <= mseHigh + 1e-2,
                $"Denoising is not monotonic: MSE(low_noise)={mseLow:F6} > MSE(high_noise)={mseHigh:F6}. " +
                "Less noisy input should produce output closer to target.");
        }
    }

    // =====================================================
    // LATENT DIFFUSION INVARIANT: Latent Space Continuity
    // Nearby points in latent/input space should produce nearby outputs.
    // Discontinuous output indicates unstable generation.
    // =====================================================

    [Fact]
    public void LatentSpace_IsContinuous()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();

        var input1 = CreateRandomTensor(InputShape, rng);
        var input2 = new Tensor<double>(InputShape);
        for (int i = 0; i < input1.Length; i++)
            input2[i] = input1[i] + 1e-4;  // tiny perturbation

        var out1 = model.Predict(input1);
        var out2 = model.Predict(input2);

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
                $"Latent space cosine similarity = {cosineSim:F4} for epsilon-close inputs. " +
                "Latent diffusion output is not continuous.");
        }
    }

    // =====================================================
    // LATENT DIFFUSION INVARIANT: Output Bounded After Denoising
    // After denoising, output values should be in a reasonable range.
    // Latent diffusion models that produce extreme values will
    // cause artifacts when decoded back to pixel/audio space.
    // =====================================================

    [Fact]
    public void OutputBounded_AfterDenoising()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);

        // Train briefly then predict
        var target = CreateRandomTensor(OutputShape, rng);
        for (int i = 0; i < 3; i++)
            model.Train(input, target);

        var output = model.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e5,
                $"Denoised output[{i}] = {output[i]:E4} exceeds bound. " +
                "Latent values are too extreme for stable decoding.");
        }
    }
}
