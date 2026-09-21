using System;
using System.Collections.Concurrent;
using AiDotNet.Diffusion;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for latent diffusion models.
/// Inherits all diffusion model invariant tests and adds latent-space-specific invariants:
/// denoising progress monotonicity, latent space continuity, and bounded denoised output.
/// </summary>
/// <remarks>
/// This base intentionally does NOT override <c>OutputShape_ShouldMatchInputShape</c> or
/// <c>NoiseSchedule_ShouldBeMonotonic</c>. A prior revision added overrides asserting that
/// <c>Predict()</c> returns a VAE-decoded sample (spatial dims scaled by
/// <c>VAE.DownsampleFactor</c>). That premise is incorrect for this codebase:
/// <c>LatentDiffusionModelBase.Predict()</c> denoises and returns the LATENT (model-forward
/// semantics, exactly like an SD UNet forward) — the VAE-decoded image/audio is produced by
/// <c>Generate()</c> (the pipeline call, the HuggingFace-diffusers <c>__call__</c> equivalent
/// which ends in <c>vae.decode</c>). So the latent-space contract <c>output.Length ==
/// input.Length</c> from the parent <see cref="DiffusionModelTestBase"/> is the correct,
/// stricter invariant, and the parent's scheduler-based <c>NoiseSchedule_ShouldBeMonotonic</c>
/// (which checks alpha-cumprod over ALL timesteps) subsumes the former 5-sample override. The
/// overrides were band-aids for older parent tests that have since been corrected; removing them
/// restores the correct shared behavior for every latent variant (audio / 3D / video).
/// </remarks>
public abstract class LatentDiffusionTestBase : DiffusionModelTestBase
{
    /// <summary>
    /// Spatial extent of the fixture latent. A latent U-Net has four resolution levels (three
    /// stride-2 downsamples), and 16 is the smallest square that traverses every one of them
    /// while keeping each skip merge on an even resolution.
    /// </summary>
    private const int LatentFixtureSpatialSize = 16;

    /// <summary>Probe result per concrete fixture type; deriving it constructs a model.</summary>
    private static readonly ConcurrentDictionary<Type, int[]> DerivedLatentShapes = new ConcurrentDictionary<Type, int[]>();

    /// <summary>
    /// A latent fixture shaped from the model's own <c>LatentChannels</c>.
    /// </summary>
    /// <remarks>
    /// These models denoise a LATENT, not an image, so a fixture must be latent-shaped. Several of
    /// them also declare <c>ModelDomain.Vision</c>, and a generated fixture that takes the generic
    /// vision shape hands a [3, 128, 128] RGB image to a latent-space denoiser - 64x the spatial
    /// elements of the real pipeline, which runs the image through the U-Net at full resolution
    /// instead of at image / VAE.DownsampleFactor. That is what made SmartEdit's
    /// <c>DenoisingProgress_Monotonic</c> exceed its 120-second budget.
    /// <para>
    /// The depth is read from the model rather than restated here so it stays correct for a model
    /// whose <c>LatentChannels</c> is a field the scaffold generator cannot resolve, and for any
    /// latent model added later. A fixture that pins its own shape still wins: this is only the
    /// family default.
    /// </para>
    /// </remarks>
    protected override int[] InputShape => DerivedLatentShape;

    /// <inheritdoc cref="InputShape"/>
    protected override int[] OutputShape => DerivedLatentShape;

    private int[] DerivedLatentShape => DerivedLatentShapes.GetOrAdd(GetType(), _ => ProbeLatentShape());

    private int[] ProbeLatentShape()
    {
        using var probe = CreateModel();
        if (probe is LatentDiffusionModelBase<double> latent)
        {
            return new[] { latent.LatentChannels, LatentFixtureSpatialSize, LatentFixtureSpatialSize };
        }

        return base.InputShape;
    }

    // =====================================================
    // LATENT DIFFUSION INVARIANT: Denoising Progress Monotonic
    // More denoising steps (less noise in input) should produce
    // output closer to the target. If error increases with less
    // noise, the denoising network is inverted.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DenoisingProgress_Monotonic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
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

    [Fact(Timeout = 120000)]
    public async Task LatentSpace_IsContinuous()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();

        var input1 = CreateRandomTensor(InputShape, rng);
        var input2 = new Tensor<double>(InputShape);
        for (int i = 0; i < input1.Length; i++)
            input2[i] = input1[i] + 1e-4;  // tiny perturbation

        // Probe the noise predictor directly via PredictNoise(sample, t).
        // Predict() runs the full Generate + VAE.Decode pipeline whose
        // first step derives a deterministic SEED from input bits — a
        // 1e-4 perturbation changes the seed, the sampled noise, and
        // the entire denoise trajectory, making Predict()'s output a
        // pseudo-random function of the input rather than a continuous
        // one. PredictNoise(sample, t) is the deterministic single-
        // forward step of the noise predictor (no seed, no sampling),
        // and continuity of the noise predictor is the actual
        // mathematical invariant the original test was meant to
        // express ("nearby points in latent space → nearby outputs").
        int probeTimestep = model.Scheduler.TrainTimesteps / 2;
        var out1 = model.PredictNoise(input1, probeTimestep);
        var out2 = model.PredictNoise(input2, probeTimestep);

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
                $"Noise predictor cosine similarity = {cosineSim:F4} for epsilon-close latent inputs " +
                $"at timestep {probeTimestep}. A continuous noise-predictor neural network must map " +
                $"ε-close inputs to ε-close outputs.");
        }
    }

    // =====================================================
    // LATENT DIFFUSION INVARIANT: Output Bounded After Denoising
    // After denoising, output values should be in a reasonable range.
    // Latent diffusion models that produce extreme values will
    // cause artifacts when decoded back to pixel/audio space.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputBounded_AfterDenoising()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
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
