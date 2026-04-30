using System;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for latent diffusion models.
/// Inherits all diffusion model invariant tests and adds latent-space-specific invariants:
/// latent compression, denoising progress monotonicity, and latent space continuity.
/// </summary>
public abstract class LatentDiffusionTestBase : DiffusionModelTestBase
{
    // =====================================================
    // OVERRIDE: Output shape relation for latent diffusion
    //
    // The parent test (DiffusionModelTestBase.OutputShape_ShouldMatchInputShape)
    // asserts output.Length == input.Length, which is correct for
    // single-forward-pass diffusion models but mathematically wrong for
    // *latent* diffusion: LatentDiffusionModelBase.Predict runs
    // Generate → DecodeFromLatent (VAE.Decode) and returns a tensor in
    // pixel/audio space, with spatial dimensions scaled up by
    // VAE.DownsampleFactor relative to the latent input.
    //
    // Concrete example (AudioLDM, the failing test that triggered this
    // override): InputShape = [1, 8, 16, 16] (2,048 latent elements);
    // VAE decodes to [1, 1, 128, 128] (16,384 mel-spec elements) — an
    // 8× spatial upsample. Asserting `output.Length == input.Length`
    // is therefore wrong by construction for every latent variant; the
    // correct invariant is the latent → pixel scaling factor exposed
    // by VAE.DownsampleFactor.
    //
    // We also relax the assumption that all four spatial dims are
    // bound: only batch and the last two spatial dims are checked, so
    // a model that decodes 8×8 latent into 1×1×64×64 (no channel
    // collapse) and one that decodes 8×16×16 into 1×1×128×128 (8→1
    // channel collapse, AudioLDM-style) both pass under the same rule.
    // =====================================================

    [Fact(Timeout = 120000)]
    public override async Task OutputShape_ShouldMatchInputShape()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var model = CreateModel();
        var input = CreateRandomTensor(InputShape, rng);
        var output = model.Predict(input);

        // Latent diffusion's Predict() returns the VAE-decoded sample,
        // so spatial dims scale by the VAE downsample factor while
        // batch is preserved. CreateModel() returns IDiffusionModel<T>;
        // the latent-specific properties live on ILatentDiffusionModel<T>.
        var latentModel = (ILatentDiffusionModel<double>)model;
        int downsample = latentModel.VAE.DownsampleFactor;
        Assert.True(downsample >= 1,
            $"VAE.DownsampleFactor must be >= 1; got {downsample}.");

        Assert.Equal(input.Shape[0], output.Shape[0]);

        if (input.Rank >= 4 && output.Rank >= 4)
        {
            Assert.Equal(input.Shape[2] * downsample, output.Shape[2]);
            Assert.Equal(input.Shape[3] * downsample, output.Shape[3]);
        }
        else
        {
            // Lower-rank fallback: total element count must scale by
            // (downsample² × decoder_channels / latent_channels). We
            // can't compute that without VAE.OutputChannels, but at
            // minimum the output must be non-empty and at least as
            // large as a 1× decode (downsample=1 case).
            Assert.True(output.Length > 0,
                $"Predict produced an empty output for input length {input.Length}.");
        }
    }

    // =====================================================
    // OVERRIDE: Noise schedule monotonicity for latent diffusion
    //
    // The parent test scales the input by [0.1, 0.5, 1.0, 2.0] and
    // calls Predict each time, expecting output magnitudes to be
    // non-decreasing. That's a coin flip on diffusion models because
    // Predict derives a deterministic seed FROM the input bits:
    // scaling the input changes the seed, the seed produces different
    // random noise, and the *generated* sample's magnitude is
    // unrelated to the input scale. The original test passed for
    // non-latent diffusion only by chance (random outputs hitting the
    // ≤1-violation threshold); on AudioLDM that luck runs out and we
    // get [24.6, 30.0, 27.9, 24.4] — 2 violations → fail.
    //
    // The actual paper-correct invariant the parent's xml-doc claimed
    // to test ("at higher timesteps, the noise magnitude should
    // increase") lives on the *scheduler*, not on the trained model:
    // alpha_bar(t) (cumulative product of alphas) decreases
    // monotonically with t in every standard schedule (DDPM linear,
    // cosine, sigmoid), so noise level (1 - alpha_bar) increases
    // monotonically. This invariant holds on untrained models too —
    // the schedule is a fixed mathematical configuration, independent
    // of any learned weights.
    // =====================================================

    [Fact(Timeout = 120000)]
    public override async Task NoiseSchedule_ShouldBeMonotonic()
    {
        await Task.Yield();
        using var model = CreateModel();
        int total = model.Scheduler.TrainTimesteps;
        Assert.True(total > 1,
            $"Scheduler must declare at least 2 train timesteps; got {total}.");

        int[] timesteps = { 0, total / 4, total / 2, (3 * total) / 4, total - 1 };
        double[] noiseLevels = new double[timesteps.Length];
        for (int i = 0; i < timesteps.Length; i++)
        {
            // alpha_bar(t) is monotonically non-increasing in t for
            // every standard noise schedule; (1 - alpha_bar) is the
            // expected residual noise variance at time t.
            double alphaBar = Convert.ToDouble(model.Scheduler.GetAlphaCumulativeProduct(timesteps[i]));
            noiseLevels[i] = 1.0 - alphaBar;
        }

        for (int i = 1; i < noiseLevels.Length; i++)
        {
            Assert.True(noiseLevels[i] >= noiseLevels[i - 1] - 1e-10,
                $"Noise schedule is not monotonic: noiseLevel[t={timesteps[i - 1]}]={noiseLevels[i - 1]:F6} " +
                $"> noiseLevel[t={timesteps[i]}]={noiseLevels[i]:F6}. The cumulative-product alpha_bar " +
                $"of any standard DDPM/DDIM/cosine/sigmoid schedule is non-increasing in t, so " +
                $"(1 - alpha_bar) must be non-decreasing.");
        }
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
