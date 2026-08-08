using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Foundation-scale-at-default: the model's full-scale default config has a Training peak (weights +
// gradients + Adam state + activations ~ 4x the ~1 GB SD/DiT-scale weights) that OOMs the 16 GB CI
// runner (verified via the CI logs — testhost/runner OOM at default scale; fits only on a larger box).
// Moved to the HeavyTimeout nightly lane so the default PR-gate shard fits and passes (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4)
public class CycleGANTurboModelTests : DiffusionModelTestBase<float>
{
    // Timeout ladder, rung 3. Rung 1 was already in place (this fixture has always been <float>),
    // and rung 2 -- the TrainingIterations cap below -- is applied here too, but neither can close
    // this one, because the cost is per-STEP and structural to the paper's objective:
    // CycleGAN-Turbo evaluates FOUR one-step SD-Turbo generator passes per training step (forward
    // and reverse generators, each run twice for the cycle-consistency term). Dividing the step
    // COUNT does not help when a single step is the problem.
    //
    // The evidence that this needed shrinking rather than another cap: with identical code it
    // PASSED one local run and, on the next, timed out at the 120 s gate and took the test host
    // down with it ("Test host process crashed"). A probe whose verdict flips with machine load is
    // not passing, it is winning a race.
    //
    // 16x16 latent instead of 64x64 is a 16x reduction in elements per pass, and the injected
    // predictor/VAE cut channel widths (320 -> 32, context 2048 -> 64, VAE 128 -> 16). Everything
    // STRUCTURAL survives: three UNet resolution levels, both attention resolutions, the
    // four-level VAE, and the full cycle-consistent path with all four generator passes. The test
    // still exercises this architecture rather than a different one.
    //
    // The model's own defaults are untouched -- the constructor already accepted an injected
    // predictor and VAE, so anyone constructing CycleGANTurboModel normally still gets the paper
    // configuration. Only this fixture is small.
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    // Rung 2, retained: the base runs 10 training iterations; four generator passes each makes
    // that 40 full forward/backward passes for one invariant.
    protected override int TrainingIterations => 2;

    protected override IDiffusionModel<float> CreateModel()
        => new CycleGANTurboModel<float>(
            predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<float>(
                inputChannels: 8, outputChannels: 4, baseChannels: 32,
                channelMultipliers: [1, 2, 4], numResBlocks: 1,
                attentionResolutions: [4, 2], contextDim: 64, inputHeight: 16, seed: 42),
            vae: new AiDotNet.Diffusion.VAE.StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 16,
                channelMultipliers: [1, 2, 4, 4], numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
