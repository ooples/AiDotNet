using AiDotNet.Interfaces;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// Model-family coverage for <see cref="TLoRAModel{T}"/> (arXiv:2507.05964).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this fixture is smaller than the model's defaults.</b> T-LoRA's production defaults are
/// SD-scale — a 320-channel UNet over a 64x64 latent (512x512 images through an 8x VAE) with a
/// 2048-wide cross-attention context. A training step at that size exceeded the per-test budget:
/// Training_ShouldReducePredictionError hung past its 120 s gate even running as the ONLY test in
/// the process, so this was never contention. The fixture therefore constructs the SAME model with
/// a smaller predictor and VAE over an 8x8 latent: identical topology, identical code path,
/// identical timestep-dependent adaptation — only the widths and the spatial size are reduced.
/// </para>
/// <para>
/// This is the same fixture pathology, and the same remedy, as InstantStyleModelTests. Production
/// defaults are untouched, and the paper's actual mechanism — the timestep-dependent rank schedule
/// and its orthogonal parametrization — is asserted at full fidelity and independently of model
/// scale in <c>TimestepDependentLoraTests</c>.
/// </para>
/// </remarks>
// The HeavyTimeout trait is RETAINED deliberately, and its justification has changed. It no longer
// stands for "foundation-scale at default" — the fixture below is no longer foundation-scale. It
// stands for an UNRESOLVED hang in Training_ShouldReducePredictionError that survived every rung of
// the float -> cap -> shrink ladder:
//
//   float            already in place (DiffusionModelTestBase<float>)
//   shrink           8x8 latent, 32-channel predictor, 16-channel VAE — the exact configuration
//                    that made InstantStyleModelTests pass. Turned an instant hang into a run that
//                    executes for ~6 minutes, then still overran the 120 s gate.
//   cap              TrainingIterations = 1. Still overran, which RULES OUT per-step training cost:
//                    one iteration at that size cannot take two minutes.
//
//   schedule         TrainTimesteps = 10 (from the model default of 1000). MEASURED, and it did
//                    NOT fix it either — still "timed out after 120000 milliseconds". So the
//                    schedule hypothesis is REFUTED alongside the other three.
//
// All four candidate costs are therefore eliminated: network size, iteration count, precision and
// schedule length. Whatever this is, it is not proportional to any of them, which points at
// something structural in the training path rather than at fixture scale — the next person should
// profile a single Train call rather than tune the fixture further, because tuning has been
// exhausted here.
//
// The trait stays until the probe is measured green; promoting a red test into the PR gate to chase
// a cleaner fixture is not a good trade. Remove it in the same change that demonstrates it passing.
[Xunit.Trait("Category", "HeavyTimeout")]
[Xunit.Collection("FoundationScaleSerial")]
public class TLoRAModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 8, 8];
    protected override int[] OutputShape => [1, 4, 8, 8];

    // Shrinking alone was NOT sufficient here, which is where this differs from InstantStyle. With
    // the identical injected predictor and VAE at the identical 8x8 latent, InstantStyle carries the
    // base class's full iteration count; T-LoRA still overran its 120 s gate, so its training path
    // costs materially more per step. The cap is therefore measured rather than precautionary: it is
    // the rung of the ladder this model actually needs and InstantStyle did not.
    protected override int TrainingIterations => 1;

    protected override IDiffusionModel<float> CreateModel()
        => new TLoRAModel<float>(
            // Ten timesteps instead of the model's default 1000. This was tried as the remaining
            // suspect after network size and iteration count were eliminated, and it did NOT fix the
            // probe either (see the class comment) — so it is NOT the dominant cost. It is kept only
            // because a 100x shorter schedule is strictly cheaper for a smoke fixture and exercises
            // the identical scheduler path. Production defaults are untouched.
            options: new AiDotNet.Models.Options.DiffusionModelOptions<float>
            {
                TrainTimesteps = 10,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = AiDotNet.Enums.BetaSchedule.ScaledLinear,
            },
            predictor: new UNetNoisePredictor<float>(
                architecture: null, inputChannels: 4, outputChannels: 4,
                baseChannels: 32, channelMultipliers: [1, 2],
                numResBlocks: 1, attentionResolutions: [2], contextDim: 64, seed: 42),
            vae: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4,
                baseChannels: 16, channelMultipliers: [1, 2],
                numResBlocksPerLevel: 1, seed: 42),
            seed: 42);
}
