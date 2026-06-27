using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ImageEditing;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// SDXLInpainting test scaffold — inherits from the generic
/// <see cref="DiffusionModelTestBase{TNum}"/> with <c>TNum = float</c> instead
/// of the default-double <see cref="DiffusionModelTestBase"/> shim because
/// SDXLInpainting at paper-scale defaults (UNet baseChannels=320,
/// channelMultipliers=[1,2,4], contextDim=2048, ≈2.6 B parameters; VAE
/// baseChannels=128 channelMultipliers=[1,2,4,4]) is too large for FP64 on a
/// 16 GB CI host — <c>SDXLInpaintingModel&lt;double&gt;</c> OOMs in the 1.28 B-element
/// kernel allocation during <c>UNetNoisePredictor.CreateDefaultEncoderBlocks</c>
/// (verified via testconsole/SdxlMemProfile). Even at FP32, the paper-scale
/// default is too slow for every model-family invariant because tests such as
/// determinism and input scaling run multiple <c>Predict</c> calls. This scaffold
/// keeps the defining SDXL-inpainting contracts (9-channel inpainting UNet,
/// 4-channel latents, scaled-linear 1000-step training schedule, 2048-dim SDXL
/// text context shape) while reducing width/resolution for CPU-bound invariants.
/// </summary>
/// <remarks>
/// FP32 is also the production-canonical numeric type for diffusion-model
/// weights — SD / SDXL / Flux / SD3 paper checkpoints are FP32 master / FP16
/// working precision. Testing against <see cref="IDiffusionModel{T}"/> with
/// <c>T = float</c> therefore mirrors the actual deployment configuration
/// rather than an FP64 test-only path whose memory cost and numerics would
/// silently diverge from any real SDXL pipeline. Per the
/// <see cref="AiDotNet.AiModelBuilder{T, TInput, TOutput}.ConfigureMixedPrecision"/>
/// contract, mixed-precision training is float-only too (the facade rejects
/// <c>T = double</c> at configure-time), so this is also the only numeric
/// type compatible with the documented production training path.
/// </remarks>
// HeavyTimeout (#1706): correct but too slow for the default per-test gate (foundation-scale diffusion,
// ~100 s/forward x N-step Generate); runs in the nightly lane. Drop this trait once it fits the budget.
[Xunit.Trait("Category", "HeavyTimeout")]
public class SDXLInpaintingModelTests : DiffusionModelTestBase<float>
{
    // SDXL inpainting denoises 4-channel latents. The UNet itself receives
    // 9 channels internally: latent + mask + masked-image latent.
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];
    protected override int TrainingIterations => 2;

    protected override IDiffusionModel<float> CreateModel()
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 9,
            outputChannels: 4,
            baseChannels: 32,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 1,
            attentionResolutions: [1, 2],
            contextDim: 2048,
            numHeads: 4,
            inputHeight: 32,
            seed: 42);

        var vae = new StandardVAE<float>(
            inputChannels: 3,
            latentChannels: 4,
            baseChannels: 32,
            channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 0.13025,
            seed: 42);

        return new SDXLInpaintingModel<float>(
            options: new DiffusionModelOptions<float>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear,
                DefaultInferenceSteps = 1
            },
            predictor: unet,
            vae: vae,
            seed: 42);
    }
}
