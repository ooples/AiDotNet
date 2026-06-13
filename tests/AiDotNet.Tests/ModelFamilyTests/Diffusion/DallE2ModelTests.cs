using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Models.Options;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// DALL-E 2 test scaffold — see <see cref="DiffusionModelTestBase{TNum}"/>
/// for the FP32 rationale. The production defaults preserve the DALL-E 2
/// paper topology (CLIP-latent diffusion prior + GLIDE-style RGB decoder),
/// but the full default decoder is too large for every model-family invariant
/// to run on CPU. This scaffold injects a reduced prior/decoder/VAE with the
/// same contracts: 768-dim CLIP prior, RGB pixel-space decoder, time-conditioned
/// UNet blocks, attention, and the DALL-E 2 linear 1000-step training schedule.
/// </summary>
/// <remarks>
/// <c>DefaultInferenceSteps = 1</c>: these tests assert deterministic,
/// finite, shape-preserving denoising behavior, not sample quality. PyTorch
/// diffusers exposes <c>num_inference_steps</c> for exactly this reason:
/// smoke/invariant tests can run a short sampler while quality-sensitive
/// callers pass a larger step count to <c>Generate</c> / <c>GenerateFromText</c>.
/// </remarks>
public class DallE2ModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 3, 16, 16];
    protected override int[] OutputShape => [1, 3, 16, 16];
    protected override int TrainingIterations => 2;

    protected override IDiffusionModel<float> CreateModel()
    {
        var prior = new UNetNoisePredictor<float>(
            inputChannels: 768,
            outputChannels: 768,
            baseChannels: 32,
            channelMultipliers: [1, 2],
            numResBlocks: 1,
            attentionResolutions: [1],
            contextDim: 768,
            numHeads: 4,
            inputHeight: 2,
            seed: 42);

        var decoder = new UNetNoisePredictor<float>(
            inputChannels: 3,
            outputChannels: 3,
            baseChannels: 32,
            channelMultipliers: [1, 2, 4],
            numResBlocks: 1,
            attentionResolutions: [1, 2],
            contextDim: 768,
            numHeads: 4,
            inputHeight: 16,
            seed: 42);

        var vae = new StandardVAE<float>(
            inputChannels: 3,
            latentChannels: 3,
            baseChannels: 16,
            channelMultipliers: [1, 2],
            numResBlocksPerLevel: 1,
            latentScaleFactor: 1.0,
            seed: 42);

        return new DallE2Model<float>(
            options: new DiffusionModelOptions<float>
            {
                // Model defaults from DallE2Model's ctor, unchanged:
                TrainTimesteps = 1000,
                BetaStart = 0.0001,
                BetaEnd = 0.02,
                BetaSchedule = BetaSchedule.Linear,
                // Test-budget override (see class remarks):
                DefaultInferenceSteps = 1,
            },
            priorUnet: prior,
            decoderUnet: decoder,
            vae: vae,
            seed: 42);
    }
}
