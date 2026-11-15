using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

public class SyncDreamerModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
    {
        var unet = new UNetNoisePredictor<float>(
            inputChannels: 4, outputChannels: 4,
            baseChannels: 64, channelMultipliers: [1, 2, 4],
            numResBlocks: 1, attentionResolutions: [1, 2],
            contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);

        var vae = new StandardVAE<float>(
            inputChannels: 3, latentChannels: 4,
            baseChannels: 32, channelMultipliers: [1, 2, 4],
            numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42);

        // A small configuration of the SyncDreamer-specific networks (CLIP, the volumes and the depth-wise attention),
        // so the parameter-surface tests do not materialize the paper-sized ViT-L/14 image encoder.
        var options = new SyncDreamerOptions
        {
            ImageSize = 64,
            CrossAttentionDim = 16,
            TimeEmbedDim = 8,
            TargetEncoderWidth = 8,
            TargetEncoderOutputDim = 4,
            NormGroups = 4,
            SpatialVolumeSize = 8,
            SpatialVolumeDims = [8, 8, 8, 8],
            FrustumVolumeDepth = 8,
            FrustumVolumeDims = [8, 8, 8, 8],
            DepthAttentionHeads = 2,
            ClipImageSize = 16,
            ClipPatchSize = 8,
            ClipWidth = 16,
            ClipLayers = 1,
            ClipHeads = 2,
            Seed = 42,
        };

        return new SyncDreamerModel<float>(options, unet: unet, vae: vae);
    }
}
