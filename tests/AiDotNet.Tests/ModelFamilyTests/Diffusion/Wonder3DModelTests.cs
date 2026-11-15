using AiDotNet.Interfaces;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

/// <summary>
/// The generic diffusion contract on a tiny Wonder3D: one object's joint batch, its six normal views then its six
/// color views (12 latents of 4 channels at 8x8), denoised through the unconditional cross-domain branch.
/// </summary>
public class Wonder3DModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [12, 4, 8, 8];
    protected override int[] OutputShape => [12, 4, 8, 8];

    internal static Wonder3DOptions TinyOptions(int seed = 42) => new()
    {
        ImageSize = 16,
        BaseChannels = 32,
        ChannelMultipliers = [1, 2],
        ResBlocksPerLevel = 1,
        AttentionLevels = [0, 1],
        AttentionHeads = 4,
        CrossAttentionDim = 16,
        VaeBaseChannels = 32,
        VaeChannelMultipliers = [1, 2],
        VaeResBlocksPerLevel = 1,
        InferenceSteps = 2,
        ReconstructionIterations = 2,
        ReconstructionBatchSize = 32,
        SdfHiddenDim = 16,
        SdfHiddenLayers = 2,
        SdfSkipLayers = [],
        SdfFeatureDim = 8,
        SdfFrequencies = 2,
        ColorHiddenDim = 16,
        ColorHiddenLayers = 1,
        ViewFrequencies = 1,
        RaySamples = 8,
        ImportanceSamples = 0,
        UpSampleSteps = 1,
        MeshResolution = 12,
        Seed = seed,
    };

    protected override IDiffusionModel<float> CreateModel() => new Wonder3DModel<float>(TinyOptions());
}
